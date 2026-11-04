"""자연 활성 측정. 벡터를 더하지 않고, 상황 글에 답할 때 내부 값이 감정 방향으로 기우는지 읽는다.

토큰별 활성값과 감정 벡터의 코사인 유사도를 입력 구간과 답변 구간에서 따로 평균한다.
기준은 ANALYSIS_PLAN.md의 "자연 활성 측정"을 따른다.

사용법:
    .venv\\Scripts\\python.exe scripts\\natural_activation.py
    .venv\\Scripts\\python.exe scripts\\natural_activation.py --limit 2 --name smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from progress import Progress
from run_steering import behavior_jobs
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "kakaocorp/kanana-2-3b-instruct"
VECTORS = {23: "vectors_kanana_stories_ko_L23", 29: "vectors_kanana_stories_ko"}
PAIRS = {"praise": "joy", "hostile": "anger", "loss": "sadness", "pressure": "desperation"}
BATCH = 8
MAX_NEW = 96
BOOT = 10000
SEED = 42


def load_items(limit):
    data = json.loads((ROOT / "data" / "natural_scenarios.json").read_text(encoding="utf-8"))
    items = []
    for cat, texts in data.items():
        for i, t in enumerate(texts[:limit]):
            items.append({"id": f"{cat}{i}", "category": cat, "prompt": t})
    for j in behavior_jobs("ko", 1):
        items.append({"id": j["kind"], "category": "behavior", "prompt": j["prompt"]})
    return items


def load_vectors(layer, device):
    d = np.load(RES / f"{VECTORS[layer]}.npz")
    names = [str(n) for n in d["names"]] + [f"null{i}" for i in range(len(d["nulls"]))]
    mat = np.vstack([d["centered"], d["nulls"]])
    return names, torch.tensor(mat, device=device, dtype=torch.float32)


def cosine(h, vecs):
    """h: 토큰 수 × 차원. 결과: 토큰 수 × 벡터 수. vecs는 단위 벡터."""
    h = h.float()
    return (h / (h.norm(dim=-1, keepdim=True) + 1e-9)) @ vecs.T


def run(model, tok, items, layers, vectors, bar):
    state = {k: {"prompt": None, "steps": []} for k in layers}

    def make_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if state[k]["prompt"] is None:
                state[k]["prompt"] = h.detach()
            else:
                state[k]["steps"].append(h[:, -1].detach())

        return hook

    for k in layers:
        model.model.layers[k].register_forward_hook(make_hook(k))

    rows, answers = [], []
    items = sorted(items, key=lambda x: len(x["prompt"]))
    for i in range(0, len(items), BATCH):
        part = items[i : i + BATCH]
        texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for x in part]
        enc = tok(texts, return_tensors="pt", padding=True).to(0)
        for k in layers:
            state[k] = {"prompt": None, "steps": []}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        start = enc["input_ids"].shape[1]
        new = gen[:, start:]
        specials = torch.tensor(sorted(tok.all_special_ids), device=new.device)
        for n, x in enumerate(part):
            text = tok.decode(new[n], skip_special_tokens=True).strip()
            answers.append({**x, "text": text})
            for k in layers:
                names, vecs = vectors[k]
                p = cosine(state[k]["prompt"][n][enc["attention_mask"][n].bool()], vecs).mean(0)
                # k번째 스텝의 활성값은 그 앞 스텝에서 만든 토큰 자리다.
                steps = torch.stack(state[k]["steps"], dim=1)[n]
                toks = new[n][: steps.shape[0]]
                keep = ~torch.isin(toks, specials)
                g = cosine(steps[keep], vecs).mean(0) if keep.any() else torch.full_like(p, float("nan"))
                for window, val in (("prompt", p), ("answer", g)):
                    for name, v in zip(names, val.tolist()):
                        rows.append({"id": x["id"], "category": x["category"], "layer": k, "window": window, "vector": name, "cos": v})
        bar.update(min(i + BATCH, len(items)), {}, f"{part[-1]['id']} → {text[:160]}")
    return pd.DataFrame(rows), answers


def boot_ci(a, b, rng):
    ia = rng.integers(0, len(a), (BOOT, len(a)))
    ib = rng.integers(0, len(b), (BOOT, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return np.percentile(d, [2.5, 97.5])


def summarize(df):
    rng = np.random.default_rng(SEED)
    out = []
    df = df[df["category"] != "behavior"]
    for (layer, window), g in df.groupby(["layer", "window"]):
        wide = g.pivot(index="id", columns="vector", values="cos")
        cat = g.groupby("id")["category"].first()
        base = wide[cat[wide.index] == "neutral"]
        emos = [c for c in wide.columns if not c.startswith("null")]
        nulls = [c for c in wide.columns if c.startswith("null")]
        for c, target in PAIRS.items():
            part = wide[cat[wide.index] == c]
            if len(part) < 2:
                continue
            diff = part.mean() - base.mean()
            lo, hi = boot_ci(part[target].values, base[target].values, rng)
            null_max = diff[nulls].abs().max()
            out.append(
                {
                    "layer": layer,
                    "window": window,
                    "category": c,
                    "target": target,
                    "diff": diff[target],
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "null_max": null_max,
                    "rank_of_20": int((diff[emos] > diff[target]).sum()) + 1,
                    "top3": ", ".join(diff[emos].sort_values(ascending=False).index[:3]),
                    "pass": (lo > 0 or hi < 0) and abs(diff[target]) > null_max,
                }
            )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="범주마다 앞에서 N개만 쓴다")
    ap.add_argument("--name", default="natural")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    layers = list(VECTORS)
    vectors = {k: load_vectors(k, "cuda") for k in layers}
    items = load_items(args.limit)
    bar = Progress(args.name, len(items), "자연 활성 측정")
    df, answers = run(model, tok, items, layers, vectors, bar)
    bar.finish()

    df.to_csv(RES / f"{args.name}_raw.csv", index=False)
    (RES / f"{args.name}_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    table = summarize(df)
    table.to_csv(RES / f"{args.name}_summary.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
