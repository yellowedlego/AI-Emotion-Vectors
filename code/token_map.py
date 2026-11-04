"""글의 토큰마다 감정 방향과의 코사인 유사도를 계산해 색칠 페이지(results/token_map.html)를 만든다.

대상은 미리 정한 글 전체다. 골라 낸 사례가 아니다.
- 자연 활성 답변 40개
- 중립 질문 10개의 스티어링 전후 답변(base, 23층 joy·anger·sadness s=0.4)
스티어링 답변도 벡터를 더하지 않고 읽기만 한다. 값은 중립 상황 답변의 평균을 뺀 차이다.

사용법:
    .venv\\Scripts\\python.exe scripts\\token_map.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from natural_activation import VECTORS, load_vectors
from run_steering import behavior_jobs, hard_jobs, situation_jobs
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "kakaocorp/kanana-2-3b-instruct"
STEER_S = 0.4
MAX_TOKENS = 420
SHOW = ("base", "joy", "anger", "sadness", "desperation", "calm", "random0")
EXTRA = (
    ("steer_hard03_L23", "난감한 상황 s=0.3", 1, SHOW),
    ("steer_hard_L23", "난감한 상황 s=0.5", 1, SHOW),
    ("steer_situation_L23", "상황형", 2, SHOW),
    ("steer_situation_think_L23", "상황형 생각 모드", 3, SHOW),
    ("steer_behavior_L23", "행동", 2, SHOW),
    ("steer_behavior_think_L23", "행동 생각 모드", 3, SHOW),
)
THINK_END = re.compile(r"</think>|</생각>")


def texts():
    out = []
    for a in json.loads((RES / "natural_answers.json").read_text(encoding="utf-8")):
        if a["category"] != "behavior":
            out.append({"group": f"자연 활성: {a['category']}", "label": a["id"], "prompt": a["prompt"], "answer": a["text"]})
    q = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    prompts = {x["id"]: x["ko"] for x in q["neutral"]}
    rows = []
    for name in ("steer_base_ko", "steer_main_L23"):
        rows += [json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        keep = r["condition"] == "base" or (r["condition"] in ("joy", "anger", "sadness") and r["s"] == STEER_S)
        if keep and r["kind"] == "neutral":
            out.append({"group": f"스티어링: {r['condition']}", "label": r["qid"], "prompt": prompts[r["qid"]], "answer": r["text"]})
    jobs = {j["qid"]: j["prompt"] for j in hard_jobs() + situation_jobs(5) + behavior_jobs("ko", 30)}
    for name, title, reps, kinds in EXTRA:
        path = RES / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rep = re.search(r"_(\d+)$", r["qid"])
            if r["kind"] != "crisis" and int(rep[1] if rep else 0) < reps and r["condition"] in kinds:
                out.append({"group": f"{title}: {r['condition']}", "label": r["qid"], "prompt": jobs[r["qid"]], "answer": r["text"]})
    return out


def score(model, tok, item, layers, vectors):
    prefix = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    full = prefix + item["answer"]
    enc = tok(full, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    spans = enc.pop("offset_mapping")[0].tolist()
    got = {}
    handles = []
    for k in layers:
        handles.append(model.model.layers[k].register_forward_hook(lambda _m, _i, out, k=k: got.__setitem__(k, (out[0] if isinstance(out, tuple) else out)[0].float())))
    with torch.no_grad():
        model(**enc.to(0))
    for h in handles:
        h.remove()

    start = len(prefix)
    picked = [i for i, (a, b) in enumerate(spans) if a >= start and b > a][:MAX_TOKENS]
    # 토큰 사이 공백과 줄바꿈이 빠지지 않게 다음 토큰 시작까지 잘라 쓴다.
    words = [full[spans[i][0] : spans[picked[n + 1]][0] if n + 1 < len(picked) else spans[i][1]] for n, i in enumerate(picked)]
    cos = {}
    for k in layers:
        h = got[k][picked]
        cos[k] = ((h / (h.norm(dim=-1, keepdim=True) + 1e-9)) @ vectors[k][1].T).cpu().numpy()
    return words, cos


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    layers = list(VECTORS)
    vectors = {k: load_vectors(k, "cuda") for k in layers}
    names = [n for n in vectors[layers[0]][0] if not n.startswith("null")]

    items = texts()
    results = [score(model, tok, x, layers, vectors) for x in items]
    base = {}
    for k in layers:
        rows = [c[k] for x, (_w, c) in zip(items, results) if x["group"] == "자연 활성: neutral"]
        base[k] = np.vstack(rows).mean(0)

    data = {"emotions": names, "layers": layers, "items": [], "scale": {}}
    for x, (words, cos) in zip(items, results):
        m = THINK_END.search("".join(words))
        think_end = None
        if m:
            pos = 0
            for n, w in enumerate(words):
                pos += len(w)
                if pos >= m.end():
                    think_end = n + 1
                    break
        row = {"group": x["group"], "label": x["label"], "prompt": x["prompt"], "tokens": words, "think_end": think_end, "scores": {}}
        for k in layers:
            d = cos[k][:, : len(names)] - base[k][: len(names)]
            row["scores"][str(k)] = {n: [round(float(v) * 1000) for v in d[:, i]] for i, n in enumerate(names)}
        data["items"].append(row)
    for k in layers:
        vals = np.concatenate([np.abs(cos[k][:, : len(names)] - base[k][: len(names)]).ravel() for _w, cos in results])
        data["scale"][str(k)] = round(float(np.percentile(vals, 98)) * 1000)

    (RES / "token_map.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    page = (ROOT / "scripts" / "token_map.html").read_text(encoding="utf-8")
    page = page.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (RES / "token_map.html").write_text(page, encoding="utf-8")
    print(f"[끝] token_map.html 글 {len(items)}개", flush=True)


if __name__ == "__main__":
    main()
