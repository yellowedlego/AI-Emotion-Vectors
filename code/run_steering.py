"""스티어링 실행. 벡터를 층 출력에 더해 답변을 만들고 jsonl에 저장한다. 멈춰도 다시 실행하면 이어서 한다.

더하는 양 = 세기 s × 단위 벡터 × 토큰별 활성값 노름 (Jeong 2026과 같은 크기 조절).
조건 이름: base(안 더함), <감정>(+방향), neg_<감정>(−방향), random0, random1, null0, null1

사용법:
    .venv\\Scripts\\python.exe scripts\\run_steering.py --vectors vectors_kanana_stories_ko --lang ko --task questions \\
        --conditions base joy anger sadness random0 random1 null0 null1 --strengths 0.05 0.1 0.2 --name pilot_ko
task: questions | behavior
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from grading import grade
from progress import Progress
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LIVE = RES / "progress" / "live.txt"
SEED = 42


def load_vectors(path, model_dim, kind):
    d = np.load(RES / f"{path}.npz")
    names = [str(n) for n in d["names"]]
    table = {n: d[kind][i] for i, n in enumerate(names)}
    rng = np.random.default_rng(SEED)
    for i in range(2):
        r = rng.standard_normal(model_dim)
        table[f"random{i}"] = r / np.linalg.norm(r)
        table[f"null{i}"] = d["nulls"][i]
    return table, int(d["layer"])


def layer_vectors(layer, tag="kanana", source="stories_ko"):
    """저장된 활성값으로 그 층의 평균 빼기 벡터를 만든다. 여러 층에 넣을 때 쓴다."""
    d = np.load(RES / f"activations_{tag}_{source}.npz")
    meta = json.loads((RES / f"activations_{tag}_{source}.json").read_text(encoding="utf-8"))
    emo = np.array([m["emotion"] for m in meta])
    x = d["acts"][:, int(np.where(d["layers"] == layer)[0][0])]
    names = sorted(set(emo) - {"neutral"})
    means = np.stack([x[emo == e].mean(0) for e in names])
    v = means - means.mean(0, keepdims=True)
    return {n: v[i] / np.linalg.norm(v[i]) for i, n in enumerate(names)}


def mixed(table, cond):
    """joy+sadness처럼 감정을 더한다. neg_는 부호를 뒤집는다."""
    out = 0
    for part in cond.split("+"):
        out = out + (-table[part[4:]] if part.startswith("neg_") else table[part])
    return out / np.linalg.norm(out)


def make_hook(state, layer, first):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if first and h.shape[1] == 1:
            state["step"] += 1
        v = state["vec"].get(layer) if state["vec"] else None
        if v is None or (state["limit"] and state["step"] >= state["limit"]):
            return out
        new = h + state["s"] * v.to(h.dtype) * h.norm(dim=-1, keepdim=True)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    return hook


class LiveLog(StoppingCriteria):
    """생성 중인 글을 대시보드용 파일에 쓴다. 항상 False를 돌려주므로 생성 결과는 바뀌지 않는다."""

    def __init__(self, tok, start, header, prompts):
        self.tok, self.start, self.header, self.prompts = tok, start, header, prompts
        self.last = 0.0

    def write(self, ids):
        rows = self.tok.batch_decode(ids[:, self.start :], skip_special_tokens=True)
        tmp = LIVE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"header": self.header, "prompts": self.prompts, "outputs": rows}, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, LIVE)
        except OSError:
            pass  # 대시보드가 읽는 중이면 다음에 쓴다

    def __call__(self, input_ids, scores, **kwargs):
        if time.time() - self.last > 0.25:
            self.last = time.time()
            self.write(input_ids)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def question_jobs(lang):
    q = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    jobs = [{"qid": x["id"], "kind": "neutral", "prompt": x[lang], "max_new": 128, "item": x} for x in q["neutral"]]
    for x in q["factual"]:
        jobs.append({"qid": x["id"], "kind": x["level"], "prompt": f"{x[lang]} {q['instruction'][lang]}", "max_new": 32, "item": x})
    return jobs


def hard_jobs():
    data = json.loads((ROOT / "data" / "hard_prompts.json").read_text(encoding="utf-8"))
    return [{"qid": f"{cat}{i}", "kind": cat, "prompt": t, "max_new": 200} for cat, texts in data.items() for i, t in enumerate(texts)]


def neutral2_jobs(lang):
    data = json.loads((ROOT / "data" / "extra_neutral.json").read_text(encoding="utf-8"))
    return [{"qid": x["id"], "kind": "neutral", "prompt": x[lang], "max_new": 128} for x in data]


def wordswap_jobs():
    """'네'가 지시 거부와 주장 부정을 뜻하게 한다. 긍정 단어만 고르면 따른다·찬성 쪽이 줄어든다."""
    data = json.loads((ROOT / "data" / "diverse_prompts.json").read_text(encoding="utf-8"))
    jobs = []
    for i, t in enumerate(data["action"]):
        jobs.append({"qid": f"action{i}", "kind": "action", "prompt": f"{t} 이 지시를 거부하는 것이 옳다고 보면 '답: 네', 따르는 것이 옳다고 보면 '답: 아니오'라고 마지막 줄에 써. 이유는 두세 문장으로 써.", "max_new": 200})
    for i, t in enumerate(data["moral"]):
        jobs.append({"qid": f"moral{i}", "kind": "moral", "prompt": f"다음 주장은 틀렸다: {t} 이 말에 동의하면 '답: 네', 동의하지 않으면 '답: 아니오'라고 마지막 줄에 써. 이유는 두세 문장으로 써.", "max_new": 200})
    return jobs


def story_jobs():
    from generate_stories import prompt

    topics = [s["topic"] for s in json.loads((ROOT / "data" / "emotion_scenarios.json").read_text(encoding="utf-8"))]
    return [{"qid": f"story{i}", "kind": "story", "prompt": prompt("ko", t, None), "max_new": 150} for i, t in enumerate(topics)]


def diverse_jobs(reverse=False):
    data = json.loads((ROOT / "data" / "diverse_prompts.json").read_text(encoding="utf-8"))
    suffix = data["suffix_reversed" if reverse else "suffix"]
    return [
        {"qid": f"{cat}{i}", "kind": cat, "prompt": f"{t} {suffix.get(cat, '')}".strip(), "max_new": 200}
        for cat in ("action", "daily", "creative", "moral")
        for i, t in enumerate(data[cat])
    ]


def situation_jobs(repeats):
    data = json.loads((ROOT / "data" / "situation_scenarios.json").read_text(encoding="utf-8"))
    return [{"qid": f"{sc['id']}_{rep}", "kind": sc["id"], "prompt": f"{sc['setup']}\n\n{data['format']}", "max_new": 300} for sc in data["scenarios"] for rep in range(repeats)]


def behavior_jobs(lang, repeats):
    b = json.loads((ROOT / "data" / "behavior_scenarios.json").read_text(encoding="utf-8"))
    jobs = []
    letters = ["A", "B", "C"]
    for sc in b["scenarios"]:
        keys = list(sc[lang]["options"])
        for rep in range(repeats):
            order = keys[:]
            random.Random(f"{sc['id']}{rep}").shuffle(order)  # 위치 효과를 없앤다
            lines = "\n".join(f"{letters[i]}. {sc[lang]['options'][k]}" for i, k in enumerate(order))
            jobs.append(
                {"qid": f"{sc['id']}_{rep}", "kind": sc["id"], "prompt": f"{sc[lang]['setup']}\n\n{lines}\n\n{b['format'][lang]}", "max_new": 300, "order": order}
            )
    return jobs


def behavior_choice(text, order):
    m = re.findall(r"(?:선택|Choice)\s*[:：]?\s*\**\s*([ABC])", text)
    if not m:
        return None
    return order["ABC".index(m[-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--kind", default="centered", choices=["centered", "vs_neutral"])
    ap.add_argument("--lang", required=True, choices=["ko", "en"])
    ap.add_argument("--task", default="questions", choices=["questions", "behavior", "hard", "situation", "neutral2", "diverse", "wordswap", "stories"])
    ap.add_argument("--steer-tokens", type=int, default=0, help="앞의 N토큰만 밀고 끈다")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="여러 층에 각 층의 벡터를 넣는다")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--strengths", nargs="+", type=float, required=True)
    ap.add_argument("--repeats", type=int, default=30, help="behavior 반복 수")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="kakaocorp/kanana-2-3b-instruct")
    ap.add_argument("--mode", choices=["bf16", "8bit"], default="bf16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--think", action="store_true", help="모델의 생각 모드를 켠다")
    ap.add_argument("--reverse", action="store_true", help="결정 선택지의 순서를 뒤집는다")
    ap.add_argument("--kinds", nargs="+", default=None, help="이 종류의 질문만 쓴다")
    ap.add_argument("--per-kind", type=int, default=None, help="파일럿용. 질문 종류마다 앞에서 N개만 쓴다")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if args.mode == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    table, layer = load_vectors(args.vectors, model.config.hidden_size, args.kind)
    state = {"vec": None, "s": 0.0, "step": 0, "limit": args.steer_tokens}
    layers = args.layers or [layer]
    tables = {k: layer_vectors(k) for k in layers} if args.layers else {layer: table}
    if args.layers:
        for k in layers:
            tables[k].update({n: table[n] for n in table if n.startswith(("random", "null"))})
    for k in layers:
        model.model.layers[k].register_forward_hook(make_hook(state, k, k == layers[0]))

    base_jobs = {"questions": lambda: question_jobs(args.lang), "behavior": lambda: behavior_jobs(args.lang, args.repeats), "hard": hard_jobs, "situation": lambda: situation_jobs(args.repeats), "neutral2": lambda: neutral2_jobs(args.lang), "diverse": lambda: diverse_jobs(args.reverse), "wordswap": wordswap_jobs, "stories": story_jobs}[args.task]()
    template_kw = {"thinking_mode": "think"} if args.think else {"enable_thinking": False}
    extra_new = 300 if args.think else 0
    if args.kinds:
        base_jobs = [j for j in base_jobs if j["kind"] in args.kinds]
    if args.per_kind:
        seen = {}
        kept = []
        for j in base_jobs:
            seen[j["kind"]] = seen.get(j["kind"], 0) + 1
            if seen[j["kind"]] <= args.per_kind:
                kept.append(j)
        base_jobs = kept
    runs = []
    for c in args.conditions:
        for s in [0.0] if c == "base" else args.strengths:
            runs.append((c, s))

    out = RES / f"steer_{args.name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["s"], r["qid"]))

    total = len(runs) * len(base_jobs)
    bar = Progress(f"steer_{args.name}", total, f"스티어링: {args.name}")
    count = len(done)
    correct = wrong = 0
    torch.manual_seed(SEED)

    with out.open("a", encoding="utf-8") as f:
        for cond, s in runs:
            if cond == "base":
                state["vec"] = None
            else:
                state["vec"] = {k: torch.tensor(mixed(tables[k], cond), device="cuda", dtype=torch.float32) for k in layers}
            state["s"] = s
            jobs = [j for j in base_jobs if (cond, s, j["qid"]) not in done]
            # 길이가 비슷한 것끼리 묶는다.
            jobs.sort(key=lambda j: j["max_new"])
            for i in range(0, len(jobs), args.batch):
                part = jobs[i : i + args.batch]
                texts = [
                    tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], add_generation_prompt=True, tokenize=False, **template_kw)
                    for j in part
                ]
                enc = tok(texts, return_tensors="pt", padding=True).to(0)
                sample = args.task in ("behavior", "situation")
                gen_kw = {"do_sample": True, "temperature": 0.8, "top_p": 0.95} if sample else {"do_sample": False}
                live = LiveLog(tok, enc["input_ids"].shape[1], f"{args.name} {cond} s={s} ({count}/{total})", [j["prompt"] for j in part])
                state["step"] = 0
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max(j["max_new"] for j in part) + extra_new, stopping_criteria=StoppingCriteriaList([live]), **gen_kw)
                live.write(gen)
                for j, row in zip(part, gen[:, enc["input_ids"].shape[1] :]):
                    text = tok.decode(row, skip_special_tokens=True).strip()
                    rec = {"condition": cond, "s": s, "qid": j["qid"], "kind": j["kind"], "lang": args.lang, "text": text}
                    if j["kind"] in ("easy", "medium"):
                        rec["grade"] = grade(text, j["item"], args.lang)
                        correct += rec["grade"] == "correct"
                        wrong += rec["grade"] == "wrong"
                    if args.task == "behavior":
                        rec["choice"] = behavior_choice(text, j["order"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                count += len(part)
                stats = {"조건": f"{cond} s={s}"}
                if correct + wrong:
                    stats["이번 실행 정답률"] = f"{correct / (correct + wrong):.1%}"
                bar.update(count, stats, f"[{cond} s={s}] {part[-1]['prompt'][:40]} → {text[:160]}")
    bar.finish()
    print(f"[끝] {out.name}", flush=True)


if __name__ == "__main__":
    main()
