"""층별 측정 1단계 (LAYER_PLAN.md). 감정 벡터를 층 하나에 넣고 세기를 올려 가며 중립 질문 10개에 답하게 한다.

한 방향(감정·층·부호)에서 세기를 작은 것부터 올리다가 두 세기 연속 붕괴율(collapsed2)이 90% 이상이면 더 큰 세기는 건너뛴다.
건너뛴 칸은 layer_sweep_skips.jsonl에 남긴다. 멈춰도 다시 실행하면 이어서 한다.
층 순서는 짝수 층 먼저, 그다음 홀수 층이다. 중간에 멈춰도 층 전체의 모양을 볼 수 있게 하기 위해서다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_layer_sweep.py [--emotions rest]
    --emotions: 쉼표로 감정 이름을 주거나, rest(아직 안 잰 감정 전부)를 준다. 없으면 기쁨·화남·슬픔·가짜 벡터.
    무작위 벡터를 더 재려면 --emotions random1,random2,random3,random4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from collapse import collapsed2
from progress import Progress
from run_steering import SEED, make_hook, question_jobs
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "kakaocorp/kanana-2-3b-instruct"
OUT = RES / "layer_sweep.jsonl"
SKIPS = RES / "layer_sweep_skips.jsonl"
EMOTIONS = ["joy", "anger", "sadness", "random0"]
STRENGTHS = [round(0.1 * i, 1) for i in range(1, 11)] + [1.5, 2.0]
STOP_RATE = 0.9
STOP_RUN = 2
MAX_NEW = 128
RANDOMS = 5  # random0~random4. random1~4는 붕괴 비교를 위해 나중에 더했다(ANALYSIS_PLAN.md 2026-11-04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    d = np.load(RES / "layer_vectors_kanana_ko.npz")
    names = [str(n) for n in d["names"]]
    layers = [int(x) for x in d["layers"]]
    # 무작위 벡터는 run_steering.py와 같은 순서로 같은 시드에서 차례로 뽑는다(random0이 첫째, random1이 둘째).
    rng = np.random.default_rng(SEED)
    randoms = {}
    for i in range(RANDOMS):
        r = rng.standard_normal(d["centered"].shape[-1])
        randoms[f"random{i}"] = r / np.linalg.norm(r)
    if args.emotions == "rest":
        emotions = [n for n in names if n not in EMOTIONS]
    else:
        emotions = args.emotions.split(",") if args.emotions else EMOTIONS
    assert all(e in names + list(randoms) for e in emotions), emotions

    state = {"vec": None, "s": 0.0, "step": 0, "limit": 0}
    for k in layers:
        model.model.layers[k].register_forward_hook(make_hook(state, k, False))

    jobs = [j for j in question_jobs("ko") if j["kind"] == "neutral"]
    texts = [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for j in jobs]
    enc = tok(texts, return_tensors="pt", padding=True).to(0)

    saved = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            saved.setdefault((x["condition"], x["layer"], x["s"]), []).append(x)
    skipped = set()
    if SKIPS.exists():
        skipped = {(x["condition"], x["layer"], x["s"]) for x in map(json.loads, SKIPS.read_text(encoding="utf-8").splitlines())}

    order = [k for k in layers if k % 2 == 0] + [k for k in layers if k % 2 == 1]
    total = (len(emotions) * len(layers) * len(STRENGTHS) * 2 + 1) * len(jobs)
    bar = Progress("layer_sweep", total, f"층별 탐색: 감정 {len(emotions)} × 32층 × 세기 ±24")
    mine = lambda key: key[0] in emotions or key[0] == "base"
    count = sum(len(v) for k, v in saved.items() if mine(k)) + sum(mine(k) for k in skipped) * len(jobs)
    t0, made = time.time(), 0

    def run(cond, layer, s, vec):
        nonlocal count, made
        key = (cond, layer, s)
        if key in saved:
            return saved[key]
        state["vec"] = None if vec is None else {layer: torch.tensor(vec, device=0, dtype=torch.float32)}
        state["s"] = s
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        rows = []
        with OUT.open("a", encoding="utf-8") as f:
            for j, row in zip(jobs, gen[:, enc["input_ids"].shape[1] :]):
                text = tok.decode(row, skip_special_tokens=True).strip()
                rows.append({"condition": cond, "layer": layer, "s": s, "qid": j["qid"], "text": text, "collapsed2": collapsed2(text)})
                f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        saved[key] = rows
        count += len(rows)
        made += len(rows)
        rate = np.mean([x["collapsed2"] for x in rows])
        speed = made / (time.time() - t0)
        bar.update(count, {"조건": f"{cond} {layer}층 s={s}", "붕괴율": f"{rate:.0%}", "답변/초": f"{speed:.2f}"}, f"[{cond} L{layer} s={s}] {rows[0]['text'][:120]}")
        return rows

    run("base", -1, 0.0, None)
    for layer in order:
        table = {n: d["centered"][layers.index(layer)][i] for i, n in enumerate(names)}
        table.update(randoms)
        for cond in emotions:
            for sign in (1, -1):
                streak = 0
                for s in STRENGTHS:
                    key = (cond, layer, sign * s)
                    if key in skipped:
                        continue
                    if streak >= STOP_RUN:
                        with SKIPS.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"condition": cond, "layer": layer, "s": sign * s}) + "\n")
                        skipped.add(key)
                        count += len(jobs)
                        continue
                    rows = run(cond, layer, sign * s, table[cond])
                    streak = streak + 1 if np.mean([x["collapsed2"] for x in rows]) >= STOP_RATE else 0
    state["vec"] = None
    bar.finish()
    print(f"[끝] {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
