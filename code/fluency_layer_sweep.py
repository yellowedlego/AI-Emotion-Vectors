"""층별 탐색 답변이 말이 되는지 본다. 벡터를 넣지 않은 원래 모델로 답변 토큰마다 놀람(음의 로그 확률)을 재서 평균 낸다.

뜻이 깨진 답변은 원래 모델이 예상하지 못하므로 값이 커진다. 같은 말 반복은 오히려 값이 작아지므로 collapsed2와 함께 쓴다.
이미 잰 줄은 건너뛴다.

--generic: 실제 질문 대신 상관없는 질문 뒤에 답변을 붙여 잰다. 두 값의 차이로 답변이 질문에 맞춘 것인지 본다.

사용법:
    .venv\\Scripts\\python.exe scripts\\fluency_layer_sweep.py [--generic]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from progress import Progress
from run_steering import question_jobs
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "layer_sweep.jsonl"
OUT = ROOT / "results" / "fluency_layer_sweep.csv"
OUT_GENERIC = ROOT / "results" / "fluency_generic_layer_sweep.csv"
# 질문과 상관없는 요청. 실제 질문 뒤에서 답변이 훨씬 덜 놀라우면 그 질문에 맞춘 답변이다.
GENERIC = "아무 이야기나 해 줘."
MODEL = "kakaocorp/kanana-2-3b-instruct"
BATCH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", action="store_true", help="실제 질문 대신 GENERIC 질문 뒤에 답변을 붙여 잰다")
    args = ap.parse_args()
    out_path = OUT_GENERIC if args.generic else OUT
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    prompts = {j["qid"]: tok.apply_chat_template([{"role": "user", "content": GENERIC if args.generic else j["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for j in question_jobs("ko") if j["kind"] == "neutral"}
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines()]
    done = len(pd.read_csv(out_path)) if out_path.exists() else 0
    out = []
    bar = Progress("fluency_generic" if args.generic else "fluency", len(rows), "자연스러움 채점" + (" (상관없는 질문 뒤)" if args.generic else ""))
    for i in range(done, len(rows), BATCH):
        part = rows[i : i + BATCH]
        heads = [tok(prompts[r["qid"]], add_special_tokens=False)["input_ids"] for r in part]
        bodies = [tok(r["text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for r in part]
        width = max(len(h) + len(b) for h, b in zip(heads, bodies))
        ids = torch.full((len(part), width), tok.pad_token_id or 0)
        mask = torch.zeros_like(ids)
        for k, (h, b) in enumerate(zip(heads, bodies)):
            ids[k, : len(h) + len(b)] = torch.tensor(h + b)
            mask[k, : len(h) + len(b)] = 1
        with torch.no_grad():
            hidden = model.model(input_ids=ids.to(0), attention_mask=mask.to(0)).last_hidden_state
        for k, (h, b) in enumerate(zip(heads, bodies)):
            # 어휘 전체 확률은 답변 위치에서만 계산해 메모리를 아낀다.
            pos = torch.arange(len(h) - 1, len(h) + len(b) - 1, device=0)
            with torch.no_grad():
                logp = torch.log_softmax(model.lm_head(hidden[k, pos]).float(), -1)
            nll = -logp[torch.arange(len(b), device=0), torch.tensor(b, device=0)]
            out.append({"row": i + k, "nll": round(float(nll.mean()), 4), "nll_max20": round(float(nll.topk(min(20, len(b))).values.mean()), 4), "tokens": len(b)})
        if len(out) >= 2000 or i + BATCH >= len(rows):
            pd.DataFrame(out).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            print(f"{i + len(part)}/{len(rows)}", flush=True)
            bar.update(i + len(part))
            out = []
    bar.finish()
    print(f"[끝] {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
