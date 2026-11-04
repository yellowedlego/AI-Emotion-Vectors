"""스티어링 답변의 KOTE 점수를 results/kote_<이름>.csv에 쌓는다. 이미 채점한 답변은 건너뛴다.

정답 질문은 답이 짧아 감정 점수를 내지 않는다.

사용법:
    .venv\\Scripts\\python.exe scripts\\kote_scores.py steer_main_L23
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from score_kote import TARGETS, Scorer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    sc = Scorer(args.device)
    for name in args.names + ["steer_base_ko"]:
        out = RES / f"kote_{name}.csv"
        old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["condition", "s", "qid"])
        seen = set(zip(old["condition"], old["s"], old["qid"]))
        rows = [json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if r["kind"] not in ("easy", "medium") and (r["condition"], r["s"], r["qid"]) not in seen]
        if not rows:
            continue
        scores = sc.targets([r["text"] for r in rows])[list(TARGETS)]
        scores.insert(0, "qid", [r["qid"] for r in rows])
        scores.insert(0, "s", [r["s"] for r in rows])
        scores.insert(0, "condition", [r["condition"] for r in rows])
        pd.concat([old, scores]).to_csv(out, index=False)
        print(f"[끝] {out.name} 새 답변 {len(rows)}개", flush=True)


if __name__ == "__main__":
    main()
