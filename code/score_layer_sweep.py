"""층별 탐색 답변(layer_sweep.jsonl)에 KOTE 감정 점수(score_kote.TARGETS 전부)를 매긴다. 이미 매긴 줄은 건너뛴다.
예전 파일의 감정 열이 지금과 다르면 처음부터 다시 매긴다.

사용법:
    .venv\\Scripts\\python.exe scripts\\score_layer_sweep.py [--device cuda]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from score_kote import TARGETS, Scorer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "layer_sweep.jsonl"
OUT = ROOT / "results" / "kote_layer_sweep.csv"
KEEP = list(TARGETS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines()]
    done = 0
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = len(old) if list(old.columns[5:]) == KEEP else 0
        if not done:
            OUT.unlink()
    rows = rows[done:]
    if not rows:
        print("[끝] 새로 매길 답변 없음", flush=True)
        return
    df = Scorer(args.device).targets([r["text"] for r in rows])[KEEP].round(4)
    meta = pd.DataFrame([{k: r[k] for k in ("condition", "layer", "s", "qid", "collapsed2")} for r in rows])
    pd.concat([meta, df], axis=1).to_csv(OUT, mode="a", header=not done, index=False, encoding="utf-8")
    print(f"[끝] {OUT.name} {done + len(rows)}줄", flush=True)


if __name__ == "__main__":
    main()
