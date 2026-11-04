"""영어 답변의 감정 점수. GoEmotions로 학습한 분류기를 쓴다.

--check: 정답을 아는 data/emotion_scenarios.json(영어 160문장)으로 분류기를 검증한다.
--input: 스티어링 결과 jsonl을 채점해 results/en_scores.csv로 저장한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\score_en.py --check
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "SamLowe/roberta-base-go_emotions"
TARGETS = {"joy": ["joy", "amusement", "excitement"], "anger": ["anger", "annoyance"], "sadness": ["sadness", "grief"], "neutral": ["neutral"]}


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def targets(self, texts, batch=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i : i + batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
                out.append(torch.sigmoid(self.model(**enc).logits).numpy())
        s = np.vstack(out)
        idx = {l: i for i, l in enumerate(self.labels)}
        return pd.DataFrame({e: s[:, [idx[l] for l in ls]].max(1) for e, ls in TARGETS.items()})


def check(sc):
    data = json.loads((ROOT / "data" / "emotion_scenarios.json").read_text(encoding="utf-8"))
    names = {"joy": "joy", "anger": "anger", "sad": "sadness", "neutral": "neutral"}
    texts, truth = [], []
    for s in data:
        for k, v in names.items():
            texts.append(s["en"][k])
            truth.append(v)
    df = sc.targets(texts)
    df["truth"] = truth
    print(df.groupby("truth").mean().loc[list(names.values())].round(2).to_string())
    pred = df[list(names.values())].idxmax(axis=1)
    print(f"[4지선다 정확도] 전체 {(pred == df['truth']).mean():.3f}")
    for v in names.values():
        m = df["truth"] == v
        print(f"  {v}: {(pred[m] == v).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--input")
    args = ap.parse_args()
    sc = Scorer()
    if args.check:
        check(sc)
    if args.input:
        rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if r["kind"] == "neutral"]
        df = sc.targets([r["text"] for r in rows])
        for k in ("s", "qid", "condition"):
            df.insert(0, k, [r[k] for r in rows])
        df.to_csv(ROOT / "results" / "en_scores.csv", index=False)


if __name__ == "__main__":
    main()
