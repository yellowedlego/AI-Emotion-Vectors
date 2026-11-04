"""KOTE 학습 모델로 한국어 글의 감정 점수를 매긴다.

--check: 정답을 아는 data/emotion_scenarios.json(한국어 160문장)으로 분류기를 검증한다.
--input: 글이 든 jsonl(text, emotion 필드)을 채점해 감정별 평균을 낸다.

사용법:
    .venv\\Scripts\\python.exe scripts\\score_kote.py --check [--device cpu]
    .venv\\Scripts\\python.exe scripts\\score_kote.py --input results\\stories_ko.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = "searle-j/kote_for_easygoing_people"

# 우리 감정 → KOTE 라벨. 여러 개면 그중 최댓값을 쓴다.
# 근사 대응: love(아껴주는), desperation(절망), eogul(불평/불만)
# 대응 없음: loneliness, jealousy, dapdap
TARGETS = {
    "joy": ["즐거움/신남", "기쁨", "행복"],
    "anger": ["화남/분노", "짜증"],
    "sadness": ["슬픔"],
    "fear": ["공포/무서움", "불안/걱정"],
    "surprise": ["놀람", "경악"],
    "disgust": ["역겨움/징그러움", "증오/혐오"],
    "calm": ["편안/쾌적"],
    "love": ["아껴주는"],
    "gratitude": ["고마움"],
    "pride": ["뿌듯함"],
    "shame": ["부끄러움"],
    "guilt": ["죄책감"],
    "desperation": ["절망"],
    "curiosity": ["신기함/관심"],
    "boredom": ["재미없음", "귀찮음"],
    "eogul": ["불평/불만"],
    "seoreum": ["서러움"],
    "neutral": ["없음"],
}


class Scorer:
    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def scores(self, texts, batch=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i : i + batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                out.append(torch.sigmoid(self.model(**enc).logits).cpu().numpy())
        return np.vstack(out)

    def targets(self, texts):
        """글 수 × 감정 수 표. 칸마다 그 감정에 대응하는 KOTE 라벨의 최댓값."""
        s = self.scores(texts)
        idx = {l: i for i, l in enumerate(self.labels)}
        return pd.DataFrame({e: s[:, [idx[l] for l in ls]].max(1) for e, ls in TARGETS.items()})


def check(sc):
    data = json.loads((ROOT / "data" / "emotion_scenarios.json").read_text(encoding="utf-8"))
    names = {"joy": "joy", "anger": "anger", "sad": "sadness", "neutral": "neutral"}
    texts, truth = [], []
    for s in data:
        for k, v in names.items():
            texts.append(s["ko"][k])
            truth.append(v)
    df = sc.targets(texts)[list(names.values())]
    df["truth"] = truth
    print("[감정별 평균 점수] 행=정답, 열=분류기")
    print(df.groupby("truth").mean().loc[list(names.values())].round(2).to_string())
    pred = df[list(names.values())].idxmax(axis=1)
    print(f"\n[4지선다 정확도] 전체 {(pred == df['truth']).mean():.3f}")
    for v in names.values():
        m = df["truth"] == v
        print(f"  {v}: {(pred[m] == v).mean():.3f}")


def score_file(sc, path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]
    df = sc.targets([r["text"] for r in rows])
    df["emotion"] = [r["emotion"] for r in rows]
    table = df.groupby("emotion").mean()
    print("[감정별 평균 점수] 행=의도한 감정, 열=분류기")
    print(table.round(2).to_string())
    print("\n[의도한 감정의 점수가 가장 높은 글의 비율]")
    cols = list(TARGETS)
    pred = df[cols].idxmax(axis=1)
    for e in sorted(set(df["emotion"])):
        m = df["emotion"] == e
        rate = (pred[m] == e).mean() if e in TARGETS else float("nan")
        print(f"  {e}: {rate:.3f} (글 {m.sum()}개)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    sc = Scorer(args.device)
    if args.check:
        check(sc)
    if args.input:
        score_file(sc, args.input)


if __name__ == "__main__":
    main()
