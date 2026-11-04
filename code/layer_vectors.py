"""32층 전부의 감정 벡터를 계산한다. 층별 측정(LAYER_PLAN.md)의 0단계.

입력: results/activations_kanana_all_stories_ko.npz (extract_activations.py --tag kanana_all --first 0 --last 0.97)
출력:
- results/layer_vectors_kanana_ko.npz: 층마다 centered 단위 벡터, 가짜 벡터 2개, 활성값 평균 크기
- results/layer_scan_kanana_all_stories_ko.csv: 층마다 교차 검증 AUC와 가짜 벡터 AUC 최대값
기존 13~29층 활성값과 같은지도 확인한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\layer_vectors.py
"""
import json

import numpy as np
import pandas as pd
from compute_vectors import N_NULL, RES, SEED, cv_auc, null_vectors, unit, vectors
from sklearn.metrics import roc_auc_score


def main():
    data = np.load(RES / "activations_kanana_all_stories_ko.npz")
    acts, layers = data["acts"], data["layers"]
    meta = json.loads((RES / "activations_kanana_all_stories_ko.json").read_text(encoding="utf-8"))
    emo = np.array([m["emotion"] for m in meta])
    topic = np.array([m["topic_id"] for m in meta])
    names = sorted(set(emo) - {"neutral"})

    old = np.load(RES / "activations_kanana_stories_ko.npz")
    for k, layer in enumerate(old["layers"]):
        a, b = old["acts"][:, k], acts[:, int(np.where(layers == layer)[0][0])]
        gap = np.abs(a - b).max() / np.abs(a).max()
        if gap > 1e-2:
            raise SystemExit(f"층 {layer}: 기존 활성값과 다르다 (상대 차이 {gap:.4f})")
    print(f"기존 13~29층과 일치 확인", flush=True)

    rows, cen, nul, norm = [], [], [], []
    for k, layer in enumerate(layers):
        x = acts[:, k]
        auc = cv_auc(x, emo, topic, names, "centered")
        nulls = null_vectors(x, emo, topic, np.random.default_rng(SEED))
        null_auc = np.array([[roc_auc_score(emo == e, x @ unit(n)) for e in names] for n in nulls[:N_NULL]])
        rows.append({"layer": int(layer), "mean_auc": auc.mean(), "min_auc": auc.min(), "null_auc_max": null_auc.max(), **dict(zip(names, auc))})
        cen.append(unit(vectors(x, emo, names, "centered")))
        nul.append(unit(nulls[:2]))
        norm.append(np.linalg.norm(x, axis=1).mean())
        print(f"층 {layer}: 평균 AUC {auc.mean():.3f}, 최저 {auc.min():.3f}, 가짜 최대 {null_auc.max():.3f}", flush=True)

    pd.DataFrame(rows).to_csv(RES / "layer_scan_kanana_all_stories_ko.csv", index=False)
    np.savez(RES / "layer_vectors_kanana_ko.npz", names=np.array(names), layers=layers, centered=np.stack(cen), nulls=np.stack(nul), mean_norm=np.array(norm))
    print("[끝] layer_vectors_kanana_ko.npz", flush=True)


if __name__ == "__main__":
    main()
