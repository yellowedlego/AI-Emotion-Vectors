"""저장된 활성값으로 감정 벡터를 계산하고, 층을 고르고, 감정 지도와 PCA를 만든다.

벡터는 두 가지로 계산한다.
- vs_neutral: 감정 평균 − 중립 평균. "감정적인 글투" 공통 성분이 섞인다.
- centered: 감정 평균 − 전체 감정의 평균. 공통 성분을 뺀다.
가짜 벡터(null): 중립 글을 소재별로 무작위 반으로 나눈 차이. 데이터에서 나온 방향이라 무작위 벡터보다 엄격한 대조군이다.

사용법:
    .venv\\Scripts\\python.exe scripts\\compute_vectors.py --tag kanana --source stories_ko
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FOLDS = 4
N_NULL = 20
SEED = 42
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def emotion_means(x, emo, names):
    return np.stack([x[emo == e].mean(0) for e in names])


def vectors(x, emo, names, kind):
    means = emotion_means(x, emo, names)
    if kind == "centered":
        return means - means.mean(0, keepdims=True)
    return means - x[emo == "neutral"].mean(0, keepdims=True)


def cv_auc(x, emo, topic, names, kind):
    """소재 단위 교차 검증: 학습 소재로 벡터를 만들고, 처음 보는 소재에서 감정을 가려내는지 본다."""
    ids = np.unique(topic)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    out = np.zeros((FOLDS, len(names)))
    for f, test_ids in enumerate(np.array_split(ids, FOLDS)):
        te = np.isin(topic, test_ids)
        v = unit(vectors(x[~te], emo[~te], names, kind))
        for i, e in enumerate(names):
            out[f, i] = roc_auc_score(emo[te] == e, x[te] @ v[i])
    return out.mean(0)


def null_vectors(x, emo, topic, rng):
    idx = np.where(emo == "neutral")[0]
    out = []
    for _ in range(N_NULL):
        perm = rng.permutation(idx)
        half = len(perm) // 2
        out.append(x[perm[:half]].mean(0) - x[perm[half:]].mean(0))
    return np.stack(out)


def heatmap(mat, names, title, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title)
    fig.colorbar(im, shrink=0.8, label="코사인 유사도")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="kanana")
    ap.add_argument("--source", default="stories_ko")
    ap.add_argument("--layer", type=int, default=None, help="층을 직접 정할 때. 없으면 교차 검증 AUC로 고른다")
    ap.add_argument("--suffix", default="", help="결과 파일 이름 뒤에 붙인다. 예: _L23")
    args = ap.parse_args()

    stem = RES / f"activations_{args.tag}_{args.source}"
    data = np.load(stem.with_suffix(".npz"))
    acts, layers = data["acts"], data["layers"]
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    emo = np.array([m["emotion"] for m in meta])
    topic = np.array([m["topic_id"] for m in meta])
    names = sorted(set(emo) - {"neutral"})
    name = f"{args.tag}_{args.source}{args.suffix}"

    rows = []
    for k, layer in enumerate(layers):
        auc = cv_auc(acts[:, k], emo, topic, names, "centered")
        rows.append({"layer": int(layer), "mean_auc": auc.mean(), "min_auc": auc.min()})
        print(f"층 {layer}: 평균 AUC {auc.mean():.3f}, 최저 {auc.min():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"layer_scan_{name}.csv", index=False)
    best = args.layer if args.layer is not None else int(pd.DataFrame(rows).sort_values("mean_auc").iloc[-1]["layer"])
    k = int(np.where(layers == best)[0][0])
    x = acts[:, k]
    print(f"\n[선택한 층] {best}", flush=True)

    v_n = vectors(x, emo, names, "vs_neutral")
    v_c = vectors(x, emo, names, "centered")
    rng = np.random.default_rng(SEED)
    nulls = null_vectors(x, emo, topic, rng)
    auc_c = cv_auc(x, emo, topic, names, "centered")
    auc_n = cv_auc(x, emo, topic, names, "vs_neutral")

    # 가짜 벡터로 감정을 가려내는 능력(우연이면 0.5 근처)과 크기
    null_auc = np.array([[roc_auc_score(emo == e, x @ unit(n)) for e in names] for n in nulls[:N_NULL]])
    table = pd.DataFrame(
        {
            "auc_centered": auc_c,
            "auc_vs_neutral": auc_n,
            "null_auc_max": null_auc.max(0),
            "norm_centered": np.linalg.norm(v_c, axis=1),
            "norm_null_mean": np.linalg.norm(nulls, axis=1).mean(),
        },
        index=names,
    ).sort_values("auc_centered", ascending=False)
    table.to_csv(RES / f"separation_{name}.csv")
    print(table.round(3).to_string(), flush=True)

    cos_n = unit(v_n) @ unit(v_n).T
    cos_c = unit(v_c) @ unit(v_c).T
    off = ~np.eye(len(names), dtype=bool)
    print(f"\n감정 쌍 평균 코사인: 중립 기준 {cos_n[off].mean():.3f}, 평균 빼기 {cos_c[off].mean():.3f}", flush=True)
    heatmap(cos_n, names, f"감정 벡터 유사도(중립 기준) · {name} · 층 {best}", RES / f"heatmap_vs_neutral_{name}.png")
    heatmap(cos_c, names, f"감정 벡터 유사도(평균 빼기) · {name} · 층 {best}", RES / f"heatmap_centered_{name}.png")

    # PCA: 이름을 붙이지 않은 축 찾기
    u, s, vt = np.linalg.svd(v_c, full_matrices=False)
    ratio = s**2 / (s**2).sum()
    coords = u * s
    pca = pd.DataFrame(coords[:, :3], index=names, columns=["PC1", "PC2", "PC3"])
    pca.to_csv(RES / f"pca_{name}.csv")
    print(f"\nPCA 설명 비율: {np.round(ratio[:5], 3)}", flush=True)
    for pc in ("PC1", "PC2"):
        o = pca[pc].sort_values()
        print(f"{pc} 한쪽 끝: {list(o.index[:4])} / 다른 쪽 끝: {list(o.index[-4:])}", flush=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pca["PC1"], pca["PC2"], color="#2f6fb5")
    for n, r in pca.iterrows():
        ax.annotate(n, (r["PC1"], r["PC2"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%})")
    ax.set_title(f"감정 지도(PCA) · {name} · 층 {best}")
    fig.tight_layout()
    fig.savefig(RES / f"pca_{name}.png", dpi=150)
    plt.close(fig)

    # 스티어링에 쓸 벡터 저장. 활성값 크기는 세기를 정할 때 쓴다.
    np.savez(
        RES / f"vectors_{name}.npz",
        names=np.array(names),
        centered=unit(v_c),
        vs_neutral=unit(v_n),
        nulls=unit(nulls),
        layer=best,
        mean_norm=np.linalg.norm(x, axis=1).mean(),
    )
    print(f"[끝] vectors_{name}.npz", flush=True)


if __name__ == "__main__":
    main()
