"""대시보드의 분석 보기에 쓸 표를 만든다. 숲 그림, 세기별 곡선, 답변 이동 지도, 벡터 3D와 군집 트리.

사용법:
    .venv\\Scripts\\python.exe scripts\\view_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collapse import collapsed
from metrics import VECTORS, read_rows
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS, TSNE

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EMOTIONS = ["joy", "anger", "sadness"]
CONTROLS = ["random0", "random1", "null0", "null1"]
BOOT = 2000
SEED = 0
# 이름, KOTE 파일, 답변 파일, 기준선 파일. 기준선이 None이면 같은 파일의 base 행을 쓴다.
SETS = {"23층 질문 10개": ("steer_main_L23", "steer_base_ko"), "29층 질문 10개": ("steer_main_L29", "steer_base_ko"), "23층 질문 30개": ("steer_neutral2_L23", None), "23층 난감한 질문": ("steer_hard_L23", None)}


def load(name, base):
    df = pd.read_csv(RES / f"kote_{name}.csv")
    answers = read_rows(name)
    kept = {r["qid"] for r in answers if r["kind"] != "crisis"}
    b = df[df["condition"] == "base"] if base is None else pd.read_csv(RES / f"kote_{base}.csv")
    df = df[(df["condition"] != "base") & df["qid"].isin(kept)]
    b = b[b["qid"].isin(kept)].set_index("qid")
    bad = {(r["condition"], round(r["s"], 2), r["qid"]) for r in answers if collapsed(r["text"])}
    return df, b, bad


def boot(diff, rng):
    idx = rng.integers(0, len(diff), (BOOT, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def dose(df, b, bad, rng):
    ss = sorted(df["s"].unique())
    out = {"s": [float(s) for s in ss], "emotions": {}, "collapse": {}}
    for e in EMOTIONS:
        rows = {"mean": [], "lo": [], "hi": [], "control": []}
        for s in ss:
            g = df[(df["condition"] == e) & (df["s"] == s)]
            m, lo, hi = boot(g[e].values - b.loc[g["qid"], e].values, rng)
            ctrl = []
            for c in CONTROLS:
                h = df[(df["condition"] == c) & (df["s"] == s)]
                if len(h):
                    ctrl.append(boot(h[e].values - b.loc[h["qid"], e].values, rng)[2])
            for k, v in zip(rows, (m, lo, hi, max(ctrl) if ctrl else None)):
                rows[k].append(None if v is None else round(v, 4))
        out["emotions"][e] = rows
    for c in EMOTIONS + ["random0"]:
        rate = []
        for s in ss:
            g = df[(df["condition"] == c) & (df["s"] == s)]
            rate.append(round(np.mean([(c, round(float(s), 2), q) in bad for q in g["qid"]]), 3) if len(g) else None)
        out["collapse"][c] = rate
    return out


def moves(df, b, bad):
    cols = [c for c in df.columns if c not in ("condition", "s", "qid")]
    full = np.vstack([b[cols].values, df[cols].values])
    mean = full.mean(0)
    _, sv, vt = np.linalg.svd(full - mean, full_matrices=False)
    to_xy = lambda m: np.round((m - mean) @ vt[:2].T, 3)
    base_xy = to_xy(b[cols].values)
    pos = {q: i for i, q in enumerate(b.index)}
    conds = {}
    for (c, s), g in df.groupby(["condition", "s"]):
        xy = to_xy(g[cols].values)
        pts = [[q, float(x), float(y), int((c, round(float(s), 2), q) in bad)] for q, (x, y) in zip(g["qid"], xy)]
        dist = float(np.mean([np.hypot(x - base_xy[pos[q], 0], y - base_xy[pos[q], 1]) for q, x, y, _ in pts]))
        conds[f"{c}|{s:g}"] = {"points": pts, "dist": round(dist, 3)}
    axes = [sorted(zip(vt[k], cols)) for k in range(2)]
    return {"base": {q: [float(base_xy[i, 0]), float(base_xy[i, 1])] for q, i in pos.items()}, "conds": conds, "ratio": (sv[:2] ** 2 / (sv**2).sum()).round(3).tolist(), "axes": [{"neg": [n for _, n in a[:3]], "pos": [n for _, n in a[-3:][::-1]]} for a in axes]}


def forest(rng):
    out = []
    for label, tag in (("주 분석 23층 · 질문 10개 · s_max", "L23"), ("탐색 29층 · 질문 10개 · s_max", "L29")):
        t = pd.read_csv(RES / f"analysis_kote_{tag}.csv")
        out.append({"title": label, "rows": [{"e": r.emotion, "m": r.mean, "lo": r.ci95_lo, "hi": r.ci95_hi, "c": r.control_hi_max, "ok": bool(r.effect), "n": int(r.n_no_collapse), "t": 10} for r in t.itertuples()]})
    t = pd.read_csv(RES / "analysis_replicate.csv")
    for (name, s), g in t.groupby(["set", "s"], sort=False):
        out.append({"title": f"{name} · s={s:g}", "rows": [{"e": r.emotion, "m": r.mean, "lo": r.ci_lo, "hi": r.ci_hi, "c": r.control_hi, "ok": bool(r.effect), "col": r.collapse} for r in g.itertuples()]})
    df, b, bad = load(*SETS["23층 난감한 질문"])
    rows = []
    for e in EMOTIONS:
        d = lambda c: (lambda g: g[e].values - b.loc[g["qid"], e].values)(df[(df["condition"] == c) & (df["s"] == 0.5)])
        m, lo, hi = boot(d(e), rng)
        ctrl = max(boot(d(c), rng)[2] for c in ("random0", "null0"))
        g = df[(df["condition"] == e) & (df["s"] == 0.5)]
        rows.append({"e": e, "m": m, "lo": lo, "hi": hi, "c": ctrl, "ok": bool(lo > 0 and lo > ctrl), "col": float(np.mean([(e, 0.5, q) in bad for q in g["qid"]]))})
    out.append({"title": "난감한 질문 30개 · 위기 제외 · s=0.5", "rows": rows})
    return out


def vector_views():
    out = {}
    for label, name in VECTORS.items():
        d = np.load(RES / f"vectors_{name}.npz")
        names = [str(n) for n in d["names"]]
        m = d["centered"]
        u, s, _ = np.linalg.svd(m - m.mean(0), full_matrices=False)
        xyz = u[:, :3] * s[:3]
        dist = np.clip(1 - m @ m.T, 0, None)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=8, normalized_stress="auto").fit(dist)
        tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=5, random_state=SEED).fit_transform(dist)
        tree = dendrogram(linkage(squareform(dist, checks=False), "average"), no_plot=True, labels=names)
        out[label] = {"names": names, "ratio": (s[:3] ** 2 / (s**2).sum()).round(3).tolist(), "pca": np.round(xyz, 3).tolist(), "mds": np.round(mds.embedding_, 3).tolist(), "tsne": np.round(tsne, 2).tolist(), "tree": {"x": tree["icoord"], "h": np.round(tree["dcoord"], 3).tolist(), "leaves": tree["ivl"]}}
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {"forest": forest(rng), "dose": {}, "moves": {}, "vectors": vector_views()}
    for name, spec in SETS.items():
        df, b, bad = load(*spec)
        out["dose"][name] = dose(df, b, bad, rng)
        out["moves"][name] = moves(df, b, bad)
    (RES / "view_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print((RES / "view_data.json").stat().st_size // 1024, "KB")
    for name, m in out["moves"].items():
        print(name, m["ratio"], m["axes"], {k: v["dist"] for k, v in m["conds"].items() if k.endswith("|0.5")})
    for f in out["forest"]:
        print(f["title"], [(r["e"], round(r["m"], 3), r["ok"]) for r in f["rows"]])


if __name__ == "__main__":
    main()
