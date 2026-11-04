"""대시보드용 수치 집계. 결과 파일을 읽어 JSON으로 쓸 표를 만든다. 파일이 바뀌면 다시 계산한다."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from collapse import BROKEN_MAX20, collapsed, ignored

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
RUNS = {"23층": "steer_main_L23", "29층": "steer_main_L29"}
EXTRA_RUNS = {"23층": ["steer_others_L23"]}  # 같은 질문에 나머지 17개 감정(세기 0.2, 0.4)
VECTORS = {"23층": "kanana_stories_ko_L23", "29층": "kanana_stories_ko"}
_cache = {}


def cached(key, paths, fn):
    stamp = tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = fn()
    _cache[key] = (stamp, value)
    return value


def read_rows(name):
    path = RES / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cell_stats(rows):
    facts = [r for r in rows if r["kind"] in ("easy", "medium")]
    neutral = [r for r in rows if r["kind"] == "neutral"]
    return {
        "n": len(rows),
        "붕괴율": np.mean([collapsed(r["text"]) for r in rows]),
        "정답률": np.mean([r["grade"] == "correct" for r in facts]) if facts else None,
        "쉬운문제": np.mean([r["grade"] == "correct" for r in facts if r["kind"] == "easy"]) if facts else None,
        "중간문제": np.mean([r["grade"] == "correct" for r in facts if r["kind"] == "medium"]) if facts else None,
        "이모티콘": np.mean([len(EMOJI.findall(r["text"])) for r in neutral]) if neutral else None,
        "글자수": np.mean([len(r["text"]) for r in neutral]) if neutral else None,
    }


def kote_table(name):
    path = RES / f"kote_{name}.csv"
    base = RES / "kote_steer_base_ko.csv"
    if not path.exists() or not base.exists():
        return {}
    df = pd.read_csv(path)
    b = pd.read_csv(base).set_index("qid")
    emos = [e for e in df.columns[3:] if e != "neutral" and e in b.columns]
    out = {}
    for (cond, s), g in df.groupby(["condition", "s"]):
        d = {f"Δ{e}": float((g[e].values - b.loc[g["qid"], e].values).mean()) for e in emos}
        own = cond.replace("neg_", "")
        if own in emos:
            d["Δ목표"] = d[f"Δ{own}"]  # 넣은 감정 자신의 점수 변화
        out[f"{cond}|{s:g}"] = d
    return out


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


def steering():
    base_rows = read_rows("steer_base_ko")

    def build():
        out = {"base": cell_stats(base_rows), "runs": {}}
        for label, name in RUNS.items():
            names = [name] + EXTRA_RUNS.get(label, [])
            groups = defaultdict(list)
            for n in names:
                for r in read_rows(n):
                    groups[f"{r['condition']}|{r['s']:g}"].append(r)
            cells = {k: cell_stats(v) for k, v in groups.items()}
            for n in names:
                for k, d in kote_table(n).items():
                    if k in cells:
                        cells[k].update(d)
            out["runs"][label] = cells
        return clean(out)

    names = list(RUNS.values()) + [n for v in EXTRA_RUNS.values() for n in v]
    paths = [RES / f"{n}.jsonl" for n in names] + [RES / f"kote_{n}.csv" for n in names]
    return cached("steering", paths, build)


def eta():
    path = RES / "eta_model.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def neuron_map():
    path = RES / "neuron_map.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summary():
    return json.loads((ROOT / "data" / "summary.json").read_text(encoding="utf-8"))


def fidelity(dist, xy):
    """지도의 거리가 실제 거리 순서를 얼마나 따르는지, 가장 가까운 3개 이웃이 얼마나 남는지."""
    from scipy.stats import spearmanr

    iu = np.triu_indices(len(dist), 1)
    flat = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    near = lambda m: np.argsort(m + np.eye(len(m)) * 9, 1)[:, :3]
    a, b = near(dist), near(flat)
    return {"rank": round(float(spearmanr(dist[iu], flat[iu])[0]), 2), "nn3": round(float(np.mean([len(set(a[i]) & set(b[i])) / 3 for i in range(len(dist))])), 2)}


def align(x, prev):
    """앞 층 좌표에 맞춰 회전·반사만 한다. 거리는 바뀌지 않는다."""
    if prev is None:
        return x
    a, _, b = np.linalg.svd(x.T @ prev)
    return x @ (a @ b)


def layer_geometry():
    path = RES / "layer_vectors_kanana_ko.npz"

    def build():
        from sklearn.manifold import MDS

        d = np.load(path)
        names = [str(n) for n in d["names"]] + ["random0"]
        r = np.random.default_rng(42).standard_normal(d["centered"].shape[-1])
        r = r / np.linalg.norm(r)
        items, p3, p2 = [], None, None
        for v in d["centered"]:
            v = np.vstack([v, r])
            gram = v @ v.T
            dist = np.sqrt(np.maximum(0, 2 - 2 * gram))
            u, s, _ = np.linalg.svd(v, full_matrices=False)
            pca = p3 = align(u[:, :3] * s[:3], p3)
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=8, normalized_stress="auto").fit_transform(dist)
            mds = p2 = align(mds, p2)
            items.append({
                "gram": np.round(gram, 3).tolist(),
                "pca": np.round(pca, 3).tolist(),
                "mds": np.round(mds, 3).tolist(),
                "ratio": np.round(s[:3] ** 2 / (s**2).sum(), 3).tolist(),
                "fit": {"pca2": fidelity(dist, pca[:, :2]), "pca3": fidelity(dist, pca), "mds2": fidelity(dist, mds)},
            })
        return {"names": names, "layers": d["layers"].tolist(), "items": items}

    return cached("layer_geo", [path], build) if path.exists() else {}


SWEEP_FILES = ["layer_sweep.jsonl", "layer_sweep_skips.jsonl", "fluency_layer_sweep.csv", "fluency_generic_layer_sweep.csv", "kote_layer_sweep.csv"]


def layer_sweep():
    """층별 탐색의 칸마다 [감정, 층, 세기, 답변 수, 붕괴 수, 질문에 답한 답변 수, 그 답변들의 KOTE 감정 평균(emos 순서), 무너지지 않았지만 질문에 답하지 않은 답변 수].
    자연스러움 점수가 있는 줄은 collapsed3, 없는 줄은 collapsed2로 판정한다.
    질문과 상관없는 질문 뒤 점수(fluency_generic)가 없는 줄은 무너지지 않았으면 답한 것으로 센다.
    KOTE 감정은 점수 파일에 있는 열을 모두 쓴다(답답함·외로움·질투는 KOTE에 맞는 이름표가 없다)."""
    return cached("layer_sweep", [RES / f for f in SWEEP_FILES], build_layer_sweep)


def build_layer_sweep():
    skip_path = RES / "layer_sweep_skips.jsonl"
    flu_path, gen_path, kote_path = RES / "fluency_layer_sweep.csv", RES / "fluency_generic_layer_sweep.csv", RES / "kote_layer_sweep.csv"
    rows = read_rows("layer_sweep")
    flu = pd.read_csv(flu_path)[["nll", "nll_max20"]].values.tolist() if flu_path.exists() else []
    gen = pd.read_csv(gen_path)["nll"].tolist() if gen_path.exists() else []
    kdf = pd.read_csv(kote_path) if kote_path.exists() else pd.DataFrame()
    emos = [c for c in kdf.columns[5:] if c != "neutral"]
    kote = kdf[emos].values.tolist() if emos else []
    q = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    prompts = {x["id"]: x["ko"] for x in q["neutral"]}
    # 칸마다 [답변 수, 붕괴 수, 질문에 답한 수, KOTE 합, 질문 무시 수, KOTE 점수가 있는 답한 답변 수]
    cells, base = defaultdict(lambda: [0, 0, 0, [0.0] * len(emos), 0, 0]), None
    for i, x in enumerate(rows):
        broke = x["collapsed2"] or (i < len(flu) and flu[i][1] >= BROKEN_MAX20)
        c = cells[(x["condition"], x["layer"], x["s"])]
        c[0] += 1
        c[1] += broke
        if broke:
            continue
        if i < len(gen) and ignored(x["text"], prompts[x["qid"]], gen[i] - flu[i][0]):
            c[4] += 1
            continue
        c[2] += 1
        if i < len(kote):
            c[5] += 1
            c[3] = [a + b for a, b in zip(c[3], kote[i])]
    out = []
    for k, (n, broke, ok, total, skipped, scored) in cells.items():
        mean = [round(t / scored, 3) for t in total] if scored else [None] * len(emos)
        if k[0] == "base":
            base = mean
        out.append([*k, n, broke, ok, mean, skipped])
    skips = [[x["condition"], x["layer"], x["s"]] for x in map(json.loads, skip_path.read_text(encoding="utf-8").splitlines())] if skip_path.exists() else []
    return {"cells": out, "skips": skips, "base": base, "emos": emos, "judge": "collapsed3" if flu else "collapsed2"}


def sweep_search(q="", cond="all", layer="all", sign="all", smin=0.0, smax=2.0, broken="all", page=0, size=50):
    """층별 탐색 답변 찾기. q는 글자 그대로 찾고, /…/로 감싸면 정규식이다. 찾은 곳의 앞뒤 문맥도 돌려준다."""
    rows = cached("sweep_rows", [RES / "layer_sweep.jsonl"], lambda: read_rows("layer_sweep"))
    if q.startswith("/") and q.endswith("/") and len(q) > 2:
        try:
            pat = re.compile(q[1:-1])
        except re.error as e:
            return {"error": f"정규식 오류: {e}"}
    else:
        pat = re.compile(re.escape(q)) if q else None
    hits, total = [], 0
    for i, x in enumerate(rows):
        c, s = x["condition"], x["s"]
        if cond != "all" and c != cond:
            continue
        if layer != "all" and x["layer"] != int(layer):
            continue
        if c != "base" and ((sign == "+" and s < 0) or (sign == "-" and s > 0) or not smin <= abs(s) <= smax):
            continue
        if broken != "all" and bool(x["collapsed2"]) != (broken == "only"):
            continue
        found = list(pat.finditer(x["text"])) if pat else []
        if pat and not found:
            continue
        total += 1
        if page * size <= total - 1 < (page + 1) * size:
            spans = [[m.start(), m.end()] for m in found[:20]]
            hits.append({"row": i, "cond": c, "layer": x["layer"], "s": s, "qid": x["qid"], "collapsed": x["collapsed2"], "text": x["text"], "spans": spans, "count": len(found)})
    return {"total": total, "page": page, "size": size, "hits": hits}


def layers():
    full = RES / "layer_scan_kanana_all_stories_ko.csv"
    scan = pd.read_csv(full).round(3).to_dict("records") if full.exists() else []  # 층마다 평균·최소 AUC, 가짜 벡터 최대, 감정별 AUC
    return {"scan": scan, "geo": layer_geometry(), "sweep": layer_sweep()}


def views():
    path = RES / "view_data.json"
    return cached("views", [path], lambda: json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})


def tokens():
    path = RES / "token_map.json"
    return cached("tokens", [path], lambda: json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": []})


def analysis():
    out = {}
    for label, tag in (("23층", "L23"), ("29층", "L29")):
        kote, acc = RES / f"analysis_kote_{tag}.csv", RES / f"analysis_accuracy_{tag}.csv"
        if kote.exists() and acc.exists():
            out[label] = {"kote": pd.read_csv(kote).round(3).to_dict("records"), "accuracy": pd.read_csv(acc).round(3).to_dict("records")}
    return out


def extras():
    """탐색 분석 표. 파일이 있는 것만 돌려준다."""
    names = {"replicate": "중립 30개와 영어 재현", "decisions": "결정 변화 (기준 대비)", "diverse": "다양한 상황", "neurons": "층 내부 뉴런", "mix": "감정 섞기", "wordswap": "네와 아니오의 뜻 바꾸기", "wordstrip": "영어 단어 삭제 뒤 재채점", "persist": "효과 지속 (앞 20토큰)", "multi": "여러 층 동시 주입", "steer_models": "8비트와 Qwen 판정", "8bit": "8비트와 16비트 비교", "qwen_decisions": "Qwen 결정", "qwen_accuracy": "Qwen 정답률", "behavior": "행동 시나리오", "emoji": "이모티콘 제거"}
    out = {}
    for key, title in names.items():
        path = RES / f"analysis_{key}.csv"
        if path.exists():
            t = pd.read_csv(path).round(3)
            out[title] = {"columns": list(t.columns), "rows": json.loads(t.to_json(orient="values", force_ascii=False))}
    return out


def answers(layer, cond, s):
    q = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    prompts = {x["id"]: x["ko"] for x in q["neutral"] + q["factual"]}
    out = []
    for r in [r for n in [RUNS[layer]] + EXTRA_RUNS.get(layer, []) for r in read_rows(n)]:
        if r["condition"] == cond and abs(r["s"] - s) < 1e-9:
            out.append({"qid": r["qid"], "kind": r["kind"], "grade": r.get("grade"), "collapsed": collapsed(r["text"]), "q": prompts.get(r["qid"], ""), "text": r["text"][:700]})
    out.sort(key=lambda r: (r["kind"] != "neutral", r["qid"]))
    return out


def cosine(mat):
    return np.round(mat @ mat.T, 3).tolist()


def vectors():
    def build():
        out = {"layers": {}}
        for label, name in VECTORS.items():
            d = np.load(RES / f"vectors_{name}.npz")
            sep = pd.read_csv(RES / f"separation_{name}.csv", index_col=0)
            pca = pd.read_csv(RES / f"pca_{name}.csv", index_col=0)
            out["layers"][label] = {
                "names": [str(n) for n in d["names"]],
                "centered": cosine(d["centered"]),
                "vs_neutral": cosine(d["vs_neutral"]),
                "separation": sep.round(3).reset_index(names="emotion").to_dict("records"),
                "pca": pca.round(1).reset_index(names="emotion").to_dict("records"),
            }
        scan = {}
        for lang in ("ko", "en"):
            scan[lang] = pd.read_csv(RES / f"layer_scan_kanana_stories_{lang}.csv").round(3).to_dict("records")
        full = RES / "layer_scan_kanana_all_stories_ko.csv"
        if full.exists():
            scan["ko_all"] = pd.read_csv(full)[["layer", "mean_auc", "min_auc", "null_auc_max"]].round(3).to_dict("records")
        out["scan"] = scan
        ko, en = np.load(RES / "vectors_kanana_stories_ko.npz"), np.load(RES / "vectors_kanana_stories_en.npz")
        names = [str(n) for n in ko["names"]]
        en_idx = {str(n): i for i, n in enumerate(en["names"])}
        out["ko_en"] = sorted(([n, round(float(ko["centered"][i] @ en["centered"][en_idx[n]]), 3)] for i, n in enumerate(names)), key=lambda x: x[1])
        return out

    return cached("vectors", [RES / f"vectors_{n}.npz" for n in VECTORS.values()], build)
