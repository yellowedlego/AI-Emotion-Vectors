"""Claude가 따로 해 본 분석(참고용). 층별 탐색 결과를 세 가지로 본다.

1. 붕괴: 감정 벡터가 같은 크기의 무작위 벡터(random0~4)보다 글을 더 쉽게 무너뜨리는가.
   판정은 ANALYSIS_PLAN.md(2026-11-04)에 돌리기 전에 적은 규칙을 따른다: 감정의 층 32개 중앙값이 무작위 5개 중
   가장 낮은 값보다 낮으면 '무작위보다 일찍 무너진다', 무작위 5개 범위 안이면 '무작위와 구별되지 않는다'.
2. 감정 효과(교차 검증): 질문 10개를 반으로 나눠, 한쪽 절반으로 감정마다 가장 좋은 층·세기를 고르고
   다른 절반에서 그 칸의 효과를 잰다. 가장 좋은 칸을 고르면 우연히 큰 값이 뽑히기 쉬우므로(선택 편향)
   고른 질문과 잰 질문을 나눈다. 두 방향(앞 절반으로 고르고 뒤로 재기, 그 반대)의 평균을 쓴다.
   같은 절차를 다른 벡터 전부(무작위 5개 포함)에도 똑같이 돌려, 그 감정 점수를 그 감정 벡터가 가장 많이 올리는지 본다.
3. 반대 방향: 고른 층에서 세기를 음수로 바꾸면 그 감정 점수가 내려가는가.

무너진 답변(collapsed3)과 질문을 무시한 답변은 뺀다. 효과 Δ는 같은 질문의 기준 답변(주입 없음) KOTE 점수와의 차이다.
결과는 results/claude_analysis_*.csv에 쓰고 요약을 화면에 찍는다.

--drop-short: 15글자 이하 답변("죄송합니다.", "안녕하세요." 같은 것)도 질문 무시로 보고 뺀다. 무작위 벡터 하나(random3, 5층)가
"죄송합니다."만 내고 KOTE가 이를 슬픔으로 읽는 것을 본 뒤에 더한 확인용 계산이라(미리 정한 계획에 없음) 결과는 _short 파일에 따로 쓴다.

사용법:
    .venv\\Scripts\\python.exe scripts\\claude_analysis.py [--drop-short]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from collapse import BROKEN_MAX20, ignored

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
HALVES = [["n01", "n02", "n03", "n04", "n05"], ["n06", "n07", "n08", "n09", "n10"]]
MIN_OK = 3  # 칸 하나에서 절반(질문 5개) 중 쓸 수 있는 답변이 이만큼은 있어야 본다
RANDOMS = [f"random{i}" for i in range(5)]


def load(drop_short=False):
    rows = [json.loads(x) for x in (RES / "layer_sweep.jsonl").read_text(encoding="utf-8").splitlines()]
    flu = pd.read_csv(RES / "fluency_layer_sweep.csv")
    gen = pd.read_csv(RES / "fluency_generic_layer_sweep.csv")
    kote = pd.read_csv(RES / "kote_layer_sweep.csv")
    prompts = {x["id"]: x["ko"] for x in json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["neutral"]}
    df = kote.copy()
    df["broke"] = df["collapsed2"] | (flu["nll_max20"] >= BROKEN_MAX20)
    df["ignored"] = [not b and ignored(r["text"], prompts[r["qid"]], g - f) for r, b, g, f in zip(rows, df["broke"], gen["nll"], flu["nll"])]
    if drop_short:
        df["ignored"] |= np.array([not b and len(r["text"].strip()) <= 15 for r, b in zip(rows, df["broke"])])
    return df


def collapse_onset(df):
    """조건·층마다 붕괴율이 50%를 처음 넘는 양(+)의 세기. 끝까지 안 넘으면 빈칸. 건너뛴 칸은 붕괴로 본다."""
    skips = {(x["condition"], x["layer"], x["s"]) for x in map(json.loads, (RES / "layer_sweep_skips.jsonl").read_text(encoding="utf-8").splitlines())}
    rate = df[df.condition != "base"].groupby(["condition", "layer", "s"])["broke"].mean()
    out = []
    for sign in (1, -1):
        for (c, l), g in rate.groupby(level=[0, 1]):
            g = g.droplevel([0, 1])
            onset = None
            for s in sorted({abs(x) for x in g.index} | {abs(k[2]) for k in skips if k[0] == c and k[1] == l}):
                if s == 0:
                    continue
                r = g.get(sign * s)
                if (c, l, sign * s) in skips or (r is not None and r >= 0.5):
                    onset = s
                    break
            out.append({"condition": c, "layer": l, "sign": sign, "onset": onset})
    return pd.DataFrame(out)


def cell_deltas(df, emos):
    """칸(조건·층·세기)과 질문 절반마다, 쓸 수 있는 답변의 감정별 평균 Δ와 답변 수."""
    base = df[df.condition == "base"].set_index("qid")[emos]
    ok = df[(df.condition != "base") & ~df.broke & ~df.ignored].copy()
    ok[emos] = ok[emos].values - base.loc[ok.qid].values
    out = {}
    for h, qs in enumerate(HALVES):
        part = ok[ok.qid.isin(qs)]
        g = part.groupby(["condition", "layer", "s"])
        m = g[emos].mean()
        m["n"] = g.size()
        out[h] = m[m.n >= MIN_OK]
    return out


def pick_and_test(d, cond, col):
    """cond 벡터로 col 점수를 올리는 가장 좋은 칸을 한 절반에서 고르고, 다른 절반에서 잰다. 두 방향 평균."""
    res = []
    for sel, test in ((0, 1), (1, 0)):
        a, b = d[sel], d[test]
        if cond not in a.index.get_level_values(0):
            continue
        cand = a.loc[cond]
        cand = cand[cand.index.get_level_values(1) > 0]
        cand = cand[[k in b.index for k in ((cond, l, s) for l, s in cand.index)]]
        if cand.empty:
            continue
        l, s = cand[col].idxmax()
        row = b.loc[(cond, l, s)]
        neg = b.loc[(cond, l, -s)][col] if (cond, l, -s) in b.index else np.nan
        with warnings.catch_warnings():  # 무작위 5개가 모두 빈 칸이면 nan
            warnings.simplefilter("ignore", RuntimeWarning)
            rnd = np.nanmean([b.loc[(r, l, s)][col] if (r, l, s) in b.index else np.nan for r in RANDOMS])
        others = row.drop(["n"])
        res.append({"layer": l, "s": s, "test": row[col], "rank": int((others > row[col]).sum()) + 1, "neg": neg, "random_same_cell": rnd})
    if not res:
        return None
    r = pd.DataFrame(res)
    return {"layers": "/".join(f"{int(l)}층 {s:+.1f}" for l, s in zip(r.layer, r.s)), "test": r.test.mean(), "rank": r["rank"].mean(), "neg": r.neg.mean(), "random_same_cell": r.random_same_cell.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-short", action="store_true")
    args = ap.parse_args()
    tag = "_short" if args.drop_short else ""
    df = load(args.drop_short)
    emos = [c for c in df.columns[5:23] if c != "neutral"]
    conds = sorted(c for c in df.condition.unique() if c != "base")

    # 1. 붕괴 시작 세기: 감정 벡터 vs 무작위 5개
    on = collapse_onset(df)
    on.to_csv(RES / f"claude_analysis_onset{tag}.csv", index=False, encoding="utf-8-sig")
    print("[1] 붕괴가 50%를 넘는 세기 (층 32개 중앙값, 끝까지 안 넘은 층 수)")
    verdicts = []
    for sign in (1, -1):
        t = on[on.sign == sign].groupby("condition")["onset"].agg(median="median", never=lambda x: int(x.isna().sum()))
        rnd, emo = t.loc[RANDOMS], t.drop(RANDOMS)
        lo, hi = rnd["median"].min(), rnd["median"].max()
        for c, r in emo.iterrows():
            v = "무작위보다 일찍 무너진다" if r["median"] < lo else "무작위와 구별되지 않는다" if r["median"] <= hi else "무작위보다 늦게 무너진다"
            verdicts.append({"condition": c, "sign": sign, "median": r["median"], "never": r["never"], "random_min": lo, "random_max": hi, "verdict": v})
        mine = [v for v in verdicts if v["sign"] == sign]
        print(f"  {'+' if sign > 0 else '-'} 방향: 무작위 5개 중앙값 " + ", ".join(f"{c} {r['median']}(안 넘은 층 {r['never']})" for c, r in rnd.iterrows()))
        print(f"    감정 {len(emo)}개 중앙값 범위 {emo['median'].min()}~{emo['median'].max()} | "
              + ", ".join(f"{k} {n}개" for k, n in pd.Series([v["verdict"] for v in mine]).value_counts().items()))
        for kind in ("일찍", "늦게"):
            print(f"    {kind} 무너지는 감정: " + ", ".join(f"{v['condition']}({v['median']})" for v in mine if kind in v["verdict"]))
    pd.DataFrame(verdicts).to_csv(RES / f"claude_analysis_onset_verdict{tag}.csv", index=False, encoding="utf-8-sig")
    by_layer = on[on.sign == 1].pivot(index="layer", columns="condition", values="onset")
    lay = pd.DataFrame({"random_min": by_layer[RANDOMS].min(axis=1), "random_median": by_layer[RANDOMS].median(axis=1),
                        "random_max": by_layer[RANDOMS].max(axis=1), "emotion_median": by_layer.drop(columns=RANDOMS).median(axis=1)})
    lay.to_csv(RES / f"claude_analysis_onset_by_layer{tag}.csv", encoding="utf-8-sig")
    print("  층별(+): 층, 무작위 최소~최대, 감정 중앙값 →", "; ".join(f"{l}:{r.random_min}~{r.random_max}/{r.emotion_median}" for l, r in lay.iloc[::4].iterrows()))

    # 쓸 수 있는 답변 비율
    use = df[df.condition != "base"]
    print(f"  전체 {len(use)}개 중 무너짐 {use.broke.mean():.1%}, 질문 무시 {use.ignored.mean():.1%}")

    # 2·3. 교차 검증한 감정 효과
    d = cell_deltas(df, emos)
    out = []
    for col in emos:
        cell = {c: pick_and_test(d, c, col) for c in conds}
        cell = {c: v for c, v in cell.items() if v}
        own = cell.get(col)
        if not own:
            continue
        others = sorted(((v["test"], c) for c, v in cell.items() if c != col), reverse=True)
        place = 1 + sum(t > own["test"] for t, _ in others)
        out.append({"emotion": col, "best_cells": own["layers"], "delta_test": round(own["test"], 4),
                    "random_same_cell": round(own["random_same_cell"], 4), "delta_neg": round(own["neg"], 4),
                    "rank_among_columns": own["rank"], "place_among_vectors": place, "n_vectors": len(cell),
                    "best_other_vector": others[0][1], "best_other_delta": round(others[0][0], 4),
                    "random_tuned_max": round(max(cell[r]["test"] for r in RANDOMS if r in cell), 4),
                    "random_tuned_mean": round(np.mean([cell[r]["test"] for r in RANDOMS if r in cell]), 4),
                    "best_random_place": 1 + sum(t > max(cell[r]["test"] for r in RANDOMS if r in cell) for t, _ in others)})
    eff = pd.DataFrame(out)
    eff.to_csv(RES / f"claude_analysis_effect{tag}.csv", index=False, encoding="utf-8-sig")
    print("\n[2] 교차 검증 효과 (고른 절반과 다른 절반에서 잰 Δ, 두 방향 평균)")
    print(eff[["emotion", "best_cells", "delta_test", "random_same_cell", "random_tuned_max", "best_random_place", "delta_neg", "rank_among_columns", "place_among_vectors", "best_other_vector"]].to_string(index=False))
    print(f"\n  자기 감정 점수를 벡터 {eff.n_vectors.max()}개 중 1등으로 올린 감정: {int((eff.place_among_vectors == 1).sum())}/{len(eff)}"
          f", 3등 안: {int((eff.place_among_vectors <= 3).sum())}/{len(eff)}")
    print(f"  같은 칸 무작위 5개 평균보다 큰 감정: {int((eff.delta_test > eff.random_same_cell).sum())}/{len(eff)}"
          f", 각자 최적 칸을 고른 무작위 5개 중 가장 큰 것보다도 큰 감정: {int((eff.delta_test > eff.random_tuned_max).sum())}/{len(eff)}")
    print(f"  반대 방향(음수)에서 그 감정 점수가 내려간 감정: {int((eff.delta_neg < 0).sum())}/{int(eff.delta_neg.notna().sum())}")
    print(f"  그 칸에서 17개 감정 열 중 자기 감정이 가장 많이 오른 경우(순위 평균 1.0): {int((eff.rank_among_columns == 1).sum())}/{len(eff)}")


if __name__ == "__main__":
    main()
