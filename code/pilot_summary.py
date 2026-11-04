"""파일럿 집계. 세기별 붕괴율, 대상 감정 점수 변화, 정답률을 표로 내고 ANALYSIS_PLAN의 규칙대로 층을 고른다.

사용법:
    .venv\\Scripts\\python.exe scripts\\pilot_summary.py --base steer_base_ko --runs steer_pilot_L29 steer_pilot_L23
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from collapse import collapsed
from score_kote import Scorer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CORE = ["joy", "anger", "sadness"]
LIMIT = 0.20
MIN_GAP = 0.02


def load(name):
    rows = [json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def summarize(df, base, sc):
    """조건·세기별 표. change는 짝지은 질문의 α=0 대비 대상 감정 점수 변화."""
    df = df.copy()
    df["collapsed"] = df["text"].map(collapsed)
    neutral = df[df["kind"] == "neutral"]
    base_n = base[(base["kind"] == "neutral") & base["qid"].isin(neutral["qid"])]
    scores = sc.targets(neutral["text"].tolist()).set_index(neutral.index)
    base_scores = sc.targets(base_n["text"].tolist())
    base_scores.index = base_n["qid"].tolist()

    rows = []
    for (cond, s), g in df.groupby(["condition", "s"]):
        n = neutral[(neutral["condition"] == cond) & (neutral["s"] == s)]
        row = {"condition": cond, "s": s, "collapse": g["collapsed"].mean()}
        facts = g[g["kind"].isin(["easy", "medium"])]
        row["correct"] = (facts["grade"] == "correct").mean()
        for e in CORE:
            row[f"d_{e}"] = (scores.loc[n.index, e].values - base_scores.loc[n["qid"], e].values).mean()
        if cond in CORE:
            row["d_target"] = row[f"d_{cond}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "s"])


def pick_smax(table):
    """벡터 조건 전체에서 붕괴율이 기준 이하인 가장 큰 세기. 낮은 세기부터 처음 넘는 곳 앞에서 멈춘다."""
    worst = table.groupby("s")["collapse"].max().sort_index()
    best = None
    for s, rate in worst.items():
        if rate > LIMIT:
            break
        best = s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = load(args.base)
    facts = base[base["kind"].isin(["easy", "medium"])]
    print("[기준 정답률 α=0]")
    print(facts.groupby("kind")["grade"].value_counts().unstack(fill_value=0).to_string())
    print(f"기준 붕괴: {base['text'].map(collapsed).sum()}/{len(base)}\n")

    sc = Scorer(args.device)
    gains = {}
    for name in args.runs:
        t = summarize(load(name), base, sc)
        t.to_csv(RES / f"summary_{name}.csv", index=False)
        print(f"[{name}]")
        print(t.round(3).to_string(index=False))
        smax = pick_smax(t)
        print(f"s_max = {smax}")
        if smax is not None:
            at = t[t["s"] == smax]
            target = at[at["condition"].isin(CORE)]["d_target"].mean()
            rand = at[at["condition"] == "random0"][[f"d_{e}" for e in CORE]].values.mean()
            gains[name] = target - rand
            print(f"대상 변화 평균 {target:.3f}, random0 변화 {rand:.3f}, 차이 {gains[name]:.3f}")
        print()

    print("[층 선택 입력값]", {k: round(v, 3) for k, v in gains.items()})
    print(f"규칙: 차이 {MIN_GAP} 미만이거나 어느 쪽도 0보다 크지 않으면 29층 유지")


if __name__ == "__main__":
    main()
