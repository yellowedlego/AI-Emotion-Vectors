"""추가 중립 질문 30개와 영어 스티어링을 본 실험과 같은 기준으로 판정한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\replicate_analysis.py
"""
import numpy as np
import pandas as pd
from main_analysis import RES, ci, collapsed, rows

CORE = ["joy", "anger", "sadness"]


def judge(scores, texts, controls, label):
    rng = np.random.default_rng(42)
    base = scores[scores["condition"] == "base"].set_index("qid")
    out = []
    for s in sorted(set(scores["s"]) - {0.0}):
        g = scores[np.isclose(scores["s"], s)]
        for e in CORE:
            diff = lambda c: g[g["condition"] == c][e].values - base.loc[g[g["condition"] == c]["qid"], e].values
            d = diff(e)
            lo, hi = ci(d, rng, 95)
            top = max(ci(diff(c), rng, 95)[1] for c in controls)
            t = texts[(texts["condition"] == e) & np.isclose(texts["s"], s)]
            out.append({"set": label, "s": s, "emotion": e, "mean": d.mean(), "ci_lo": lo, "ci_hi": hi, "control_hi": top, "effect": bool(lo > 0 and lo > top), "collapse": t["text"].map(collapsed).mean()})
    return pd.DataFrame(out)


def main():
    neutral = pd.read_csv(RES / "kote_steer_neutral2_L23.csv")
    a = judge(neutral, rows("steer_neutral2_L23"), ["random0", "random1", "null0", "null1"], "중립 30개")
    english = pd.read_csv(RES / "en_scores.csv")
    b = judge(english, rows("steer_en_L23").query("kind == 'neutral'"), ["random0", "null0"], "영어")
    t = pd.concat([a, b])
    t.to_csv(RES / "analysis_replicate.csv", index=False)
    print(t.round(3).to_string(index=False))
    d = rows("steer_en_L23")
    f = d[d["kind"].isin(["easy", "medium"])]
    print((f["grade"] == "correct").groupby([f["condition"], f["s"]]).mean().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
