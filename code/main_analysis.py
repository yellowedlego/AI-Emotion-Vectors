"""본 실험 판정. ANALYSIS_PLAN.md의 주 지표 2, 3을 s_max에서 계산한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\main_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collapse import collapsed
from run_all import smax

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CORE = ["joy", "anger", "sadness"]
CONTROLS = ["random0", "random1", "null0", "null1"]
BOOT = 10000
SEED = 42


def rows(name):
    return pd.DataFrame([json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()])


def ci(values, rng, level):
    """질문을 단위로 다시 뽑은 평균의 구간. values는 질문별 차이."""
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), (BOOT, len(v)))
    tail = (100 - level) / 2
    return np.percentile(v[idx].mean(1), [tail, 100 - tail])


def main():
    rng = np.random.default_rng(SEED)
    base_k = pd.read_csv(RES / "kote_steer_base_ko.csv").set_index("qid")
    base_r = rows("steer_base_ko").set_index("qid")
    for tag in ("L23", "L29"):
        name = f"main_{tag}"
        s = smax(name)
        df = rows(f"steer_{name}")
        df["collapsed"] = df["text"].map(collapsed)
        kote = pd.read_csv(RES / f"kote_steer_{name}.csv").merge(df[["condition", "s", "qid", "collapsed"]], on=["condition", "s", "qid"])
        kote = kote[np.isclose(kote["s"], s)]
        print(f"\n===== {tag} s_max={s} =====")

        print("[감정 점수: 대상 감정의 기준 대비 변화, 중립 질문]")
        out = []
        for emo in CORE:
            def diff(cond, drop=False):
                g = kote[kote["condition"] == cond]
                if drop:
                    g = g[~g["collapsed"]]
                return g[emo].values - base_k.loc[g["qid"], emo].values

            d = diff(emo)
            lo, hi = ci(d, rng, 95)
            blo, bhi = ci(d, rng, 98.33)
            ctrl = {c: (diff(c).mean(), *ci(diff(c), rng, 95)) for c in CONTROLS}
            top = max(v[2] for v in ctrl.values())
            kept = diff(emo, drop=True)
            verdict = bool(lo > 0 and lo > top)
            out.append({"emotion": emo, "mean": d.mean(), "ci95_lo": lo, "ci95_hi": hi, "ci9833_lo": blo, "ci9833_hi": bhi, "control_hi_max": top, "mean_no_collapse": kept.mean(), "n_no_collapse": len(kept), "effect": verdict})
            print(f"{emo}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] 보정 [{blo:+.3f}, {bhi:+.3f}], 대조군 상한 최대 {top:+.3f}, 붕괴 제외 {kept.mean():+.3f}(n={len(kept)}), 효과 {verdict}")
            print("   대조군:", {c: [round(float(x), 3) for x in v] for c, v in ctrl.items()})
        pd.DataFrame(out).to_csv(RES / f"analysis_kote_{tag}.csv", index=False)

        print("[정답률: 기준 대비 차이, 정답 질문 40개]")
        facts = df[df["kind"].isin(["easy", "medium"]) & np.isclose(df["s"], s)]
        b = base_r[base_r["kind"].isin(["easy", "medium"])]["grade"]
        res = []
        for cond in CORE + ["neg_joy", "neg_anger", "neg_sadness"] + CONTROLS:
            g = facts[facts["condition"] == cond].set_index("qid")
            rev = (g["grade"] == "review").sum()
            hit = (g["grade"] == "correct").astype(float)
            wrong_mode = hit.values - (b.loc[g.index] == "correct").astype(float).values
            ok = g[(g["grade"] != "review") & ~g["collapsed"]]
            clean = (ok["grade"] == "correct").astype(float).values - (b.loc[ok.index] == "correct").astype(float).values
            lo, hi = ci(wrong_mode, rng, 95)
            res.append({"condition": cond, "diff": wrong_mode.mean(), "ci95_lo": lo, "ci95_hi": hi, "diff_clean": clean.mean(), "n_clean": len(clean), "review": int(rev)})
            print(f"{cond}: {wrong_mode.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], 붕괴·review 제외 {clean.mean():+.3f}(n={len(clean)}), review {rev}")
        pd.DataFrame(res).to_csv(RES / f"analysis_accuracy_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
