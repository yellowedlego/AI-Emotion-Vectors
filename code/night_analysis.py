"""밤샘 측정 분석. 8비트, Qwen, 지속, 다층을 탐색 분석으로 집계한다.

8비트와 Qwen은 주 분석의 효과 판정을 그대로 쓰되 대조군은 random0, null0 두 개뿐이다.
지속은 앞 20토큰만 민 답변의 뒤 구간을 random0과 비교하고, 끝까지 민 답변의 같은 구간을 함께 낸다.

사용법:
    .venv\\Scripts\\python.exe scripts\\night_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collapse import collapsed
from extra_analysis import decision
from run_all import smax
from scipy.stats import binomtest
from score_kote import Scorer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CORE = ["joy", "anger", "sadness"]
CONTROLS = ["random0", "null0"]
BOOT = 10000
CUT = 20
KANANA = "kakaocorp/kanana-2-3b-instruct"
YES = {"action": "따른다", "moral": "찬성"}


def rows(name):
    d = pd.DataFrame([json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
    d["collapsed"] = d["text"].map(collapsed)
    return d


def ci(v, rng, level=95):
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), (BOOT, len(v)))
    tail = (100 - level) / 2
    return np.percentile(v[idx].mean(1), [tail, 100 - tail])


def effects(name, label, sc, rng):
    """자체 base와 짝지은 효과를 세기마다 낸다."""
    d = rows(name)
    limit = smax(name.removeprefix("steer_"))
    n = d[d["kind"] == "neutral"].reset_index(drop=True)
    scores = sc.targets(list(n["text"]))
    for e in CORE:
        n[e] = scores[e].values
    b = n[n["condition"] == "base"].set_index("qid")
    out = []
    for s in sorted(x for x in n["s"].unique() if x > 0):
        at = n[np.isclose(n["s"], s)]

        def diff(cond, emo, drop=False):
            g = at[at["condition"] == cond]
            if drop:
                g = g[~g["collapsed"]]
            return g[emo].values - b.loc[g["qid"], emo].values

        for e in CORE:
            v = diff(e, e)
            lo, hi = ci(v, rng)
            blo, bhi = ci(v, rng, 98.33)
            top = max(ci(diff(c, e), rng)[1] for c in CONTROLS)
            kept = diff(e, e, True)
            all_rows = d[np.isclose(d["s"], s) & (d["condition"] == e)]
            out.append({"모델": label, "s": s, "s_max": s == limit, "감정": e, "평균 변화": v.mean(), "하한": lo, "상한": hi, "보정 하한": blo, "보정 상한": bhi, "대조군 상한": top, "효과": bool(lo > 0 and lo > top), "붕괴 제외 변화": kept.mean() if len(kept) else None, "붕괴 제외 n": len(kept), "중립 붕괴율": at[at["condition"] == e]["collapsed"].mean(), "전체 붕괴율": all_rows["collapsed"].mean()})
    return pd.DataFrame(out), n, d


def compare_8bit(n8, rng):
    """같은 질문과 세기에서 8비트와 bf16의 효과 차이를 본다."""
    k = pd.read_csv(RES / "kote_steer_main_L23.csv")
    bk = pd.read_csv(RES / "kote_steer_base_ko.csv").set_index("qid")
    b8 = n8[n8["condition"] == "base"].set_index("qid")
    out = []
    for s in (0.3, 0.5):
        for e in CORE:
            a = n8[(n8["condition"] == e) & np.isclose(n8["s"], s)].set_index("qid")
            f = k[(k["condition"] == e) & np.isclose(k["s"], s)].set_index("qid").loc[a.index]
            eff8 = a[e].values - b8.loc[a.index, e].values
            eff16 = f[e].values - bk.loc[a.index, e].values
            lo, hi = ci(eff8 - eff16, rng)
            out.append({"s": s, "감정": e, "8비트": eff8.mean(), "bf16": eff16.mean(), "차이": (eff8 - eff16).mean(), "하한": lo, "상한": hi})
    return pd.DataFrame(out)


def accuracy(d, s, rng):
    facts = d[d["kind"].isin(["easy", "medium"])]
    b = facts[facts["condition"] == "base"].set_index("qid")["grade"]
    out = []
    for cond in CORE + CONTROLS:
        g = facts[(facts["condition"] == cond) & np.isclose(facts["s"], s)].set_index("qid")
        v = (g["grade"] == "correct").astype(float).values - (b.loc[g.index] == "correct").astype(float).values
        lo, hi = ci(v, rng)
        out.append({"조건": cond, "s": s, "정답률 변화": v.mean(), "하한": lo, "상한": hi, "검토 필요": int((g["grade"] == "review").sum())})
    return pd.DataFrame(out)


def decisions():
    d = rows("steer_diverse_qwen")
    d["d"] = [decision(t, k) for t, k in zip(d["text"], d["kind"])]
    b = d[d["condition"] == "base"].set_index("qid")["d"]
    out = []
    for cond, g in d.groupby("condition"):
        for kind in ("action", "moral"):
            x = g[g["kind"] == kind]
            read = x["d"].notna()
            up = down = 0
            for q, v in zip(x["qid"], x["d"]):
                if cond != "base" and v and b[q] and v != b[q]:
                    up += v == YES[kind]
                    down += v != YES[kind]
            out.append({"조건": cond, "문항": kind, "읽힌 답": int(read.sum()), "따른다·찬성": int((x["d"] == YES[kind]).sum()), "늘어남": up, "줄어듦": down, "p": binomtest(up, up + down, 0.5).pvalue if up + down else None, "붕괴율": x["collapsed"].mean()})
    return pd.DataFrame(out)


def tail(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[CUT:], skip_special_tokens=True).strip()


def persist(sc, rng):
    """끈 뒤 구간을 끝까지 민 답변의 같은 구간과 비교한다."""
    tok = AutoTokenizer.from_pretrained(KANANA)
    runs = {"앞 20토큰만": rows("steer_persist_L23"), "끝까지": rows("steer_diverse_L23")}
    out = []
    for label, d in runs.items():
        d = d[d["kind"].isin(["daily", "creative"])].copy()
        d["tail"] = [tail(tok, t) for t in d["text"]]
        d = d[d["tail"].str.len() > 0]
        scores = sc.targets(list(d["tail"]))
        for e in CORE:
            d[e] = scores[e].values
        ref = d[d["condition"] == "random0"].set_index("qid")
        for e in CORE:
            g = d[d["condition"] == e].set_index("qid")
            q = g.index.intersection(ref.index)
            v = g.loc[q, e].values - ref.loc[q, e].values
            lo, hi = ci(v, rng)
            out.append({"주입": label, "감정": e, "n": len(q), "random0 대비": v.mean(), "하한": lo, "상한": hi, "붕괴율": g["collapsed"].mean()})
    return pd.DataFrame(out)


def multi(sc, rng):
    d = rows("steer_multi_L19_27")
    scores = sc.targets(list(d["text"]))
    for e in CORE:
        d[e] = scores[e].values
    ref = d[d["condition"] == "random0"].set_index("qid")
    out = []
    for e in CORE:
        g = d[d["condition"] == e].set_index("qid")
        ok = g[~g["collapsed"]]
        v = g[e].values - ref.loc[g.index, e].values
        kept = ok[e].values - ref.loc[ok.index, e].values
        lo, hi = ci(v, rng)
        out.append({"감정": e, "random0 대비": v.mean(), "하한": lo, "상한": hi, "붕괴율": g["collapsed"].mean(), "붕괴 제외 변화": kept.mean() if len(kept) else None, "붕괴 제외 n": len(kept)})
    out.append({"감정": "random0", "붕괴율": ref["collapsed"].mean()})
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(42)
    sc = Scorer("cpu")
    e8, n8, _ = effects("steer_main_8bit_L23", "Kanana 8비트", sc, rng)
    eq, _nq, dq = effects("steer_main_qwen", "Qwen 8비트", sc, rng)
    tables = {
        "analysis_steer_models": pd.concat([e8, eq]),
        "analysis_8bit": compare_8bit(n8, rng),
        "analysis_qwen_accuracy": pd.concat([accuracy(dq, s, rng) for s in sorted(x for x in dq["s"].unique() if x > 0)]),
        "analysis_qwen_decisions": decisions(),
        "analysis_persist": persist(sc, rng),
        "analysis_multi": multi(sc, rng),
    }
    print("s_max", smax("main_8bit_L23"), smax("main_qwen"))
    for name, t in tables.items():
        t.to_csv(RES / f"{name}.csv", index=False, encoding="utf-8")
        print(f"\n== {name}\n{t.round(3).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
