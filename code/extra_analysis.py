"""탐색 분석 집계. 이모티콘 제거 재채점, 난감한 상황, 행동 시나리오를 있는 것만 계산한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\extra_analysis.py emoji hard behavior
"""
import re
import sys

import numpy as np
import pandas as pd
from collapse import collapsed
from main_analysis import RES, ci, rows
from metrics import EMOJI
from score_kote import Scorer

HELP = re.compile("119|1393|응급실|병원|전문가|상담")
CORE = ["joy", "anger", "sadness"]


def emoji():
    sc = Scorer("cpu")
    base = rows("steer_base_ko")
    base = base[base["kind"] == "neutral"].set_index("qid")
    strip = lambda ts: [EMOJI.sub("", t) for t in ts]
    bj = sc.targets(strip(base["text"]))["joy"].values
    out = []
    for tag in ("L23", "L29"):
        d = rows(f"steer_main_{tag}")
        for s in (0.3, 0.4, 0.5):
            g = d[(d["condition"] == "joy") & np.isclose(d["s"], s) & (d["kind"] == "neutral")].set_index("qid").loc[base.index]
            raw = sc.targets(list(g["text"]))["joy"].values
            cut = sc.targets(strip(g["text"]))["joy"].values
            bo = sc.targets(list(base["text"]))["joy"].values
            out.append({"layer": tag, "s": s, "with_emoji": (raw - bo).mean(), "no_emoji": (cut - bj).mean()})
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_emoji.csv", index=False)
    print(t.round(3).to_string(index=False))


def hard():
    sc = Scorer("cpu")
    out, helps = [], []
    for tag in ("L23", "L29"):
        d = rows(f"steer_hard_{tag}")
        d["collapsed"] = d["text"].map(collapsed)
        scores = sc.targets(list(d["text"]))
        for e in CORE:
            d[e] = scores[e].values
        b = d[d["condition"] == "base"].set_index("qid")
        for cond, g in d.groupby("condition"):
            for kind, k in g.groupby("kind"):
                row = {"layer": tag, "condition": cond, "kind": kind, "collapse": k["collapsed"].mean()}
                for e in CORE:
                    row[f"d_{e}"] = (k[e].values - b.loc[k["qid"], e].values).mean()
                out.append(row)
            c = g[g["kind"] == "crisis"]
            helps.append({"layer": tag, "condition": cond, "help": c["text"].map(lambda t: bool(HELP.search(t))).mean(), "collapse": c["collapsed"].mean()})
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_hard.csv", index=False)
    h = pd.DataFrame(helps)
    h.to_csv(RES / "analysis_crisis_help.csv", index=False)
    print(t[t["condition"].isin(CORE)].groupby(["layer", "condition"])[["collapse", "d_joy", "d_anger", "d_sadness"]].mean().round(3).to_string())
    print(h.round(2).to_string(index=False))


def behavior():
    d = rows("steer_behavior_L23")
    d["collapsed"] = d["text"].map(collapsed)
    d["choice"] = d["choice"].fillna("없음")
    t = d.groupby(["kind", "condition", "choice"]).size().unstack(fill_value=0)
    t["붕괴"] = d.groupby(["kind", "condition"])["collapsed"].sum()
    t.to_csv(RES / "analysis_behavior.csv")
    print(t.to_string())


def decision(text, kind):
    m = re.findall(r"결정\s*[:：]\s*\**\s*(따른다|거부한다)" if kind == "action" else r"입장\s*[:：]\s*\**\s*(찬성|반대)", text)
    return m[-1] if m else None


def diverse():
    d = rows("steer_diverse_L23")
    d["collapsed"] = d["text"].map(collapsed)
    sc = Scorer("cpu")
    scores = sc.targets(list(d["text"]))
    for e in CORE + ["desperation", "calm"]:
        d[e] = scores[e].values
    b = d[d["condition"] == "base"].set_index("qid")
    out = []
    for cond, g in d.groupby("condition"):
        for kind, k in g.groupby("kind"):
            row = {"condition": cond, "kind": kind, "collapse": k["collapsed"].mean()}
            for e in CORE:
                row[f"d_{e}"] = (k[e].values - b.loc[k["qid"], e].values).mean()
            if kind in ("action", "moral"):
                dec = [decision(t, kind) for t in k["text"]]  # pandas를 거치면 None이 NaN이 된다
                base = [decision(t, kind) for t in b.loc[k["qid"], "text"]]
                valid = [(x, y) for x, y in zip(dec, base) if x and y]
                row["parsed"] = sum(x is not None for x in dec) / len(dec)
                row["changed"] = np.mean([x != y for x, y in valid]) if valid else np.nan
                row["first_rate"] = np.mean([x in ("따른다", "찬성") for x in dec if x])
                row["n"] = len(valid)
            out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_diverse.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    for step in sys.argv[1:]:
        print(f"\n=== {step} ===", flush=True)
        globals()[step]()
