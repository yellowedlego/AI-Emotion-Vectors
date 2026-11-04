"""영어 단어를 지워도 KOTE 점수 변화가 남는지 본다. 원문, 영어 감정 단어 삭제, 영어 전체 삭제를 비교한다.

기준 답변에도 같은 처리를 하고, 같은 질문끼리 짝지어 부트스트랩한다. 다층 묶음은 기준 답변이 없어 random0과 비교한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\word_strip.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from score_kote import Scorer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EMO = re.compile(r"\b(rage|rages|anger|angry|fury|furious|wrath|joy|joys|joyful|happy|happiness|delight|sad|sadness|sorrow|grief)\b", re.I)
LATIN = re.compile(r"[A-Za-z]+")
CORE = ["joy", "anger", "sadness"]
BOOT = 10000
SETS = {
    "질문 10개 s=0.5": ("steer_main_L23", "steer_base_ko", 0.5, "base"),
    "질문 30개 s=0.5": ("steer_neutral2_L23", None, 0.5, "base"),
    "난감한 질문 s=0.5": ("steer_hard_L23", None, 0.5, "base"),
    "다층 s=0.1": ("steer_multi_L19_27", None, 0.1, "random0"),
}
VARIANTS = {"원문": lambda t: t, "영어 감정 단어 삭제": lambda t: EMO.sub("", t), "영어 전체 삭제": lambda t: LATIN.sub("", t)}


def read(name):
    d = pd.DataFrame([json.loads(x) for x in (RES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
    return d[~d["kind"].isin(["crisis", "easy", "medium"])]


def score(sc, cache, text):
    todo = [t for t in dict.fromkeys(text) if t not in cache]
    if todo:
        for t, row in zip(todo, sc.targets(todo)[CORE].to_dict("records")):
            cache[t] = row
    return pd.DataFrame([cache[t] for t in text])


def main():
    sc = Scorer("cpu")
    rng = np.random.default_rng(0)
    cache, out = {}, []
    for label, (name, base_name, s, ref) in SETS.items():
        d = read(name)
        ref_rows = read(base_name) if base_name else d[d["condition"] == ref]
        ref_rows = ref_rows.drop_duplicates("qid").set_index("qid")
        for e in CORE:
            g = d[(d["condition"] == e) & np.isclose(d["s"], s)].set_index("qid")
            q = g.index.intersection(ref_rows.index)
            hit = g.loc[q, "text"].map(lambda t: bool(EMO.search(t))).mean()
            for v, fn in VARIANTS.items():
                a = score(sc, cache, [fn(t) for t in g.loc[q, "text"]])[e].values
                b = score(sc, cache, [fn(t) for t in ref_rows.loc[q, "text"]])[e].values
                diff = a - b
                idx = rng.integers(0, len(diff), (BOOT, len(diff)))
                m = diff[idx].mean(1)
                out.append({"묶음": label, "감정": e, "처리": v, "n": len(q), "평균 변화": diff.mean(), "하한": np.percentile(m, 2.5), "상한": np.percentile(m, 97.5), "감정 단어 든 답변": hit})
            print(label, e, flush=True)
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_wordstrip.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
