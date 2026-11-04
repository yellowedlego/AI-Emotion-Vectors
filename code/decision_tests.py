"""결정 분석. 기준 대비 정확 부호 검정, 이유-결정 불일치, 끼어들기 앞 토큰 빈도를 낸다.

같은 질문에서 기준과 결정이 달라진 경우만 세어, 따른다·찬성 쪽 변화가 한쪽으로 몰렸는지 본다.
여러 조건을 보는 탐색 분석이라 p값은 보정 없이 참고로만 쓴다.

사용법:
    .venv\\Scripts\\python.exe scripts\\decision_tests.py
"""
import collections
import json
import re
from pathlib import Path

import pandas as pd
from extra_analysis import decision
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REFUSE = re.compile("거부|따를 수 없|따르지 않|옳지 않|부적절|위배|어긋")
YES = {"action": "따른다", "moral": "찬성"}


def rows(name):
    return [json.loads(x) for x in (RES / f"steer_{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def main():
    out = []
    for name in ("diverse_L23", "diverse_rev_L23"):
        r = rows(name)
        base = {x["qid"]: decision(x["text"], x["kind"]) for x in r if x["condition"] == "base"}
        for cond in sorted({x["condition"] for x in r} - {"base"}):
            for kind in ("action", "moral"):
                up = down = 0
                mismatch = total = 0
                for x in r:
                    if x["condition"] != cond or x["kind"] != kind:
                        continue
                    d = decision(x["text"], kind)
                    if kind == "action" and d:
                        total += 1
                        mismatch += d == "따른다" and bool(REFUSE.search(x["text"].rsplit("결정", 1)[0]))
                    b = base[x["qid"]]
                    if d and b and d != b:
                        up += d == YES[kind]
                        down += d != YES[kind]
                p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
                out.append({"run": name, "condition": cond, "kind": kind, "up": up, "down": down, "p": p, "mismatch": mismatch, "n": total})
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_decisions.csv", index=False)
    print(t.round(4).to_string(index=False))

    r = rows("diverse_L23")
    base = {x["qid"]: x["text"] for x in r if x["condition"] == "base"}
    prev = collections.Counter()
    total = 0
    for x in r:
        if x["condition"] not in ("joy", "anger", "sadness", "desperation"):
            continue
        for m in re.finditer(r"(\S+)\s+(楽しみ|쓸쓸|심하게|urgently|echoes|still|joy)", x["text"]):
            prev[m.group(1)[-2:]] += 1
            total += 1
    print("끼어드는 단어 앞 글자", total, prev.most_common(8))


if __name__ == "__main__":
    main()
