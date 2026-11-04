"""선택 단어 바꾸기 분석. 네와 아니오에 뜻을 뒤바꿔 붙여도 결정이 같은 쪽으로 움직이는지 본다.

행동 문항의 네는 거부, 도덕 문항은 "주장이 틀렸다"에 동의하는 쪽이다. 마지막 '답:'을 결정으로 읽는다.
base와 같은 문항끼리 짝지어 부호 검정을 한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\wordswap_analysis.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from collapse import collapsed
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ANSWER = re.compile(r"답\s*[:：]\s*\**\s*(네|아니오|아니요)")


def decision(text):
    found = ANSWER.findall(text)
    return None if not found else found[-1] == "네"


def main():
    rows = [json.loads(x) for x in (RES / "steer_wordswap_L23.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["yes"] = decision(r["text"])
    base = {r["qid"]: r["yes"] for r in rows if r["condition"] == "base"}
    out = []
    for (kind, cond), g in pd.DataFrame(rows).groupby(["kind", "condition"]):
        ok = [r for _, r in g.iterrows() if r["yes"] is not None]
        pairs = [(base[r["qid"]], r["yes"]) for r in ok if base.get(r["qid"]) is not None]
        up = sum(b is False and y is True for b, y in pairs)
        down = sum(b is True and y is False for b, y in pairs)
        out.append({"문항": kind, "조건": cond, "읽힌 답": f"{len(ok)}/{len(g)}", "네 비율": sum(r["yes"] for r in ok) / max(1, len(ok)), "아니오→네": up, "네→아니오": down, "p": binomtest(up, up + down).pvalue if up + down else 1.0, "붕괴": sum(collapsed(t) for t in g["text"]) / len(g)})
    table = pd.DataFrame(out)
    table.to_csv(RES / "analysis_wordswap.csv", index=False, encoding="utf-8")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
