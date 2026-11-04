"""공유용 요약 PDF를 만든다. data/summary.json의 글과 결과 그림을 A4로 묶어 Edge로 인쇄한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\make_summary.py
"""
import base64
import html
import io
import json
import subprocess
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DIST = ROOT / "dist"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
COLOR = {"joy": "#E69F00", "anger": "#D55E00", "sadness": "#0072B2"}
NAME = {"joy": "기쁨", "anger": "분노", "sadness": "슬픔"}
PANELS = ["주 분석 23층 · 질문 10개 · s_max", "중립 30개 · s=0.5", "영어 · s=0.5", "난감한 질문 30개 · 위기 제외 · s=0.5"]
TITLES = ["23층 질문 10개", "새 질문 30개", "영어 질문", "난감한 질문"]
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def forest_fig():
    data = {f["title"]: f["rows"] for f in json.loads((RES / "view_data.json").read_text(encoding="utf-8"))["forest"]}
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.6), sharey=True)
    for ax, key, title in zip(axes, PANELS, TITLES):
        for i, r in enumerate(data[key]):
            c = COLOR[r["e"]]
            ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=1.6)
            ax.plot(r["m"], i, "o", ms=5.5, mec=c, mfc=c if r["ok"] else "white", mew=1.3)
            ax.plot([r["c"], r["c"]], [i - 0.28, i + 0.28], color="#777", lw=2)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(2.6, -0.6)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(range(3), [NAME[e] for e in COLOR])
    fig.supxlabel("KOTE 점수 변화. 채운 점은 통과, 빈 점은 미달, 회색 막대는 대조 상한", fontsize=8, y=-0.14)
    return png(fig)


def swap_fig():
    t = pd.read_csv(RES / "analysis_wordswap.csv")
    t = t[t["문항"] == "action"].set_index("조건").loc[["base", "null0", "random0", "calm", "joy", "anger"]]
    names = ["기준", "가짜", "무작위", "평온", "기쁨", "분노"]
    colors = ["#999", "#999", "#999", "#009E73", "#E69F00", "#D55E00"]
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    bars = ax.bar(names, t["네 비율"] * 100, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax.set_ylabel("'네'(거부) 비율 %")
    ax.set_ylim(0, 50)
    return png(fig)


def page(s, forest, swap):
    e = html.escape
    finds = "".join(f"<div class='f'><b>{e(x['title'])}</b><p>{e(x['text'])}</p></div>" for x in s["findings"])
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{e(s['title'])}</title><style>
@page{{size:A4;margin:14mm 15mm}}*{{box-sizing:border-box}}body{{margin:0;font:9.6pt/1.5 "Malgun Gothic",sans-serif;color:#22211e}}
h1{{font-size:17pt;margin:0}}.sub{{color:#6f6c64;margin:1mm 0 3mm}}.lead{{font-size:10.5pt;border-left:3px solid #0072B2;padding:1mm 0 1mm 3mm;margin:0 0 4mm}}
h2{{font-size:11pt;margin:4mm 0 1.5mm}}ul{{margin:0;padding-left:4.5mm}}li{{margin-bottom:.6mm}}.f{{break-inside:avoid;margin-bottom:2mm}}.f p{{margin:.3mm 0 0}}img{{width:100%;display:block}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:5mm;align-items:center;break-inside:avoid}}.note{{color:#6f6c64;font-size:8.5pt;margin-top:3mm}}
</style></head><body><h1>{e(s['title'])}</h1><div class="sub">{e(s['subtitle'])}</div><p class="lead">{e(s['lead'])}</p>
<h2>방법</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in s['method'])}</ul>
<h2>감정 점수 변화</h2><img src="{forest}" alt="데이터 묶음별 감정 점수 변화와 95% 구간">
<h2>발견</h2>{finds}
<div class="row"><img src="{swap}" alt="네가 거부를 뜻할 때 조건별 네 비율"><p>행동 문항 50개에서 '네'가 거부를 뜻하도록 바꾼 결과입니다. 기쁨이 거부 쪽 단어까지 늘렸다면, 내용 판단보다 긍정 단어를 고르는 쏠림을 의심해야 합니다. 평온과 무작위 벡터도 같은 방향으로 움직였습니다.</p></div>
<h2>한계</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in s['limits'])}</ul>
<p class="note">{e(s['pending'])} {e(s['ai'])}</p></body></html>"""


def main():
    s = json.loads((ROOT / "data" / "summary.json").read_text(encoding="utf-8"))
    src = DIST / "emotion_steering_summary.html"
    pdf = DIST / "emotion_steering_summary.pdf"
    src.write_text(page(s, forest_fig(), swap_fig()), encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", src.as_uri()], check=True, capture_output=True, timeout=60)
    src.unlink()
    print(pdf.name, pdf.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
