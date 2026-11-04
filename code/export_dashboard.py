"""대시보드를 서버 없이 열리는 HTML 파일 하나로 내보낸다. 그래프는 CDN의 Chart.js를 쓴다.

사용법:
    .venv\\Scripts\\python.exe scripts\\export_dashboard.py
"""
import base64
import gzip
import json
from collections import defaultdict
from pathlib import Path

import metrics
from collapse import collapsed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "emotion_steering_dashboard.html"
MAX_CHARS = 400


def all_answers():
    q = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    prompts = {x["id"]: x["ko"] for x in q["neutral"] + q["factual"]}
    out = defaultdict(list)
    for label, name in metrics.RUNS.items():
        for r in [r for n in [name] + metrics.EXTRA_RUNS.get(label, []) for r in metrics.read_rows(n)]:
            out[f"{label}|{r['condition']}|{r['s']:g}"].append(
                {"qid": r["qid"], "kind": r["kind"], "grade": r.get("grade"), "collapsed": collapsed(r["text"]), "q": prompts.get(r["qid"], ""), "text": r["text"][:MAX_CHARS]}
            )
    for rows in out.values():
        rows.sort(key=lambda r: (r["kind"] != "neutral", r["qid"]))
    return out


def main():
    data = {"steering": metrics.steering(), "vectors": metrics.vectors(), "layers": metrics.layers(),"analysis": metrics.analysis(), "extras": metrics.extras(), "neurons": metrics.neuron_map(), "views": metrics.views(), "summary": metrics.summary(), "answers": all_answers()}
    # 모델이 낸 깨진 글자(U+FFFD)는 이스케이프로 쓴다. 그대로 두면 아티팩트 게시가 거부된다.
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("�", "\\ufffd")
    page = (ROOT / "scripts" / "dashboard.html").read_text(encoding="utf-8")
    # 토큰 지도가 파일 크기의 대부분이라 압축해 넣고, 토큰 탭을 열 때 푼다.
    tokens = base64.b64encode(gzip.compress(json.dumps(metrics.tokens(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode()
    page = page.replace("<!--STATIC-->", f"<script>window.STATIC={blob};window.STATIC_TOKENS='{tokens}'</script>")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"[끝] {OUT.name} {OUT.stat().st_size / 1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
