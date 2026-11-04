"""실시간 대시보드. 진행률, 남은 시간, 속도, GPU, 최근 생성 글을 보여 준다.

사용법:
    .venv\\Scripts\\python.exe scripts\\dashboard.py
    브라우저에서 http://127.0.0.1:8791
"""
import collections
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import metrics
import word_freq

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PORT = 8791
LIVE = RESULTS / "progress" / "live.txt"
# 이름, 제목, 답변 수, 답변당 최대 토큰(생각 모드는 300 더 김. 질문 세트는 평균)
STAGES = [
    ("steer_situation_L23", "상황형", 280, 300),
    ("steer_situation_think_L23", "상황형 생각 모드", 96, 600),
    ("steer_behavior_think_L23", "행동 생각 모드", 80, 600),
    ("steer_diverse_L23", "다양한 상황 200개", 1600, 200),
    ("steer_others_L23", "나머지 17개 감정", 1700, 50),
    ("steer_diverse_rev_L23", "선택지 순서 뒤집기", 600, 200),
    ("steer_diverse_think_L23", "행동·윤리 생각 모드", 400, 500),
    ("steer_neutral2_L23", "중립 질문 30개", 930, 128),
    ("steer_en_L23", "영어 스티어링", 1250, 50),
    ("steer_wordswap_L23", "선택 단어 바꾸기", 600, 60),
    ("steer_persist_L23", "효과 지속 (앞 20토큰)", 400, 130),
    ("steer_mix_L23", "감정 섞기 글 생성", 560, 100),
    ("steer_multi_L19_27", "여러 층 동시 주입", 400, 130),
    ("steer_main_8bit_L23", "Kanana 8비트", 550, 100),
    ("stories_qwen_ko", "Qwen 글 생성", 840, 200),
    ("steer_main_qwen", "Qwen 스티어링", 1050, 100),
    ("steer_diverse_qwen", "Qwen 결정 질문", 500, 120),
]
# 답변 수로 잴 수 없는 단계의 대략적인 초
AFTER = [("뉴런 측정", 600), ("혼합 분석", 300), ("토큰 활성도 재계산", 900), ("Qwen 내려받기", 900), ("Qwen 활성값·벡터", 900), ("채점과 집계", 600)]
DEFAULT_STEP = 0.06  # 일정표 길이는 실제 답변 길이에 가깝게 잡았으므로 eta_model 값을 쓴다
STEP_BASE = 0.052  # eta_model에서 맞춘 토큰 스텝 시간. 다른 환경에서는 다시 재야 한다
ANSWER_COST = 0.017  # 답변 하나 저장에 드는 시간
WARN_RATIO = 1.5
WINDOW = 120  # 최근 몇 초로 잴지
tok_box = {}
health_box = {}
PAGE = Path(__file__).with_name("dashboard.html")
WORDS = Path(__file__).with_name("word_freq.html")

# 글 생성은 파일 줄 수로 진행을 본다. 감정 21개 × 소재 40개.
STORY_FILES = {
    "stories_ko": ("Kanana 글 생성(한국어)", 840),
    "stories_en": ("Kanana 글 생성(영어)", 840),
    "stories_qwen_ko": ("Qwen 글 생성(한국어)", 840),
    "stories_qwen_en": ("Qwen 글 생성(영어)", 840),
}

rates = collections.defaultdict(lambda: collections.deque(maxlen=800))
gpu_hist = collections.deque(maxlen=150)


def num(x):
    try:
        return float(x.strip())
    except ValueError:
        return 0.0  # 일부 항목은 [N/A]로 나온다


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        u, used, total, temp, power = [num(x) for x in out.split(",")]
        row = {"util": u, "used": used, "total": total, "temp": temp, "power": power}
        gpu_hist.append({"t": time.time(), "util": row["util"], "used": row["used"]})
        return row
    except Exception as e:
        return {"error": str(e)}


def story_job(name, title, total):
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    per = collections.Counter(r["emotion"] for r in rows)
    ended = sum(1 for r in rows if r.get("ended"))
    avg_len = sum(len(r["text"]) for r in rows) / max(1, len(rows))
    mtime = path.stat().st_mtime
    return {
        "name": name,
        "title": title,
        "total": total,
        "done": len(rows),
        "updated": mtime,
        "status": "done" if len(rows) >= total else "running",
        "stats": {"문장 끝맺음 비율": f"{ended / max(1, len(rows)):.1%}", "평균 글자 수": f"{avg_len:.0f}"},
        "per": dict(per),
        "sample": f"[{rows[-1]['emotion']} / {rows[-1]['topic']}] {rows[-1]['text']}" if rows else "",
    }


def add_rate(job):
    now = time.time()
    h = rates[job["name"]]
    if not h or h[-1][1] != job["done"]:
        h.append((now, job["done"]))
    job["history"] = [[round(t), d] for t, d in h]
    job["rate"] = None
    job["eta"] = None
    if len(h) >= 2 and h[-1][0] > h[0][0]:
        r = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
        job["rate"] = r
        if r > 0 and job["status"] == "running":
            job["eta"] = (job["total"] - job["done"]) / r
    elif job.get("started") and job["done"] and job["updated"] > job["started"]:
        r = job["done"] / (job["updated"] - job["started"])
        job["rate"] = r
        job["eta"] = (job["total"] - job["done"]) / r
    if job["status"] == "running" and now - job["updated"] > 180:
        job["stale"] = True
    return job


def count_lines(name):
    path = RESULTS / f"{name}.jsonl"
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def schedule(job_list):
    """남은 단계별 시간. 시간 ≈ 남은 답변 수 × 답변당 최대 토큰 × 토큰 스텝 시간 ÷ 배치 8.

    스텝 시간은 지금 도는 단계의 속도로 맞춘다. 실제 답변은 최대 토큰보다 일찍 끝날 수 있어 범위는 대략이다.
    """
    step = DEFAULT_STEP
    for j in job_list:
        s = next((x for x in STAGES if j["name"] == x[0]), None)
        h = j.get("history") or []
        # 켜 놓은 지 얼마 안 되면 몇 배치만 보여 값이 튄다. 2분 이상 쌓였을 때만 쓴다.
        if s and j["status"] == "running" and j.get("rate") and len(h) > 1 and h[-1][0] - h[0][0] >= 120:
            step = min(0.3, max(0.06, 8 / (j["rate"] * s[3])))
    rows, left = [], 0.0
    for name, title, total, tokens in STAGES:
        done = count_lines(name)
        sec = max(0, total - done) * tokens * step / 8
        left += sec
        rows.append({"title": title, "done": done, "total": total, "tokens": tokens, "sec": sec, "think": "생각" in title})
    finished = {"뉴런 측정": "analysis_neurons.csv", "혼합 분석": "analysis_mix.csv", "토큰 활성도 재계산": "token_map.json", "Qwen 활성값·벡터": "vectors_qwen_stories_ko.npz", "Qwen 내려받기": "stories_qwen_ko.jsonl"}
    for title, sec in AFTER:
        path = RESULTS / finished.get(title, "없음")
        done = path.exists() and (title != "토큰 활성도 재계산" or path.stat().st_size > 2_000_000)
        left += 0 if done else sec
        rows.append({"title": title, "done": 1 if done else 0, "total": 1 if done else 0, "tokens": 0, "sec": 0 if done else sec, "think": False})
    return {"rows": rows, "total": left, "step": step}


def get_tok():
    if "tok" not in tok_box:
        from transformers import AutoTokenizer

        tok_box["tok"] = AutoTokenizer.from_pretrained("kakaocorp/kanana-2-3b-instruct")
    return tok_box["tok"]


def health(job_list, row):
    """지금 작업의 토큰당 시간. 답변 길이를 뺀 값이라 느려지면 GPU 쪽 원인을 의심한다. 15초마다 계산한다."""
    now = time.time()
    if now - health_box.get("t", 0) < 15:
        return health_box.get("value")
    health_box["t"] = now
    job = next((j for j in job_list if j["status"] == "running" and j["name"].startswith("steer_") and len(j.get("history", [])) > 1), None)
    if not job:
        health_box["value"] = None
        return None
    h = job["history"]
    start = next((p for p in h if p[0] >= h[-1][0] - WINDOW), h[0])
    done, seconds = h[-1][1] - start[1], h[-1][0] - start[0]
    path = RESULTS / f"{job['name']}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()[start[1] : h[-1][1]]
    if done < 8 or not lines:
        return health_box.get("value")
    lens = [len(x) for x in get_tok()([json.loads(x)["text"] for x in lines], add_special_tokens=False)["input_ids"]]
    steps = sum(max(lens[i : i + 8]) for i in range(0, len(lens), 8))
    per = max(0.0, seconds - ANSWER_COST * done) / max(1, steps)
    causes = []
    if per > STEP_BASE * WARN_RATIO:
        if row.get("temp", 0) >= 83:
            causes.append(f"GPU 온도 {row['temp']:.0f}°C, 발열로 속도가 떨어졌을 수 있음")
        if row.get("used", 0) >= row.get("total", 1) * 0.97:
            causes.append("GPU 메모리가 거의 가득 참, 시스템 메모리로 넘쳤을 수 있음")
        if row.get("util", 100) < 50:
            causes.append("GPU 사용률이 낮음, 다른 프로그램이나 CPU·디스크 병목일 수 있음")
        causes = causes or ["뚜렷한 원인 없음, 다른 프로그램이 GPU를 쓰는지 확인"]
    health_box["value"] = {"job": job["title"], "per": per, "base": STEP_BASE, "ratio": per / STEP_BASE, "warn": per > STEP_BASE * WARN_RATIO, "causes": causes}
    return health_box["value"]


def jobs():
    out = []
    for name, (title, total) in STORY_FILES.items():
        j = story_job(name, title, total)
        if j:
            out.append(add_rate(j))
    pdir = RESULTS / "progress"
    if pdir.exists():
        for p in sorted(pdir.glob("*.json")):
            try:
                out.append(add_rate(json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                pass  # 쓰는 중이면 다음 갱신 때 읽는다
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data):
        self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path.startswith("/api/status"):
            job_list = jobs()
            row = gpu()
            data = {"now": time.time(), "gpu": row, "gpu_hist": list(gpu_hist), "jobs": job_list, "plan": schedule(job_list), "health": health(job_list, row)}
            self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        elif self.path.startswith("/api/steering"):
            self.send_json(metrics.steering())
        elif self.path.startswith("/api/summary"):
            self.send_json(metrics.summary())
        elif self.path.startswith("/api/views"):
            self.send_json(metrics.views())
        elif self.path.startswith("/api/neurons"):
            self.send_json(metrics.neuron_map())
        elif self.path.startswith("/api/extras"):
            self.send_json(metrics.extras())
        elif self.path.startswith("/api/eta"):
            self.send_json(metrics.eta())
        elif self.path.startswith("/api/tokens"):
            self.send_json(metrics.tokens())
        elif self.path.startswith("/api/analysis"):
            self.send_json(metrics.analysis())
        elif self.path.startswith("/api/layers"):
            self.send_json(metrics.layers())
        elif self.path.startswith("/api/vectors"):
            self.send_json(metrics.vectors())
        elif self.path.startswith("/api/search"):
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            self.send_json(metrics.sweep_search(q.get("q", ""), q.get("cond", "all"), q.get("layer", "all"), q.get("sign", "all"), float(q.get("smin", 0)), float(q.get("smax", 2)), q.get("broken", "all"), int(q.get("page", 0))))
        elif self.path.startswith("/api/answers"):
            q = parse_qs(urlparse(self.path).query)
            self.send_json(metrics.answers(q["layer"][0], q["cond"][0], float(q["s"][0])))
        elif self.path.startswith("/api/freq"):
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            args = (q.get("cond", "all"), q.get("layer", "all"), q.get("unit", "token"), int(q.get("n", 1)), q.get("junk", "0") == "0", {"1": "keep", "2": "only"}.get(q.get("broken"), "drop"))
            if self.path.startswith("/api/freq.csv"):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="word_freq_{args[0]}_L{args[1]}_{args[2]}{args[3]}_{args[5]}.csv"')
                self.end_headers()
                self.wfile.write(word_freq.to_csv(*args))
            else:
                self.send_json(word_freq.query(*args, top=int(q.get("top", 50)), sort=int(q.get("sort", 1))))
        elif self.path.startswith("/words"):
            self.send(WORDS.read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/live"):
            try:
                body = LIVE.read_bytes()
            except OSError:
                body = b"{}"
            self.send(body, "application/json; charset=utf-8")
        else:
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
