"""슬라이더 대화 창 서버. 같은 질문을 기준 답변과 민 답변으로 나란히 만들어 글자 단위로 보낸다.

더하는 양 = (슬라이더 값마다 감정 단위 벡터를 곱해 합친 것) × 토큰별 활성값 노름. 23층, 그리디 생성은 주 측정과 같다.
기준과 민 답변을 한 배치로 만들어 두 답변이 같은 시각에 흐른다.

사용법:
    .venv\\Scripts\\python.exe scripts\\chat_steer.py
"""
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from collapse import collapsed
from dashboard import gpu
from run_steering import load_vectors, make_hook
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

PORT = 8792
MODEL = "kakaocorp/kanana-2-3b-instruct"
VECTORS = "vectors_kanana_stories_ko_L23"
MAX_NEW = 300
LIMIT = 0.6
PAGE = Path(__file__).with_name("chat_steer.html")
KEY_FILE = PAGE.parent.parent / ".chat_key"  # 외부에 열 때 이 파일의 글자가 비밀번호가 된다

app = {"ready": False, "error": None, "tok": None, "model": None, "table": None, "layer": None, "state": None}
gpu_lock = threading.Lock()
latest = 0


def load():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
        table, layer = load_vectors(VECTORS, model.config.hidden_size, "centered")
        state = {"vec": None, "s": 1.0, "step": 0, "limit": 0}
        model.model.layers[layer].register_forward_hook(make_hook(state, layer, False))
        app.update(tok=tok, model=model, table=table, layer=layer, state=state, ready=True)
        print("[준비됨]", flush=True)
    except Exception as e:
        app["error"] = str(e)
        raise


class Stream(StoppingCriteria):
    """토큰마다 지금까지의 글을 보낸다. 새 요청이 오거나 연결이 끊기면 생성을 멈춘다."""

    def __init__(self, send, tok, start, job):
        self.send, self.tok, self.start, self.job = send, tok, start, job
        self.last = 0.0
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        if time.time() - self.last > 0.08:
            self.last = time.time()
            try:
                self.send({"texts": self.tok.batch_decode(input_ids[:, self.start :], skip_special_tokens=True)})
            except OSError:
                self.stop = True
        self.stop = self.stop or self.job != latest
        return torch.full((input_ids.shape[0],), self.stop, dtype=torch.bool, device=input_ids.device)


def generate(body, send, job):
    tok, model, table, state = app["tok"], app["model"], app["table"], app["state"]
    mix = np.zeros(model.config.hidden_size)
    for name, s in body["sliders"].items():
        mix = mix + max(-LIMIT, min(LIMIT, float(s))) * table[name]
    v = torch.tensor(mix, device=0, dtype=torch.float32)
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False) for m in (body["base"], body["steer"])]
    enc = tok(texts, return_tensors="pt", padding=True).to(0)
    start = enc["input_ids"].shape[1]
    send({"strength": round(float(np.linalg.norm(mix)), 3), "tokens": start})
    stream = Stream(send, tok, start, job)
    state["vec"] = {app["layer"]: torch.stack([torch.zeros_like(v), v]).unsqueeze(1)}
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, stopping_criteria=StoppingCriteriaList([stream]))
    state["vec"] = None
    rows = [t.strip() for t in tok.batch_decode(gen[:, start:], skip_special_tokens=True)]
    send({"texts": rows, "done": True, "stopped": stream.stop, "collapsed": [collapsed(t) for t in rows], "seconds": round(time.time() - t0, 1)})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, data, kind="application/json; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        if not KEY_FILE.exists():
            return True
        return hmac.compare_digest(self.headers.get("X-Key", "").encode(), KEY_FILE.read_text(encoding="utf-8").strip().encode())

    def do_GET(self):
        if self.path == "/api/status":
            self.reply(json.dumps({"ready": app["ready"], "error": app["error"], "limit": LIMIT, "allowed": self.allowed(), "gpu": gpu() if self.allowed() else None}).encode())
        elif self.path == "/":
            self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self.reply(b"{}", code=404)

    def do_POST(self):
        global latest
        if not self.allowed():
            return self.reply(b"{}", code=401)
        if self.path == "/api/stop":
            latest += 1
            return self.reply(b"{}")
        if self.path != "/api/chat":
            return self.reply(b"{}", code=404)
        if not app["ready"]:
            return self.reply(b'{"error":"not ready"}', code=503)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        latest += 1
        job = latest

        def send(x):
            self.wfile.write((json.dumps(x, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with gpu_lock:
            if job != latest:
                return
            try:
                generate(body, send, job)
            except OSError:
                pass  # 창이 먼저 끊은 경우
            finally:
                app["state"]["vec"] = None


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
