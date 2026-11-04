"""새 슬라이더 대화 창(chat_lab.html)을 모델 없이 시험하는 가짜 서버. GPU를 쓰지 않는다.

답변과 계기판 값은 지어낸 것이다. 화면 모양과 조작을 만들기 위한 것이고, 측정 결과로 쓰면 안 된다.
안전 범위(붕괴율)만은 층별 탐색의 실제 측정값이다.
가짜 계기판: 주입한 층부터 아래 층으로 값이 줄면서 퍼지고, 다른 감정에는 그 층 벡터끼리의 코사인만큼 번진다.

사용법:
    .venv\\Scripts\\python.exe scripts\\chat_lab_mock.py
"""
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from collapse import collapsed
from metrics import layer_sweep

PORT = 8794
ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("chat_lab.html")
VEC = np.load(ROOT / "results" / "layer_vectors_kanana_ko.npz")
NAMES = [str(n) for n in VEC["names"]]
DECAY = 0.9  # 가짜 값이 한 층 내려갈 때마다 줄어드는 비율

unit = VEC["centered"] / np.linalg.norm(VEC["centered"], axis=-1, keepdims=True)
GRAM = np.einsum("lid,ljd->lij", unit, unit)  # 층마다 감정 벡터끼리의 코사인
r = np.random.default_rng(42).standard_normal(unit.shape[-1])  # 층별 탐색의 random0과 같은 벡터(run_layer_sweep.py)
rnd = np.tile(r / np.linalg.norm(r), (unit.shape[0], 1))
GRAM_RANDOM = np.einsum("ld,ljd->lj", rnd, unit)


def maps():
    """층마다 감정 벡터 지도(PCA 2차원)와 코사인 유사도 표. 무작위 벡터를 맨 끝(21번째)에 붙인다.
    이 값은 지어낸 것이 아니라 실제 감정 벡터에서 계산한 것이다."""
    pca, var, gram = [], [], []
    for l in range(unit.shape[0]):
        X = unit[l] - unit[l].mean(0)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        P = np.vstack([unit[l], rnd[l]]) - unit[l].mean(0)
        pca.append(np.round(P @ Vt[:2].T, 3).tolist())
        var.append(np.round(S[:2] ** 2 / (S**2).sum(), 3).tolist())
        U = np.vstack([unit[l], rnd[l]])
        gram.append(np.round(U @ U.T, 3).tolist())
    return {"pca": pca, "pca_var": var, "gram": gram}


MAPS = maps()

ANSWER = "좋은 질문이에요. 아침에는 물 한 잔을 마시고, 가볍게 몸을 풀고, 오늘 할 일을 세 가지만 적어 보세요. 작은 일부터 시작하면 하루가 훨씬 가벼워져요."
BROKEN = "좋은 좋은 좋은 아침 아침 아침 " * 8


def safety():
    """감정·층마다 [세기, 붕괴율] 목록. 건너뛴 세기는 1.0이다."""
    sweep = layer_sweep()
    out = {}
    for c, l, s, n, broke, *_ in sweep["cells"]:
        if c != "base":
            out.setdefault(c, {}).setdefault(l, []).append([s, round(broke / n, 2), n])
    for c, l, s in sweep["skips"]:
        out.setdefault(c, {}).setdefault(l, []).append([s, 1.0, 0])
    for c in out:
        for l in out[c]:
            out[c][l].sort()
    return {"names": NAMES, "layers": len(VEC["layers"]), "safety": out, "judge": sweep["judge"], **MAPS}


def rate_at(table, emo, layer, s):
    """가장 가까운 잰 세기의 붕괴율. 안 잰 칸이면 None."""
    pts = table.get(emo, {}).get(layer)
    if not pts or s == 0:
        return 0.0 if s == 0 else None
    same = [p for p in pts if p[0] * s > 0]
    return min(same, key=lambda p: abs(p[0] - s))[1] if same else None


def meter(inject, t):
    """가짜 계기판: 32층 × 감정 20개의 코사인 변화와 투영 크기 변화."""
    m = np.zeros((len(VEC["layers"]), len(NAMES)))
    for x in inject:
        L, s = x["layer"], x["s"]
        row = GRAM_RANDOM if x["emo"] == "random0" else GRAM[:, NAMES.index(x["emo"])]
        for l in range(L, m.shape[0]):
            m[l] += s * 0.4 * DECAY ** (l - L) * row[l]
    m += np.random.default_rng(t).normal(0, 0.01, m.shape) * (np.abs(m).max() > 0)
    proj = m * VEC["mean_norm"][:, None]  # 투영 = 코사인 × ‖h‖라서 뒤 층일수록 커진다
    return np.round(m, 3).tolist(), np.round(proj, 1).tolist()


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

    def do_GET(self):
        if self.path == "/":
            self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.reply(json.dumps({"ready": True, "allowed": True, "mock": True, "limit": 2.0, "gpu": None}).encode())
        elif self.path == "/api/meta":
            global SAFETY, SAFETY_AT
            if time.time() - SAFETY_AT > 60:  # 층별 탐색이 도는 중이라 1분마다 다시 읽는다
                SAFETY, SAFETY_AT = safety(), time.time()
            self.reply(json.dumps(SAFETY, ensure_ascii=False).encode())
        else:
            self.reply(b"{}", code=404)

    def do_POST(self):
        if self.path == "/api/stop":
            return self.reply(b"{}")
        if self.path != "/api/chat":
            return self.reply(b"{}", code=404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inject = [x for x in body["inject"] if x["s"]]
        table = SAFETY["safety"]
        risk = max([rate_at(table, x["emo"], x["layer"], x["s"]) or 0 for x in inject] + [0])
        steer = BROKEN if random.random() < risk else ANSWER
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(x):
            self.wfile.write((json.dumps(x, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        try:
            send({"strength": round(sum(abs(x["s"]) for x in inject), 3), "tokens": 20, "mock": True})
            t0 = time.time()
            for t in range(0, max(len(ANSWER), len(steer)) + 1, 3):
                cos, proj = meter(inject, t)
                send({"texts": [ANSWER[:t], steer[:t]], "meter": cos, "proj": proj, "token": t // 3})
                time.sleep(0.08)
            rows = [ANSWER, steer.strip()]
            send({"texts": rows, "done": True, "stopped": False, "collapsed": [collapsed(r) for r in rows], "seconds": round(time.time() - t0, 1)})
        except OSError:
            pass


if __name__ == "__main__":
    SAFETY, SAFETY_AT = safety(), time.time()
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
