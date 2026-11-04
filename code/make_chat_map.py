"""슬라이더 창의 벡터 지도 자료를 층마다 계산해 chat_steer.html의 MAP 자리에 넣는다.

벡터 사이 코사인 표(gram)가 있으면 어떤 슬라이더 조합이든 코사인을 2560차원 그대로 계산할 수 있다.
좌표는 평균을 빼지 않은 SVD의 앞 두 축이다. 슬라이더 합이 지도에서도 그대로 벡터 합이 되게 하기 위해서다.
층을 넘길 때 그림이 뒤집히거나 돌지 않도록 앞 층 좌표에 맞춰 2차원 회전·반사를 고른다(거리는 그대로).

사용법:
    .venv\\Scripts\\python.exe scripts\\make_chat_map.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "scripts" / "chat_steer.html"
SEED = 42
STEER_LAYER = 23


def main():
    d = np.load(ROOT / "results" / "layer_vectors_kanana_ko.npz")
    names = [str(n) for n in d["names"]] + ["random0"]
    # run_steering.load_vectors와 같은 무작위 방향
    r = np.random.default_rng(SEED).standard_normal(d["centered"].shape[-1])
    r = r / np.linalg.norm(r)
    out = {"names": names, "layers": d["layers"].tolist(), "steer": STEER_LAYER, "gram": [], "xy": [], "share": []}
    prev = None
    for v in d["centered"]:
        v = np.vstack([v, r])
        u, s, _ = np.linalg.svd(v, full_matrices=False)
        xy = u[:, :2] * s[:2]
        if prev is not None:
            a, _, b = np.linalg.svd(xy.T @ prev)
            xy = xy @ (a @ b)
        prev = xy
        out["gram"].append(np.round(v @ v.T, 3).tolist())
        out["xy"].append(np.round(xy, 3).tolist())
        out["share"].append(round(float((s[:2] ** 2).sum() / (s**2).sum()), 3))
    text = PAGE.read_text(encoding="utf-8")
    blob = json.dumps(out, separators=(",", ":"))
    PAGE.write_text(re.sub(r"const MAP=.*?;/\*END\*/", lambda _: f"const MAP={blob};/*END*/", text, flags=re.S), encoding="utf-8")
    print(f"층 {len(out['layers'])}개, 지도 설명력 {min(out['share']):.0%} ~ {max(out['share']):.0%}, {len(blob) // 1024} KB")


if __name__ == "__main__":
    main()
