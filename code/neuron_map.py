"""뉴런 지도. 믿을 만한 뉴런을 감정 반응이 비슷한 것끼리 가깝게 평면에 놓는다.

위치는 물리적 위치가 아니다. 뉴런 번호에는 공간적 의미가 없어서, 뉴런마다 20개 감정에 대한 반응값을 만들고
PCA로 두 축에 눌렀다. 묶음은 뉴런이 가장 크게 반응한 감정으로 정한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\neuron_map.py
"""
import json
from pathlib import Path

import numpy as np
from extract_activations import load_rows
from neuron_scan import LAYERS, SEED, THRESHOLD, effect

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GROUPS = {
    "긍정": ["joy", "love", "gratitude", "pride", "calm"],
    "슬픔": ["sadness", "loneliness", "seoreum"],
    "화남·불만": ["anger", "eogul", "jealousy", "dapdap", "disgust"],
    "두려움·수치": ["fear", "desperation", "shame", "guilt"],
    "관심": ["curiosity", "surprise", "boredom"],
}


def main():
    acts = np.load(RES / "neurons.npz")["acts"]
    rows = load_rows("stories_ko")
    emo = np.array([r["emotion"] for r in rows])
    topic = np.array([r["topic_id"] for r in rows])
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(np.unique(topic))
    half = np.isin(topic, ids[: len(ids) // 2])
    names = sorted(set(emo) - {"neutral"})
    group_of = {e: g for g, es in GROUPS.items() for e in es}
    out = {"names": names, "groups": list(GROUPS), "layers": {}}
    for k, layer in enumerate(LAYERS):
        x = acts[:, k]
        a = np.stack([effect(x[half], emo[half] == e) for e in names])
        b = np.stack([effect(x[~half], emo[~half] == e) for e in names])
        full = np.stack([effect(x, emo == e) for e in names]).T
        ok = ((np.abs(a) > THRESHOLD) & (np.abs(b) > THRESHOLD) & (np.sign(a) == np.sign(b))).any(0)
        idx = np.where(ok)[0]
        p = full[idx] - full[idx].mean(0)
        u, s, _ = np.linalg.svd(p, full_matrices=False)
        xy = (u[:, :2] * s[:2]).round(2)
        best = np.abs(full[idx]).argmax(1)
        out["layers"][str(layer)] = {
            "ratio": (s[:2] ** 2 / (s**2).sum()).round(3).tolist(),
            "total": int(ok.sum()),
            "points": [[int(idx[i]), float(xy[i, 0]), float(xy[i, 1]), names[best[i]], group_of[names[best[i]]], round(float(full[idx[i], best[i]]), 2)] for i in range(len(idx))],
        }
        print(layer, int(ok.sum()), out["layers"][str(layer)]["ratio"], flush=True)
    (RES / "neuron_map.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
