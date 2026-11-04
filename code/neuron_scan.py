"""층 내부 뉴런(MLP 은닉 8192개)이 감정별로 얼마나 다르게 반응하는지 잰다.

글 840개를 읽히며 down_proj 직전의 값을 글별로 평균한다. 소재를 반으로 나눠 앞 절반에서 고른 뉴런이
뒤 절반에서도 같은 방향으로 반응하는지 본다. 위치를 찾는 것이지 뉴런이 그 감정을 담당한다는 증명은 아니다.

사용법:
    .venv\\Scripts\\python.exe scripts\\neuron_scan.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from extract_activations import BATCH, MAX_LEN, load_rows
from progress import Progress
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "kakaocorp/kanana-2-3b-instruct"
LAYERS = [16, 23, 29]
NEURONS = 8192
THRESHOLD = 0.8
SEED = 42


def extract():
    path = RES / "neurons.npz"
    rows = load_rows("stories_ko")
    if path.exists():
        return np.load(path)["acts"], rows
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    store = {}

    def make_hook(i):
        def hook(_m, inp):
            store[i] = inp[0]

        return hook

    for i in LAYERS:
        model.model.layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i))
    acts = np.zeros((len(rows), len(LAYERS), NEURONS), dtype=np.float32)
    bar = Progress("neuron_scan", len(rows), "뉴런 측정")
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            part = rows[start : start + BATCH]
            enc = tok([r["text"] for r in part], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(0)
            model(**enc, use_cache=False)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for k, i in enumerate(LAYERS):
                h = store[i].float()
                acts[start : start + len(part), k] = ((h * mask).sum(1) / mask.sum(1)).cpu().numpy()
            bar.update(start + len(part))
    bar.finish()
    np.savez_compressed(path, acts=acts)
    return acts, rows


def effect(x, on):
    """감정 글과 나머지 글의 평균 차이를 표준편차로 나눈 값. 뉴런별로 계산한다."""
    a, b = x[on], x[~on]
    sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-6
    return (a.mean(0) - b.mean(0)) / sd


def main():
    acts, rows = extract()
    emo = np.array([r["emotion"] for r in rows])
    topic = np.array([r["topic_id"] for r in rows])
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(np.unique(topic))
    half = np.isin(topic, ids[: len(ids) // 2])
    names = sorted(set(emo) - {"neutral"})
    out, tops = [], {}
    for k, layer in enumerate(LAYERS):
        x = acts[:, k]
        for e in names:
            a = effect(x[half], emo[half] == e)
            b = effect(x[~half], emo[~half] == e)
            same = (np.abs(a) > THRESHOLD) & (np.abs(b) > THRESHOLD) & (np.sign(a) == np.sign(b))
            full = effect(x, emo == e)
            power = full**2
            top1 = np.sort(power)[-NEURONS // 100 :].sum() / power.sum()
            order = np.argsort(-np.abs(full))
            tops[(layer, e)] = set(order[:100].tolist())
            out.append({"layer": layer, "emotion": e, "reliable": int(same.sum()), "split_r": float(np.corrcoef(a, b)[0, 1]), "top1pct_share": float(top1), "top_neuron": int(order[0]), "top_d": float(full[order[0]])})
        # 대조군: 중립 글을 무작위 반으로 나눠도 같은 기준을 넘는 뉴런 수
        idx = np.where(emo == "neutral")[0]
        fake = rng.permutation(idx)[: len(idx) // 2]
        on = np.zeros(len(emo), dtype=bool)
        on[fake] = True
        n = emo == "neutral"
        t = effect(x[n & half], on[n & half])
        s = effect(x[n & ~half], on[n & ~half])
        print(f"층 {layer} 대조군 기준 통과 뉴런 {int(((np.abs(t) > THRESHOLD) & (np.abs(s) > THRESHOLD) & (np.sign(t) == np.sign(s))).sum())}", flush=True)
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_neurons.csv", index=False)
    print(t.groupby("layer")[["reliable", "split_r", "top1pct_share"]].mean().round(3).to_string())
    print(t[t["emotion"].isin(["joy", "anger", "sadness", "fear", "calm", "desperation"])].round(3).to_string(index=False))
    for a, b in (("joy", "love"), ("sadness", "loneliness"), ("joy", "anger")):
        print(a, b, {l: len(tops[(l, a)] & tops[(l, b)]) for l in LAYERS}, "우연 기대 1.2")
    print("층 간 겹침 joy", len(tops[(23, "joy")] & tops[(29, "joy")]), "(뉴런 번호는 층마다 달라 우연 수준이면 정상)")


if __name__ == "__main__":
    main()
