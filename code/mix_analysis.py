"""혼합 감정 분석. 벡터 A+B로 민 글에서 새 벡터 C를 뽑아 원래 감정들과 비교하고 이름 후보를 찾는다.

C = 그 조건 글의 평균 활성값 − 기준 글의 평균 활성값 (23층)
같은 방법을 단일 감정에도 적용해, 방법이 원래 벡터를 되찾는지 대조한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\mix_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from score_kote import Scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "kakaocorp/kanana-2-3b-instruct"
LAYER = 23


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def activations(texts):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16).eval()
    box = {}
    model.model.layers[LAYER].register_forward_hook(lambda _m, _i, out: box.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    acts = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i : i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256).to(0)
            model(**enc, use_cache=False)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            acts.append((((box["h"].float() * mask).sum(1)) / mask.sum(1)).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(acts)


def main():
    rows = [json.loads(x) for x in (RES / "steer_mix_L23.jsonl").read_text(encoding="utf-8").splitlines()]
    x = activations([r["text"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    d = np.load(RES / "vectors_kanana_stories_ko_L23.npz")
    names = [str(n) for n in d["names"]]
    V = d["centered"]
    base = x[cond == "base"].mean(0)

    sc = Scorer("cpu")
    all_scores = pd.DataFrame(sc.scores([r["text"] for r in rows]), columns=sc.labels)
    all_scores["condition"] = cond
    label = all_scores.groupby("condition").mean()
    diff = label - label.loc["base"]

    out = []
    for c in sorted(set(cond) - {"base"}):
        C = unit(x[cond == c].mean(0) - base)
        cos = V @ C
        order = np.argsort(-cos)
        row = {"condition": c, "nearest": ", ".join(f"{names[i]} {cos[i]:.2f}" for i in order[:4]), "kote_up": ", ".join(f"{k} {v:+.2f}" for k, v in diff.loc[c].sort_values(ascending=False).head(5).items())}
        parts = c.split("+")
        if len(parts) == 2:
            a, b = (V[names.index(p)] for p in parts)
            coef = np.linalg.lstsq(np.stack([a, b], axis=1), C, rcond=None)[0]
            row.update({"cos_a": float(C @ a), "cos_b": float(C @ b), "coef_a": float(coef[0]), "coef_b": float(coef[1]), "r2_ab": float(1 - ((C - np.stack([a, b], axis=1) @ coef) ** 2).sum() / (C**2).sum())})
        else:
            row["cos_self"] = float(C @ V[names.index(c)])
        out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(RES / "analysis_mix.csv", index=False)
    print(t.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
