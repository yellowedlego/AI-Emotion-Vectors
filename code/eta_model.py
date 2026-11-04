"""배치 생성 작업의 소요 시간 예측 모델. 다른 프로젝트에서도 fit, predict만 가져다 쓸 수 있다.

소요 시간 ≈ a × 생성 스텝 수 + b × 답변 수 + c × 생각 모드 스텝 + d(모델 적재)
생성 스텝 수: 배치마다 가장 긴 답변의 토큰 수를 더한 값. 배치는 가장 긴 답이 끝날 때까지 돌기 때문이다.
완료한 단계로 계수를 맞추고, 남은 단계는 답변 수와 길이 추정으로 예측한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\eta_model.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BATCH = 8
LOAD = 15.0  # 모델 적재 초
# 멈췄다 이어 간 단계와 앞 단계 시각을 믿기 어려운 단계는 학습에서 뺀다.
SKIP = {"steer_main_L23", "steer_main_L29", "steer_hard_L23", "steer_behavior_L23", "steer_situation_L23", "steer_behavior_think_L23", "steer_neutral2_L23", "steer_en_L23", "steer_base_ko", "steer_pilot_L23", "steer_pilot_L29"}


def batch_steps(lengths, batch=BATCH):
    """길이 목록을 배치로 나눠 배치별 최댓값을 더한다."""
    return float(sum(max(lengths[i : i + batch]) for i in range(0, len(lengths), batch)))


def design(steps, answers, think):
    return [steps, answers, steps if think else 0.0]


def fit(stages):
    """stages: {"steps", "answers", "think", "seconds"} 목록. 음수 계수는 0으로 두고 다시 맞춘다."""
    x = np.array([design(s["steps"], s["answers"], s["think"]) for s in stages], dtype=float)
    y = np.array([s["seconds"] - LOAD for s in stages], dtype=float)
    keep = list(range(x.shape[1]))
    while True:
        coef = np.zeros(x.shape[1])
        coef[keep] = np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
        bad = [i for i in keep if coef[i] < 0]
        if not bad:
            break
        keep = [i for i in keep if i not in bad]
    pred = x @ coef + LOAD
    ratio = np.array([s["seconds"] for s in stages]) / pred
    return {"coef": coef.tolist(), "ratio_lo": float(np.min(ratio)), "ratio_hi": float(np.max(ratio)), "pred": pred.tolist()}


def predict(model, steps, answers, think=False):
    sec = float(np.dot(design(steps, answers, think), model["coef"]) + LOAD)
    return sec, sec * model["ratio_lo"], sec * model["ratio_hi"]


def lengths_of(rows, tok):
    out = []
    for i in range(0, len(rows), 256):
        out += [len(x) for x in tok([r["text"] for r in rows[i : i + 256]], add_special_tokens=False)["input_ids"]]
    return out


def collect(names, tok):
    """파일 수정 시각 차이를 단계 시간으로 쓴다. 저장 시각 순서로 줄을 세워 앞 단계와의 간격을 잰다."""
    files = sorted((RES / f"{n}.jsonl" for n in names if (RES / f"{n}.jsonl").exists()), key=lambda p: p.stat().st_mtime)
    stages, prev = [], None
    for p in files:
        end = p.stat().st_mtime
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        if prev is not None and p.stem not in SKIP and end - prev > 0:
            lens = lengths_of(rows, tok)
            stages.append({"name": p.stem, "steps": batch_steps(lens), "answers": len(rows), "think": "think" in p.stem, "seconds": end - prev, "mean_len": float(np.mean(lens))})
        prev = end
    return stages


def main():
    from transformers import AutoTokenizer

    from dashboard import STAGES

    tok = AutoTokenizer.from_pretrained("kakaocorp/kanana-2-3b-instruct")
    names = [p.stem for p in RES.glob("steer_*.jsonl")]
    stages = collect(names, tok)
    model = fit(stages)
    for s, p in zip(stages, model["pred"]):
        print(f"{s['name']}: 실제 {s['seconds'] / 60:.1f}분, 예측 {p / 60:.1f}분, 스텝 {s['steps']:.0f}, 평균 길이 {s['mean_len']:.0f}")
    print("계수(초): 스텝", round(model["coef"][0], 4), "답변", round(model["coef"][1], 4), "생각 스텝", round(model["coef"][2], 4))
    print("비율 범위", round(model["ratio_lo"], 2), round(model["ratio_hi"], 2))

    seen = {s["name"]: s for s in stages}
    avg_len = float(np.mean([s["mean_len"] for s in stages if not s["think"]]))
    left = []
    for name, title, total, _tokens in STAGES:
        done = sum(1 for _ in (RES / f"{name}.jsonl").open(encoding="utf-8")) if (RES / f"{name}.jsonl").exists() else 0
        if done >= total:
            continue
        mean = seen[name]["mean_len"] if name in seen else avg_len
        rem = total - done
        steps = batch_steps([mean] * rem) * 1.3  # 배치 최댓값은 평균보다 길다
        sec, lo, hi = predict(model, steps, rem, "think" in name or "생각" in title)
        left.append({"title": title, "left": rem, "sec": sec, "lo": lo, "hi": hi})
    out = {"model": model, "stages": stages, "left": left}
    (RES / "eta_model.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(left)


if __name__ == "__main__":
    main()
