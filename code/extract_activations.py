"""글을 모델에 넣어 층별 평균 활성값을 저장한다. 벡터 계산과 층 고르기는 저장된 값으로 따로 한다.

입력: results/stories_{ko,en}.jsonl(모델이 쓴 글), data/emotion_scenarios.json(사람이 설계한 글)
출력: results/activations_{source}.npz (acts: 글 수 × 층 수 × hidden), results/activations_{source}.json (글별 정보)

사용법:
    .venv\\Scripts\\python.exe scripts\\extract_activations.py --source stories_ko
source: stories_ko | stories_en | scenarios_ko | scenarios_en
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from progress import Progress
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
BATCH = 16
MAX_LEN = 256


def load_rows(source, story_tag=""):
    kind, lang = source.split("_")
    if kind == "stories":
        prefix = f"{story_tag}_" if story_tag else ""
        rows = [json.loads(x) for x in (ROOT / "results" / f"stories_{prefix}{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
        return [{"emotion": r["emotion"], "topic_id": r["topic_id"], "text": r["text"]} for r in rows]
    sc = json.loads((ROOT / "data" / "emotion_scenarios.json").read_text(encoding="utf-8"))
    # 파일의 sad를 생성 글의 이름 sadness에 맞춘다.
    return [
        {"emotion": "sadness" if e == "sad" else e, "topic_id": s["id"] - 1, "text": s[lang][e]}
        for s in sc
        for e in ("joy", "anger", "sad", "neutral")
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["stories_ko", "stories_en", "scenarios_ko", "scenarios_en"])
    ap.add_argument("--model", default="kakaocorp/kanana-2-3b-instruct")
    ap.add_argument("--mode", choices=["bf16", "8bit"], default="bf16")
    ap.add_argument("--tag", default="kanana", help="출력 파일 이름에 붙일 표시")
    ap.add_argument("--story-tag", default="", help="글 파일의 표시. 예: qwen → stories_qwen_ko.jsonl")
    ap.add_argument("--first", type=float, default=0.40, help="훑을 깊이의 시작(비율)")
    ap.add_argument("--last", type=float, default=0.60, help="훑을 깊이의 끝(비율)")
    args = ap.parse_args()

    rows = load_rows(args.source, args.story_tag)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    kw = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if args.mode == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    layers = model.model.layers
    picked = list(range(round(len(layers) * args.first), round(len(layers) * args.last) + 1))
    print(f"[시작] {args.source}: 글 {len(rows)}개, 층 {picked}", flush=True)

    store = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            store[i] = out[0] if isinstance(out, tuple) else out

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in picked]
    acts = np.zeros((len(rows), len(picked), model.config.hidden_size), dtype=np.float32)
    bar = Progress(f"extract_{args.tag}_{args.source}", len(rows), f"활성값 추출: {args.tag} {args.source}")

    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            part = rows[start : start + BATCH]
            enc = tok([r["text"] for r in part], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(0)
            model(**enc, use_cache=False)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            for k, i in enumerate(picked):
                h = store[i].float()
                mean = (h * mask).sum(1) / mask.sum(1)  # 패딩을 뺀 토큰 평균
                acts[start : start + len(part), k] = mean.cpu().numpy()
            bar.update(start + len(part))
            if (start // BATCH) % 20 == 0:
                print(f"  {start + len(part)}/{len(rows)}", flush=True)

    for h in handles:
        h.remove()

    out = ROOT / "results" / f"activations_{args.tag}_{args.source}"
    np.savez_compressed(out.with_suffix(".npz"), acts=acts, layers=np.array(picked))
    out.with_suffix(".json").write_text(
        json.dumps([{"emotion": r["emotion"], "topic_id": r["topic_id"]} for r in rows], ensure_ascii=False), encoding="utf-8"
    )
    bar.finish()
    print(f"[끝] {out.with_suffix('.npz').name} {acts.shape}", flush=True)


if __name__ == "__main__":
    main()
