"""모델 적재 시험: 정밀도별로 VRAM이 얼마나 드는지, 중간 층에 hook을 걸 수 있는지 확인한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\test_load.py --model kakaocorp/kanana-2-3b-instruct --mode 8bit
mode: bf16 | 8bit | 4bit
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def vram_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kakaocorp/kanana-2-3b-instruct")
    ap.add_argument("--mode", choices=["bf16", "8bit", "4bit"], default="8bit")
    args = ap.parse_args()

    used0, total = vram_gb()
    print(f"[시작] GPU 사용 {used0:.2f} / {total:.2f} GiB")

    kw = {"device_map": {"": 0}}
    if args.mode == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif args.mode == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["torch_dtype"] = torch.bfloat16
    else:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    used1, _ = vram_gb()
    print(f"[적재 완료] {time.time() - t:.0f}초, GPU 사용 {used1:.2f} GiB (모델이 늘린 양 {used1 - used0:.2f} GiB)")

    cfg = model.config
    layers = model.model.layers
    print(f"[구조] 층 {len(layers)}개, hidden {cfg.hidden_size}, layer_types={set(getattr(cfg, 'layer_types', []) or ['(없음)'])}")

    # 중간 층(깊이 50%)에 hook을 걸어 활성값 모양을 확인한다. 값은 바꾸지 않는다.
    mid = len(layers) // 2
    seen = {}

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seen["shape"] = tuple(h.shape)
        seen["norm"] = h.float().norm(dim=-1).mean().item()

    handle = layers[mid].register_forward_hook(hook)
    msgs = [{"role": "user", "content": "물이 끓는 온도는 몇 도야? 한 문장으로 답해 줘."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=60, do_sample=False)
    handle.remove()
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[hook] 층 {mid} 활성값 shape={seen.get('shape')}, 평균 노름={seen.get('norm', 0):.1f}")
    print(f"[생성] {text!r}")
    print(f"[최대 할당] {peak:.2f} GiB")


if __name__ == "__main__":
    main()
