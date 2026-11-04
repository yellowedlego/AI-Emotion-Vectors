"""bf16 안정성 시험: 배치와 생성 길이를 키우며 메모리·속도를 재고, 중간 층에 벡터를 더하는 hook도 같이 시험한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\stress_bf16.py [--mode bf16|8bit]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "kakaocorp/kanana-2-3b-instruct"
PROMPTS = [
    "아침 루틴을 추천해 줘.",
    "고양이에 대한 짧은 글을 써 줘.",
    "물이 끓는 온도는?",
    "12 곱하기 7은 얼마야?",
    "비 오는 날 하면 좋은 일을 알려 줘.",
    "지구에서 가장 큰 바다는 어디야?",
    "친구에게 보낼 생일 축하 문자를 써 줘.",
    "공부가 잘 안 될 때 어떻게 하면 좋을까?",
]


def gib(x):
    return x / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "8bit"], default="bf16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if args.mode == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    free, total = torch.cuda.mem_get_info()
    print(f"[적재] {args.mode}: GPU 사용 {gib(total - free):.2f}/{gib(total):.2f} GiB", flush=True)

    layer = len(model.model.layers) // 2
    vec = torch.randn(model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    vec = vec / vec.norm() * 100  # 평균 활성값 노름(약 440)의 약 1/4 크기
    state = {"alpha": 0.0}

    def steer(_m, _i, out):
        if state["alpha"] == 0.0:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["alpha"] * vec,) + out[1:]
        return out + state["alpha"] * vec

    handle = model.model.layers[layer].register_forward_hook(steer)

    def run(batch, new_tokens, alpha):
        state["alpha"] = alpha
        msgs = [[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}] for i in range(batch)]
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True).to(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t
        made = (out.shape[1] - enc["input_ids"].shape[1]) * batch
        free2, total2 = torch.cuda.mem_get_info()
        print(
            f"batch={batch:<2} 길이={new_tokens:<4} α={alpha:<4} 시간 {dt:5.1f}s "
            f"{made / dt:5.1f} tok/s  최대할당 {gib(torch.cuda.max_memory_allocated()):.2f} GiB  "
            f"GPU 사용 {gib(total2 - free2):.2f} GiB",
            flush=True,
        )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    cases = [(1, 64, 0.0), (1, 256, 0.0), (4, 256, 0.0), (8, 256, 0.0), (8, 512, 0.0), (8, 256, 1.0), (8, 256, 4.0)]
    sample = {}
    for b, n, a in cases:
        try:
            sample[(b, n, a)] = run(b, n, a)
        except torch.cuda.OutOfMemoryError:
            print(f"batch={b} 길이={n} α={a}: ★ 메모리 부족(OOM)", flush=True)
            torch.cuda.empty_cache()
    handle.remove()
    for k in [(8, 256, 0.0), (8, 256, 4.0)]:
        if k in sample:
            print(f"\n[샘플 α={k[2]}] {sample[k][:200]!r}")


if __name__ == "__main__":
    main()
