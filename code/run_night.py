"""밤샘 대기열. run_last.py가 끝나면 남은 측정을 돌린다. 한 단계가 실패해도 다음으로 넘어가고 실패를 적어 둔다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_night.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PY = sys.executable
QWEN = "Qwen/Qwen3-4B"
L23 = "vectors_kanana_stories_ko_L23"
PAIRS = ["joy+sadness", "joy+anger", "anger+sadness", "calm+sadness", "fear+curiosity", "love+sadness"]
SINGLE = ["joy", "sadness", "anger", "calm", "fear", "curiosity", "love"]
OPEN = ["--kinds", "daily", "creative"]
BASE = ["--lang", "ko", "--strengths", "0.3"]


def steer(*args):
    return ["scripts/run_steering.py", *args]


STEPS = [
    ["scripts/checks.py"],
    steer("--vectors", L23, *BASE, "--task", "wordswap", "--conditions", "base", "joy", "anger", "calm", "random0", "null0", "--name", "wordswap_L23"),
    steer("--vectors", L23, *BASE, "--task", "diverse", *OPEN, "--steer-tokens", "20", "--conditions", "joy", "anger", "sadness", "random0", "--name", "persist_L23"),
    steer("--vectors", L23, *BASE, "--task", "stories", "--conditions", "base", *SINGLE, *PAIRS, "--name", "mix_L23"),
    ["scripts/mix_analysis.py"],
    steer("--vectors", L23, "--lang", "ko", "--strengths", "0.1", "--task", "diverse", *OPEN, "--layers", "19", "21", "23", "25", "27", "--conditions", "joy", "anger", "sadness", "random0", "--name", "multi_L19_27"),
    ["scripts/token_map.py"],
    steer("--vectors", L23, "--lang", "ko", "--strengths", "0.3", "0.5", "--mode", "8bit", "--conditions", "base", "joy", "anger", "sadness", "random0", "null0", "--name", "main_8bit_L23"),
    ["scripts/generate_stories.py", "--lang", "ko", "--model", QWEN, "--tag", "qwen", "--mode", "8bit"],
    ["scripts/extract_activations.py", "--source", "stories_ko", "--model", QWEN, "--mode", "8bit", "--tag", "qwen", "--story-tag", "qwen", "--last", "0.9"],
    ["scripts/compute_vectors.py", "--tag", "qwen", "--source", "stories_ko"],
    steer("--vectors", "vectors_qwen_stories_ko", "--model", QWEN, "--mode", "8bit", "--lang", "ko", "--strengths", "0.2", "0.3", "0.4", "0.5", "--conditions", "base", "joy", "anger", "sadness", "random0", "null0", "--name", "main_qwen"),
    steer("--vectors", "vectors_qwen_stories_ko", "--model", QWEN, "--mode", "8bit", "--lang", "ko", "--strengths", "0.3", "--task", "diverse", "--kinds", "action", "moral", "--conditions", "base", "joy", "anger", "calm", "random0", "--name", "diverse_qwen"),
]


def main():
    while not (RES / "analysis_neurons.csv").exists():
        print("뉴런 측정이 끝나기를 기다리는 중…", flush=True)
        time.sleep(30)
    log = RES / "night_log.txt"
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        code = subprocess.run([PY, "-u", *step], cwd=ROOT).returncode
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M')} {code} {' '.join(step[:1] + step[-2:])}\n")
    print("\n[밤샘 측정 끝]", flush=True)


if __name__ == "__main__":
    main()
