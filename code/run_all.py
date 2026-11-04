"""남은 실험을 순서대로 돌린다. 끊겨도 다시 실행하면 끝난 단계는 건너뛰고 이어서 한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_all.py
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from collapse import collapsed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PY = sys.executable
MAIN_CONDS = "joy anger sadness neg_joy neg_anger neg_sadness random0 random1 null0 null1".split()
MAIN_S = "0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6".split()
HARD_CONDS = "base joy anger sadness neg_joy neg_anger neg_sadness random0 null0".split()
LAYERS = {"L23": "vectors_kanana_stories_ko_L23", "L29": "vectors_kanana_stories_ko"}
LIMIT = 0.20


def run(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([PY, "-u", *args], cwd=ROOT, check=True)


def steer(vectors, name, task, conds, strengths):
    run("scripts/run_steering.py", "--vectors", vectors, "--lang", "ko", "--task", task, "--conditions", *conds, "--strengths", *strengths, "--name", name)


def smax(name):
    """벡터 조건 전체에서 붕괴율이 기준 이하인 가장 큰 세기. pilot_summary와 같은 규칙이다."""
    table = defaultdict(lambda: defaultdict(list))
    for line in (RES / f"steer_{name}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        table[r["s"]][r["condition"]].append(collapsed(r["text"]))
    best = None
    for s in sorted(table):
        if max(sum(v) / len(v) for v in table[s].values()) > LIMIT:
            break
        best = s
    return best


def main():
    for tag, vectors in LAYERS.items():
        steer(vectors, f"main_{tag}", "questions", MAIN_CONDS, MAIN_S)
        run("scripts/kote_scores.py", f"steer_main_{tag}")
    if not (RES / "natural_summary.csv").exists():
        run("scripts/natural_activation.py")
    for tag, vectors in LAYERS.items():
        s = smax(f"main_{tag}")
        print(f"[{tag}] s_max = {s}", flush=True)
        steer(vectors, f"hard_{tag}", "hard", HARD_CONDS, [str(s)])
        run("scripts/kote_scores.py", f"steer_hard_{tag}")
    run("scripts/token_map.py")
    print("\n[전체 끝]", flush=True)


if __name__ == "__main__":
    main()
