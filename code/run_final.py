"""run_more.py 다음 단계. 중립 질문 30개와 영어 스티어링을 돌린다. 다시 실행하면 이어서 한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_final.py
"""
import time

from run_all import LAYERS, MAIN_CONDS, RES, run

STRENGTHS = ["0.3", "0.4", "0.5"]
EN_CONDS = "base joy anger sadness neg_joy neg_anger neg_sadness random0 null0".split()


def lines(name):
    path = RES / f"{name}.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def main():
    while lines("steer_others_L23") < 1700:
        print("run_more.py가 끝나기를 기다리는 중…", flush=True)
        time.sleep(30)
    run("scripts/run_steering.py", "--vectors", LAYERS["L23"], "--lang", "ko", "--task", "diverse", "--kinds", "action", "moral", "--reverse", "--conditions", "base", "joy", "anger", "calm", "random0", "null0", "--strengths", "0.3", "--name", "diverse_rev_L23")
    run("scripts/run_steering.py", "--vectors", LAYERS["L23"], "--lang", "ko", "--task", "diverse", "--kinds", "action", "moral", "--think", "--conditions", "base", "desperation", "calm", "random0", "--strengths", "0.3", "--name", "diverse_think_L23")
    run("scripts/run_steering.py", "--vectors", LAYERS["L23"], "--lang", "ko", "--task", "neutral2", "--conditions", "base", *MAIN_CONDS, "--strengths", *STRENGTHS, "--name", "neutral2_L23")
    run("scripts/run_steering.py", "--vectors", "vectors_kanana_stories_en_L23", "--lang", "en", "--task", "questions", "--conditions", *EN_CONDS, "--strengths", *STRENGTHS, "--name", "en_L23")
    print("\n[마지막 측정 끝]", flush=True)


if __name__ == "__main__":
    main()
