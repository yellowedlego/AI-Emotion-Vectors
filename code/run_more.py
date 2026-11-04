"""run_all.py 다음 단계. 활성도 지도가 끝나면 행동 시나리오와 나머지 17개 감정을 돌린다. 다시 실행하면 이어서 한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_more.py
"""
import time

from run_all import HARD_CONDS, LAYERS, RES, run, steer

OTHERS = "boredom calm curiosity dapdap desperation disgust eogul fear gratitude guilt jealousy loneliness love pride seoreum shame surprise".split()


def main():
    while not (RES / "token_map.html").exists():
        print("run_all.py가 끝나기를 기다리는 중…", flush=True)
        time.sleep(30)
    run("scripts/run_steering.py", "--vectors", LAYERS["L23"], "--lang", "ko", "--task", "behavior", "--conditions", "base", "desperation", "calm", "random0", "--strengths", "0.3", "--repeats", "30", "--name", "behavior_L23")
    for tag, vectors in LAYERS.items():
        steer(vectors, f"hard03_{tag}", "hard", HARD_CONDS, ["0.3"])
    conds = ["--vectors", LAYERS["L23"], "--lang", "ko", "--strengths", "0.3"]
    run("scripts/run_steering.py", *conds, "--task", "situation", "--conditions", "base", "joy", "anger", "sadness", "desperation", "calm", "random0", "--repeats", "5", "--name", "situation_L23")
    run("scripts/run_steering.py", *conds, "--task", "situation", "--conditions", "base", "desperation", "calm", "random0", "--repeats", "3", "--think", "--name", "situation_think_L23")
    run("scripts/run_steering.py", *conds, "--task", "behavior", "--conditions", "base", "desperation", "calm", "random0", "--repeats", "10", "--think", "--name", "behavior_think_L23")
    run("scripts/run_steering.py", *conds, "--task", "diverse", "--conditions", "base", "joy", "anger", "sadness", "desperation", "calm", "random0", "null0", "--name", "diverse_L23")
    steer(LAYERS["L23"], "others_L23", "questions", OTHERS, ["0.2", "0.4"])
    print("\n[추가 측정 끝]", flush=True)


if __name__ == "__main__":
    main()
