"""run_final.py가 끝나면 뉴런 측정을 돌린다. 다시 실행하면 이어서 한다.

사용법:
    .venv\\Scripts\\python.exe scripts\\run_last.py
"""
import time

from run_all import run
from run_final import lines


def main():
    while lines("steer_en_L23") < 1250:
        print("run_final.py가 끝나기를 기다리는 중…", flush=True)
        time.sleep(30)
    time.sleep(20)  # 앞 단계가 GPU를 비울 시간
    run("scripts/neuron_scan.py")


if __name__ == "__main__":
    main()
