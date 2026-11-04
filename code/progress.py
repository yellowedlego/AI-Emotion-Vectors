"""긴 실행의 진행 상황을 results/progress/<이름>.json에 남긴다. dashboard.py가 이 파일을 읽어 보여 준다."""
import json
import os
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "results" / "progress"
RECENT = 30


class Progress:
    def __init__(self, name, total, title=None):
        DIR.mkdir(parents=True, exist_ok=True)
        self.path = DIR / f"{name}.json"
        self.state = {
            "name": name,
            "title": title or name,
            "total": total,
            "done": 0,
            "started": time.time(),
            "updated": time.time(),
            "status": "running",
            "stats": {},
            "sample": "",
            "recent": [],  # 최근 sample 기록. watch.py가 실행 기록으로 보여 준다.
        }
        self._write()

    def _write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def update(self, done, stats=None, sample=None):
        self.state.update(done=done, updated=time.time())
        if stats:
            self.state["stats"].update(stats)
        if sample is not None:
            self.state["sample"] = sample
            self.state["recent"] = (self.state["recent"] + [{"t": time.time(), "text": sample[:240]}])[-RECENT:]
        self._write()

    def finish(self, status="done"):
        self.state.update(status=status, updated=time.time())
        self._write()
