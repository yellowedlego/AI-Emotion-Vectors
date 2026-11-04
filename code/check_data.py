"""데이터 검증: 개수, 빈 문장, 감정 어휘 비율, 길이 차이를 확인한다."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
EMO = ["joy", "anger", "sad", "neutral"]
KO_WORDS = ["기쁘", "행복", "화가", "화난", "슬프", "슬퍼", "우울", "짜증", "즐거"]

sc = json.loads((ROOT / "emotion_scenarios.json").read_text(encoding="utf-8"))
qs = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))

problems = []
ids = [s["id"] for s in sc]
if ids != list(range(1, len(ids) + 1)):
    problems.append(f"id가 1..N 연속이 아님: {ids}")
for s in sc:
    for lang in ("ko", "en"):
        if set(s[lang]) != set(EMO):
            problems.append(f"{s['id']} {lang}: 감정 키가 {sorted(s[lang])}")
        for e in EMO:
            if not s[lang].get(e, "").strip():
                problems.append(f"{s['id']} {lang} {e}: 빈 문장")

print(f"상황 {len(sc)}개 × 감정 {len(EMO)} × 언어 2 = {len(sc) * len(EMO) * 2}문장")
for lang, unit in (("ko", "글자"), ("en", "단어")):
    row = []
    for e in EMO:
        lens = [len(s[lang][e]) if lang == "ko" else len(s[lang][e].split()) for s in sc]
        row.append(f"{e} 평균 {statistics.mean(lens):.1f}{unit}")
    print(f"[{lang}] " + " / ".join(row))

hit = {e: sum(any(w in s["ko"][e] for w in KO_WORDS) for s in sc) for e in EMO}
print(f"[ko] 감정 어휘가 들어간 문장 수: {hit}")

print(f"중립 질문 {len(qs['neutral'])}개, 정답 질문 {len(qs['factual'])}개")
for q in qs["factual"]:
    if not q["answers"]:
        problems.append(f"{q['id']}: 정답 없음")
for q in qs["neutral"] + qs["factual"]:
    if not q["ko"].strip() or not q["en"].strip():
        problems.append(f"{q['id']}: 빈 질문")

print("문제 없음" if not problems else "문제:\n- " + "\n- ".join(problems))
