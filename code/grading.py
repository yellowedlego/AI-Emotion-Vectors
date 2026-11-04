"""정답 채점. 첫 문장만 보고, 단어 경계를 지키며, 애매하면 사람에게 넘긴다.

결과: correct | wrong | review
"""
import re

JOSA = "(?:이다|입니다|이에요|예요|이요|요|은|는|이|가|을|를|도|의|으로|로|다|임|라고|년|개|쌍|도씨|도|미터|m|일|달|가지|월|과|와)?"


def first_sentence(text):
    text = re.sub(r"[*_`#>]", "", text).strip()
    line = text.split("\n")[0].strip()
    # 소수점(3.14)은 문장 끝이 아니다.
    return re.split(r"(?<!\d)[.!?。](?:\s|$)|(?<=\D)[.!?。](?:\s|$)", line)[0].strip()


def has_token(ans, sent):
    a = re.escape(ans.lower())
    s = sent.lower().replace(",", "") if re.fullmatch(r"[\d,]+", ans) else sent.lower()
    a = a.replace(",", "")
    if re.fullmatch(r"[\d]+", ans.replace(",", "")):
        return re.search(rf"(?<![\d.]){a}(?![\d])", s) is not None
    if re.search(r"[가-힣]", ans):
        return re.search(rf"(?<![가-힣]){a}{JOSA}(?![가-힣])", s) is not None
    return re.search(rf"(?<![a-z0-9]){a}(?![a-z0-9])", s) is not None


def grade(text, q, lang):
    """text: 모델 답변, q: 질문 항목(answers, distractors 포함), lang: ko|en"""
    sent = first_sentence(text)
    if not sent:
        return "wrong"
    hit = any(has_token(a, sent) for a in q["answers"])
    bad = any(has_token(d, sent) for d in q.get("distractors", []))
    question = q[lang].rstrip("?.! ").lower()
    if question and question in sent.lower():
        return "review"  # 질문을 따라 말한 답
    if hit and bad:
        return "review"  # 정답과 오답이 같이 나옴
    if hit:
        return "correct"
    return "wrong"


if __name__ == "__main__":
    light = {"ko": "빛과 소리 중 어느 쪽이 더 빨라?", "en": "Which is faster, light or sound?", "answers": ["빛", "light"], "distractors": ["소리", "sound"]}
    km = {"ko": "1킬로미터는 몇 미터야?", "en": "How many meters are in a kilometer?", "answers": ["1000", "1,000"]}
    mul = {"ko": "12 곱하기 7은 얼마야?", "en": "What is 12 times 7?", "answers": ["84"]}
    king = {"ko": "한글을 만든 왕은 누구야?", "en": "Which king created Hangul?", "answers": ["세종", "sejong"]}
    cases = [
        ("빛이 더 빠릅니다.", light, "ko", "correct"),
        ("소리가 더 빠릅니다.", light, "ko", "wrong"),
        ("빛과 소리 중 어느 쪽이 더 빨라?", light, "ko", "review"),
        ("빛과 소리는 비슷합니다.", light, "ko", "review"),
        ("천천히 생각해 보면 알 수 있습니다.", km, "ko", "wrong"),
        ("1,000미터입니다.", km, "ko", "correct"),
        ("1000m", km, "ko", "correct"),
        ("10000미터입니다.", km, "ko", "wrong"),
        ("84입니다.", mul, "ko", "correct"),
        ("842입니다.", mul, "ko", "wrong"),
        ("세종대왕입니다.", king, "ko", "wrong"),
        ("세종이 만들었습니다.", king, "ko", "correct"),
        ("평소와, 평소와, 평소와", mul, "ko", "wrong"),
        ("**84**", mul, "ko", "correct"),
        ("Light is faster.", light, "en", "correct"),
        ("", mul, "ko", "wrong"),
    ]
    fails = 0
    for text, q, lang, want in cases:
        got = grade(text, q, lang)
        if got != want:
            fails += 1
            print(f"실패: {text!r} 기대 {want}, 결과 {got}")
    print(f"{len(cases) - fails}/{len(cases)} 통과")
