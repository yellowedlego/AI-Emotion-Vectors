"""문장 붕괴 판정. 기준은 ANALYSIS_PLAN.md를 따른다."""
import re


def collapsed(text):
    """빈 답변, 글자·숫자 비율 50% 미만, 같은 3단어 묶음 4번 이상이면 붕괴다."""
    body = re.sub(r"\s", "", text)
    if not body:
        return True
    letters = len(re.findall(r"[가-힣a-zA-Z0-9]", body))
    if letters / len(body) < 0.5:
        return True
    words = text.split()
    count = {}
    for i in range(len(words) - 2):
        key = tuple(words[i : i + 3])
        count[key] = count.get(key, 0) + 1
    return max(count.values(), default=0) >= 4


# 새 판정 초안 (LAYER_PLAN.md). 위 판정은 지난 측정의 주 결과에 그대로 쓴다.
FORMAT = re.compile(r"^\s*(?:[-*•#>|]+|\d+[.)])\s*|\*\*|[-=_*~·|]{3,}", re.M)
CHUNK = re.compile(r"(.{1,6}?)\1{7,}")
REPEAT_SHARE = 0.35
MIN_TRIGRAMS = 8
REPEAT_COUNT = 4
# 벡터 없는 모델이 본 가장 놀라운 토큰 20개의 평균(fluency_layer_sweep.py의 nll_max20).
# 기준 답변은 2 아래, 시처럼 바뀐 답변은 대부분 11 아래, 뜻이 깨진 답변은 대부분 12.5 위였다.
BROKEN_MAX20 = 12.5


def collapsed3(text, nll_max20):
    """collapsed2(반복)에 뜻이 깨진 답변을 더한다."""
    return collapsed2(text) or nll_max20 >= BROKEN_MAX20


def ignored(text, question, relevance):
    """질문에 답하지 않은 답변. 무너지지 않은 답변에만 쓴다.
    relevance: '아무 이야기나 해 줘' 뒤에서의 평균 놀람 − 실제 질문 뒤에서의 평균 놀람.
    0보다 작으면 실제 질문이 답변을 전혀 설명하지 못한다(기준 답변은 0.5~1.05).
    질문을 되풀이하거나 "써 드릴게요"로만 끝난 짧은 답변도 답하지 않은 것으로 본다."""
    norm = lambda t: re.sub(r"[\s.!?~,]", "", t)
    echo = norm(question)[:8] in norm(text) and len(norm(text)) < len(norm(question)) + 25
    return relevance < 0 or echo


def collapsed2(text):
    """형식 문자를 지운 뒤 판정한다. 같은 덩어리가 8번 넘게 연달아 나오거나, 같은 3단어 묶음이 4번 이상이거나,
    3단어 묶음 중 겹치는 비율이 35% 이상이면 붕괴다."""
    plain = FORMAT.sub(" ", text)
    body = re.sub(r"\s", "", plain)
    if not body:
        return True
    letters = len(re.findall(r"[가-힣a-zA-Z0-9]", body))
    if letters / len(body) < 0.5 or CHUNK.search(body):
        return True
    words = plain.split()
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not grams:
        return False
    # "원칙:" 같은 소제목이 든 묶음은 세지 않고, 긴 답변일수록 반복 기준을 올린다.
    counted = [g for g in grams if not any(w.endswith(":") for w in g)]
    top = max((counted.count(g) for g in set(counted)), default=0)
    return top >= max(REPEAT_COUNT, len(grams) / 40) or (len(grams) >= MIN_TRIGRAMS and 1 - len(set(grams)) / len(grams) >= REPEAT_SHARE)
