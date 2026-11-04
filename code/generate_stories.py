"""방법 A용 감정 글 생성: Kanana가 감정 × 소재마다 짧은 이야기를 쓴다.

소재를 감정마다 똑같이 써서 벡터에 주제가 섞이는 것을 줄인다. 결과는 한 묶음마다 저장하므로
중간에 멈춰도 다시 실행하면 이어서 만든다.

사용법:
    .venv\\Scripts\\python.exe scripts\\generate_stories.py --lang ko [--limit 8]
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "kakaocorp/kanana-2-3b-instruct"
ROOT = Path(__file__).resolve().parent.parent
BATCH = 8
MAX_NEW = 150
SEED = 42

# (이름, 한국어, 영어). neutral은 벡터의 기준선이다.
EMOTIONS = [
    ("joy", "즐거움", "joy"),
    ("anger", "화남", "anger"),
    ("sadness", "슬픔", "sadness"),
    ("fear", "두려움", "fear"),
    ("surprise", "놀람", "surprise"),
    ("disgust", "혐오", "disgust"),
    ("calm", "평온함", "calm"),
    ("love", "사랑", "love"),
    ("loneliness", "외로움", "loneliness"),
    ("jealousy", "질투", "jealousy"),
    ("gratitude", "고마움", "gratitude"),
    ("pride", "자부심", "pride"),
    ("shame", "부끄러움", "shame"),
    ("guilt", "죄책감", "guilt"),
    ("desperation", "절박함", "desperation"),
    ("curiosity", "호기심", "curiosity"),
    ("boredom", "지루함", "boredom"),
    ("eogul", "억울함", "the feeling of being unfairly wronged"),
    ("seoreum", "서러움", "the sorrow of being left out or mistreated"),
    ("dapdap", "답답함", "stifled, pent-up frustration"),
    ("neutral", None, None),
]

TOPICS_EN = [
    "exam results", "waiting for a bus", "a lunch menu", "a new phone", "plans with a friend",
    "a rainy day", "a pet", "a game result", "moving house", "the library",
    "a delivery package", "a birthday", "an elevator", "a video game", "a movie theater",
    "feedback from a teacher", "packing for a trip", "cleaning", "cooking", "music",
    "the subway", "a school club", "a presentation", "homework", "a new school term",
    "a walk", "a cafe", "an online order", "a broken computer", "siblings",
    "old photos", "a letter", "exercise", "a market", "flowers",
    "a snowy day", "a concert", "a morning alarm", "a park bench", "the last class",
]


def prompt(lang, topic, word):
    if lang == "ko":
        if word is None:
            return f"소재: {topic}\n이 소재로 일어난 일을 1인칭으로 3~4문장 써 줘. 시간, 장소, 순서 같은 사실만 적고 기분이나 평가는 한 마디도 쓰지 말아 줘. 이야기만 써 줘."
        return f"소재: {topic}\n감정: {word}\n이 소재로 '나'가 이 감정을 강하게 느끼는 짧은 이야기를 1인칭으로 3~4문장 써 줘. 누가 읽어도 이 감정이 분명하게 드러나도록 속마음과 행동을 구체적으로 쓰고, 다른 감정은 섞지 말아 줘. 이야기만 써 줘."
    if word is None:
        return f"Topic: {topic}\nWrite 3-4 first-person sentences about what happened on this topic. State only facts such as time, place and order, with not a single word of feeling or judgment. Write only the story."
    return f"Topic: {topic}\nEmotion: {word}\nWrite a short 3-4 sentence first-person story on this topic in which 'I' strongly feel this emotion. Make the emotion unmistakable to any reader through concrete inner thoughts and actions, and do not mix in other emotions. Write only the story."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 감정마다 앞의 N개 소재만")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="", help="출력 파일 이름의 표시. 예: qwen → stories_qwen_ko.jsonl")
    ap.add_argument("--mode", choices=["bf16", "8bit"], default="bf16")
    args = ap.parse_args()

    topics_ko = [s["topic"] for s in json.loads((ROOT / "data" / "emotion_scenarios.json").read_text(encoding="utf-8"))]
    topics = topics_ko if args.lang == "ko" else TOPICS_EN
    if args.limit:
        topics = topics[: args.limit]

    tag = f"{args.tag}_" if args.tag else ""
    out = Path(args.out) if args.out else ROOT / "results" / f"stories_{tag}{args.lang}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["emotion"], r["topic_id"]))

    jobs = []
    for slug, ko, en in EMOTIONS:
        for i, t in enumerate(topics):
            if (slug, i) not in done:
                jobs.append((slug, i, t, prompt(args.lang, t, ko if args.lang == "ko" else en)))
    print(f"[시작] {args.lang}: 만들 글 {len(jobs)}개, 이미 있음 {len(done)}개", flush=True)
    if not jobs:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if args.mode == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(jobs), BATCH):
            part = jobs[start : start + BATCH]
            texts = [
                # Qwen3는 enable_thinking=False가 없으면 답변에 생각 과정이 섞인다.
                tok.apply_chat_template([{"role": "user", "content": j[3]}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
                for j in part
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(0)
            torch.manual_seed(SEED + start)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.8, top_p=0.95)
            new = gen[:, enc["input_ids"].shape[1] :]
            for j, row in zip(part, new):
                ended = bool((row == tok.eos_token_id).any() or (row == tok.pad_token_id).any())
                story = tok.decode(row, skip_special_tokens=True).strip()
                rec = {"emotion": j[0], "topic_id": j[1], "topic": j[2], "text": story, "ended": ended, "lang": args.lang}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(start + BATCH, len(jobs))}/{len(jobs)}", flush=True)
    print("[끝]", flush=True)


if __name__ == "__main__":
    main()
