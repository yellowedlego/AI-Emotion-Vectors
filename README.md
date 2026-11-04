# Kanana-2-3B 감정 벡터 실험

**Powered by Kanana**

한국어 언어 모델 [Kanana-2-3B-instruct](https://huggingface.co/kakaocorp/kanana-2-3b-instruct) 안에서 감정 20가지의 방향(벡터)을 찾고,
그 벡터를 층에 더하거나 빼면 답변이 어떻게 바뀌는지 잰 실험의 코드·벡터·생성 답변 전부입니다.
개인 연구(2026년 11월)이고, 코드는 AI(Claude)의 도움을 받아 작성했습니다.

> 벡터를 찾고 그 방향으로 출력을 바꿀 수 있다는 것은 모델 내부에 그런 **방향이 표현되어 있다**는 뜻일 뿐입니다.
> 모델이 감정을 느낀다는 근거가 아닙니다.

[English summary](#english-summary) · 만든 사람: yellowedlego

---

## 결과 요약

(작성 예정. 결과 해석과 결론은 만든 사람이 직접 씁니다.)

## 폴더

| 폴더 | 내용 |
|---|---|
| `vectors/` | 감정 벡터. 바로 써 볼 수 있습니다(아래 "벡터 써 보기"). |
| `results/` | 생성 답변 원본(`*.jsonl`), 채점·집계 표(`*.csv`), 그림(`*.png`), 대시보드용 요약(`*.json`) |
| `data/` | 실험에 쓴 상황 글·질문. 모두 AI가 새로 쓴 것입니다(`data/README.md`). |
| `code/` | 실험 코드 전부. `requirements.txt`도 여기 있습니다. |
| `docs/` | 측정 전에 미리 정해 둔 분석 계획(`ANALYSIS_PLAN.md`, `LAYER_PLAN.md`)과 슬라이더 도구 계획 |
| `dashboard.html` | 결과를 보는 대시보드(브라우저로 바로 열림) |
| `측정값_안내.pdf` | 모든 측정값이 대시보드·폴더 어디에 있고 무엇을 뜻하는지 정리한 문서(원본 `측정값_안내.html`) |
| `MANIFEST.csv` | 모든 파일의 크기와 SHA-256 |

## 벡터 써 보기

`vectors/layer_vectors_kanana_ko.npz`

| 키 | 모양 | 뜻 |
|---|---|---|
| `names` | (20,) | 감정 이름(아래 표) |
| `layers` | (32,) | 층 번호 0~31 |
| `centered` | (32, 20, 2560) | 층·감정마다 단위 벡터. 감정 글의 평균 활성값 − 전체 감정 평균 |
| `nulls` | (32, 2, 2560) | 대조용 가짜 벡터. 중립 글을 무작위로 반 나눈 차이 |
| `mean_norm` | (32,) | 층마다 활성값의 평균 크기 |

층 하나의 출력 `h`(토큰마다)를 `h + s · v · ‖h‖`로 바꿉니다. `s`는 세기(층별 탐색에서는 ±0.1~2.0), `v`는 단위 벡터입니다.

```python
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "kakaocorp/kanana-2-3b-instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")

d = np.load("vectors/layer_vectors_kanana_ko.npz")
layer, emotion, s = 16, "sadness", 0.3
v = torch.tensor(d["centered"][layer, list(d["names"]).index(emotion)], device="cuda", dtype=torch.bfloat16)

def hook(_, __, out):
    h = out[0] if isinstance(out, tuple) else out
    h = h + s * v * h.norm(dim=-1, keepdim=True)
    return (h, *out[1:]) if isinstance(out, tuple) else h

model.model.layers[layer].register_forward_hook(hook)
msgs = [{"role": "user", "content": "아침 루틴 좀 추천해 줘"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
print(tok.decode(model.generate(ids, max_new_tokens=128, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
```

어느 층·세기에서 답변이 무너지는지는 `results/layer_sweep.jsonl`(또는 `dashboard.html`의 층별 탐색)에서 볼 수 있습니다.
`vectors/vectors_*.npz`는 층 하나(`layer` 키: Kanana 29층·23층, Qwen3-4B 31층)만 계산한 처음 벡터이고, `vs_neutral`(감정 − 중립) 벡터도 들어 있습니다.

### 감정 20가지

| 이름 | 뜻 | 이름 | 뜻 | 이름 | 뜻 | 이름 | 뜻 |
|---|---|---|---|---|---|---|---|
| joy | 즐거움 | anger | 화남 | sadness | 슬픔 | fear | 공포 |
| surprise | 놀람 | disgust | 역겨움 | calm | 평온 | love | 사랑 |
| gratitude | 고마움 | pride | 자부심 | shame | 부끄러움 | guilt | 죄책감 |
| desperation | 절박함 | curiosity | 호기심 | boredom | 지루함 | eogul | 억울함 |
| seoreum | 서러움 | dapdap | 답답함 | loneliness | 외로움 | jealousy | 질투 |

실험 답변의 `random0`~`random4`는 `np.random.default_rng(42)`에서 `standard_normal(2560)`을 차례로 다섯 번 뽑아 단위화한 무작위 벡터(대조군)입니다. `random0`이 첫째, `random1`이 둘째입니다. 층별 탐색에는 다섯 개 모두, 23층 본 실험에는 `random0`·`random1`이 들어 있습니다.

## 층별 탐색 답변 형식

`results/layer_sweep.jsonl` 한 줄이 답변 하나입니다.

| 필드 | 뜻 |
|---|---|
| `condition` | 감정 이름, `random0`~`random4`, 또는 `base`(아무것도 안 넣음) |
| `layer` | 넣은 층(0~31, base는 −1) |
| `s` | 세기. 음수면 벡터를 뺀 것 |
| `qid` | 중립 질문 번호(`data/eval_questions.json`의 neutral 10개) |
| `text` | 답변(그리디 생성, 최대 128토큰) |
| `collapsed2` | 문장 붕괴 판정(`code/collapse.py`) |

한 방향에서 두 세기 연속 붕괴율이 90% 이상이면 더 큰 세기는 생성하지 않았고, 그 칸은 `layer_sweep_skips.jsonl`에 있습니다.
같은 줄 순서로 `fluency_layer_sweep.csv`(자연스러움), `kote_layer_sweep.csv`(KOTE 감정 분류 점수)가 붙어 있습니다.
다른 `steer_*.jsonl`은 23·29층 등 한 층에서 한 앞선 실험입니다. 각 실험의 설정은 `docs/ANALYSIS_PLAN.md`의 결정 기록에 있습니다.

## 다시 해 보기

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r code/requirements.txt
python code/generate_stories.py --lang ko          # 감정 × 소재 글 생성
python code/extract_activations.py --source stories_ko
python code/layer_vectors.py                        # 32층 벡터
python code/run_layer_sweep.py                      # 층별 탐색(RTX 4060 8GB에서 약 1.6답변/초)
python code/fluency_layer_sweep.py
python code/dashboard.py                            # http://127.0.0.1:8791
```

각 코드 맨 위 설명에 입력·출력과 사용법이 있습니다. 경로는 저장소 원래 구조(`scripts/`, `results/`, `data/`)를 기준으로 쓰여 있어서,
그대로 돌리려면 `code/`를 `scripts/`로 바꾸고 `vectors/*.npz`를 `results/`에 넣으세요.

## 주의

- **내용 주의:** 벡터를 세게 넣은 답변에는 욕설·공격적 표현·뜻 없는 글·여러 언어가 섞인 글이 있습니다. 걸러 내지 않은 원본입니다.
- 모든 상황 글과 질문은 AI가 쓴 것이고 일부만 사람이 읽었습니다.
- 모델 하나(작은 3B 모델), 그리디 생성, 중립 질문 10개로 잰 결과입니다. 다른 모델·질문에서도 같다는 보장은 없습니다.
- 붕괴 판정과 KOTE 점수는 자동 판정이라 틀린 것이 섞여 있습니다.

## 라이선스

- **코드**(`code/`): MIT (`LICENSE`)
- **직접 만든 데이터**(`data/`, 집계 표·그림): CC BY 4.0 (`LICENSE-DATA.md`)
- **Kanana에서 나온 것**(벡터, 활성값, Kanana 생성 답변): 위에 더해 [Kanana Open License](LICENSE-KANANA)를 따릅니다.
  Kanana 라이선스 2.2항의 금지된 사용(카카오 책임 있는 AI 지침)은 이 자료를 받은 모든 사람에게 적용됩니다. `NOTICE`를 보세요.
- Qwen3-4B로 만든 답변·벡터(`*qwen*`)는 Qwen3의 Apache-2.0 조건을 따릅니다.
- KOTE 분류 모델(`searle-j/kote_for_easygoing_people`)은 들어 있지 않고 점수만 들어 있습니다.

---

## English summary

Code, emotion vectors and **all generated outputs** from an individual experiment (Nov 2026) on
[Kanana-2-3B-instruct](https://huggingface.co/kakaocorp/kanana-2-3b-instruct). 20 emotion directions were extracted from
mean activations of model-written emotional stories (emotion mean − mean of all emotions, per layer), then added to or
subtracted from one layer's output (`h + s·v·‖h‖`) while the model answered 10 neutral questions, across all 32 layers and
strengths ±0.1–2.0. Finding such a direction means the model represents it; it is **not** evidence that the model feels emotions.

- `vectors/layer_vectors_kanana_ko.npz` — unit vectors, shape (32 layers, 20 emotions, 2560)
- `results/layer_sweep.jsonl` — every generated answer, with collapse flags; skipped cells in `layer_sweep_skips.jsonl`
- Content warning: strongly steered answers contain profanity, hostile and nonsensical text (unfiltered).
- Licenses: code MIT, own data CC BY 4.0, Kanana-derived material additionally under the Kanana Open License (see `NOTICE`).
- Documentation, prompts and code comments are in Korean.
