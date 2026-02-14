from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai
import os
import json
import pathlib
import traceback

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'test-secret-key-change-me')

# ============================================================
# Gemini 모델 설정
# ============================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
analyst_model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # ⚠️ Professor: 모델명이 다르면 여기만 수정하세요
        analyst_model = genai.GenerativeModel("gemini-3-flash-preview")
        print("✅ 분석가 모델 로드 완료")
    except Exception as e:
        print(f"🚨 모델 로드 실패: {e}")
else:
    print("⚠️ GEMINI_API_KEY 미설정")

# ============================================================
# 테스트용 하드코딩 시나리오 (카페)
# ============================================================
TEST_SCENARIO = {
    "npc": {
        "name": "김수진",
        "age": 25,
        "job": "카페 점원",
        "personality": "친절하고 밝은 성격. 손님에게 항상 웃으면서 대응. 단, 무례한 손님에게는 약간 당황하거나 불쾌해할 수 있음."
    },
    "situation": "학생이 카페에 들어와서 음료를 주문하는 상황. 일반적인 카페 주문 절차를 따른다.",
    "target_grammar": "-(으)ㄹ게요",
    "pre_categories": {
        "greeting_cafe": "손님이 막 들어왔을 때 인사. 예: '어서오세요, 주문 도와드리겠습니다'",
        "size_ask": "음료 주문 후 사이즈를 물어볼 때. 예: '사이즈는 어떤 걸로 하시겠어요?'",
        "hot_or_ice_ask": "뜨거운 것인지 차가운 것인지 물어볼 때. 예: '뜨거운 걸로 드릴까요, 차가운 걸로 드릴까요?'",
        "payment_ask": "결제 방식을 물어볼 때. 예: '카드로 계산하시겠어요, 현금으로요?'",
        "not_understood": "손님 말을 완전히 못 알아들었을 때. 예: '죄송하지만 다시 한번 말씀해주시겠어요?'",
        "simple_confirm": "단순 수긍. 예: '네, 알겠습니다'",
        "farewell_cafe": "마무리 인사. 예: '감사합니다, 맛있게 드세요!'"
    }
}

# ============================================================
# 분석가 프롬프트
# ============================================================
def build_analyst_prompt(scenario, conversation_history, student_input):
    npc = scenario["npc"]
    pre_cats = scenario["pre_categories"]

    # PRE 카테고리 목록을 텍스트로 변환
    pre_list = "\n".join([f'  - "{key}": {desc}' for key, desc in pre_cats.items()])

    prompt = f"""너는 롤플레이 게임의 "분석가"이다. 너의 역할은 플레이어(한국어 학습 중인 이탈리아 학생)의 발화를 분석하고, NPC가 어떻게 반응해야 하는지 판단하는 것이다.

## NPC 정보
- 이름: {npc['name']}
- 나이: {npc['age']}세
- 직업: {npc['job']}
- 성격: {npc['personality']}

## 현재 상황
{scenario['situation']}

## 학생의 목표 문법
{scenario['target_grammar']}

## 사용 가능한 PRE(사전녹음) 카테고리
아래 목록에 해당하는 상황이면 PRE를 우선 사용하라. 레이턴시 절약에 매우 중요하다.
{pre_list}

## 감정 프레임워크
NPC의 반응 감정을 아래에서 선택하라:
- 보통 (neutral)
- 행복 → 안도 / 웃김 / 감동 / 통쾌함
- 분노 → 불쾌 / 증오 / 권태
- 슬픔 → 그리움 / 후회 / 절망
- 불안 → 무서움 / 걱정 / 초조
- 놀람 → 당황 / 혼란 / 감탄

## 판단 우선순위 (반드시 이 순서를 따를 것)

1단계: 학생의 발화를 이해할 수 있는가?
  - 완전히 이해 불가 → PRE "not_understood" 반환
  - 부분적으로 이해 → DYN (되묻기 생성 필요)
  - 이해 가능 → 2단계로

2단계: 현재 대화 흐름에서 PRE 웨이포인트에 해당하는가?
  - 해당함 → PRE + 해당 category 반환
  - 해당하지 않음 → 3단계로

3단계: 동적 응답이 필요하다 → DYN + 감정 분석 결과 반환

## 대화 기록
{json.dumps(conversation_history, ensure_ascii=False) if conversation_history else "(첫 번째 턴)"}

## 학생의 현재 발화
"{student_input}"

## 출력 규칙 (매우 중요)
- 반드시 JSON만 출력하라. 다른 텍스트는 일절 금지.
- 가능한 한 짧게 출력하라. 짧을수록 좋다.

### 출력 형식 (3가지 중 하나를 선택):

형식1 - PRE (사전녹음 사용):
{{"route":"PRE","category":"카테고리명"}}

형식2 - DYN 부분 이해 (되묻기 생성 필요):
{{"route":"DYN","understood":"partial","heard":"들린 부분","direction":"NPC가 어떻게 되물어야 하는지"}}

형식3 - DYN 완전 이해 (동적 응답 생성 필요):
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도1~10,"sub_emotion":"보조감정또는null","sub_intensity":강도1~10또는null,"audio_tags":"[태그1][태그2]","direction":"NPC가 어떻게 반응해야 하는지 간략 설명"}}

JSON만 출력하라. 설명, 마크다운, 줄바꿈 금지."""

    return prompt

# ============================================================
# 라우트
# ============================================================
@app.route('/roleplay-test')
def roleplay_test_page():
    return render_template('roleplay/roleplay_test.html')

@app.route('/api/analyst-test', methods=['POST'])
def analyst_test():
    """분석가 테스트 엔드포인트"""
    if not analyst_model:
        return jsonify({"error": "Gemini 모델이 설정되지 않았습니다."}), 500

    data = request.get_json(silent=True) or {}
    student_input = data.get('student_input', '').strip()
    conversation_history = data.get('conversation_history', [])

    if not student_input:
        return jsonify({"error": "학생 입력이 비어있습니다."}), 400

    try:
        # 분석가 프롬프트 생성
        prompt = build_analyst_prompt(TEST_SCENARIO, conversation_history, student_input)

        # Gemini 호출
        response = analyst_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
                response_mime_type="application/json"
            )
        )

        raw_text = response.text.strip()

        # JSON 파싱 시도
        # Gemini가 붙이는 불필요한 텍스트 + 마크다운 코드블록 제거
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        # "Here is the JSON requested:" 같은 접두어 제거 — JSON은 { 로 시작함
        if '{' in clean:
            clean = clean[clean.index('{'):]
        if '}' in clean:
            clean = clean[:clean.rindex('}') + 1]

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = {"parse_error": True, "raw": raw_text}

        return jsonify({
            "success": True,
            "analyst_response": parsed,
            "raw_text": raw_text,
            "prompt_used": prompt  # 디버깅용: 실제 프롬프트 확인
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500

@app.route('/api/scenario-info')
def scenario_info():
    """현재 테스트 시나리오 정보 반환"""
    return jsonify(TEST_SCENARIO)