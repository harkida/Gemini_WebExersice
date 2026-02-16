from flask import Flask, request, jsonify, render_template, session
from google import genai
from google.genai import types
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

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ 분석가 클라이언트 로드 완료")
    except Exception as e:
        print(f"🚨 클라이언트 로드 실패: {e}")
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
def build_analyst_prompt_for_audio(scenario, conversation_history):
    """음성 입력용 분석가 프롬프트 — 텍스트 버전에 STT 지시를 추가"""
    npc = scenario["npc"]
    pre_cats = scenario["pre_categories"]
    pre_list = "\n".join([f'  - "{key}": {desc}' for key, desc in pre_cats.items()])

    prompt = f"""너는 롤플레이 게임의 "분석가"이다. 너의 역할은 플레이어(한국어 학습 중인 이탈리아 학생)의 발화를 분석하고, NPC가 어떻게 반응해야 하는지 판단하는 것이다.

## 🎤 중요: 음성 입력
첨부된 오디오 파일은 학생이 직접 말한 음성이다.
1. 먼저 음성을 듣고 한국어인지 판별하라.
2. 한국어가 아닌 경우 (영어, 이탈리아어, 기타 외국어): 형식4(음성 인식 실패)로 처리하라. 절대로 한국어로 추측하지 마라.
3. 한국어인 경우: 텍스트로 변환하여 "transcribed_text"에 포함하라.
4. 그 텍스트를 기반으로 아래 분석을 수행하라.
※ 학생은 한국어 학습자이므로 발음이 부정확할 수 있다. 관대하게 인식하되, 한국어가 전혀 들리지 않으면 추측하지 마라.
※ 음성이 너무 짧거나(1초 미만), 잡음만 있거나, 한국어가 아닌 경우 → 형식4를 사용하라.

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
{pre_list}

## 감정 프레임워크
- 보통 (neutral)
- 행복 → 안도 / 웃김 / 감동 / 통쾌함
- 분노 → 불쾌 / 증오 / 권태
- 슬픔 → 그리움 / 후회 / 절망
- 불안 → 무서움 / 걱정 / 초조
- 놀람 → 당황 / 혼란 / 감탄

## 판단 우선순위 (반드시 이 순서를 따를 것)
1단계: 학생의 발화를 이해할 수 있는가?
  - 완전히 이해 불가 → PRE "not_understood"
  - 부분적으로 이해 → DYN (되묻기)
  - 이해 가능 → 2단계로
2단계: PRE 웨이포인트에 해당하는가?
  - 해당함 → PRE + category
  - 해당하지 않음 → 3단계로
3단계: DYN + 감정 분석

## 대화 기록
{json.dumps(conversation_history, ensure_ascii=False) if conversation_history else "(첫 번째 턴)"}

## 출력 형식 (4가지 중 하나 선택):

형식1 - PRE:
{{"route":"PRE","category":"카테고리명","transcribed_text":"인식된 텍스트"}}

형식2 - DYN 부분 이해:
{{"route":"DYN","understood":"partial","heard":"들린 부분","direction":"되묻기 방향","transcribed_text":"인식된 텍스트"}}

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","transcribed_text":"인식된 텍스트"}}

형식4 - 음성 인식 실패 (잡음만 들리거나 아무 말도 안 한 경우):
{{"route":"PRE","category":"not_understood","transcribed_text":""}}

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
    if not gemini_client:
        return jsonify({"error": "Gemini 모델이 설정되지 않았습니다."}), 500

    data = request.get_json(silent=True) or {}
    student_input = data.get('student_input', '').strip()
    conversation_history = data.get('conversation_history', [])

    if not student_input:
        return jsonify({"error": "학생 입력이 비어있습니다."}), 400

    try:
        import time
        # 분석가 프롬프트 생성
        prompt = build_analyst_prompt(TEST_SCENARIO, conversation_history, student_input)

        # Gemini 호출 (분석가)
        analyst_start = time.time()
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                )
            )
        )

        raw_text = (response.text or "").strip()

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

        analyst_latency = int((time.time() - analyst_start) * 1000)
        # ============================================================
        # 연기자 체인: DYN일 때만 연기자 호출
        # ============================================================
        actor_line = None
        actor_raw = None
        actor_latency = None

        if parsed.get("route") == "DYN":
            import time
            actor_start = time.time()

            actor_prompt = build_actor_prompt(
                TEST_SCENARIO, conversation_history, parsed, student_input
            )

            actor_response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=actor_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.LOW
                    )
                )
            )

            actor_raw = (actor_response.text or "").strip()
            # 따옴표 감싸기 제거
            actor_line = actor_raw.strip('"').strip("'")
            actor_latency = int((time.time() - actor_start) * 1000)

        return jsonify({
            "success": True,
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "raw_text": raw_text,
            "actor_line": actor_line,
            "actor_latency": actor_latency,
            "prompt_used": prompt
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini 호출 실패: {str(e)}"}), 500


@app.route('/api/analyst-test-audio', methods=['POST'])
def analyst_test_audio():
    """음성 입력 → 분석가 테스트 엔드포인트"""
    if not gemini_client:
        return jsonify({"error": "Gemini 클라이언트 미설정"}), 500

    audio_file = request.files.get('audio_file')
    mime_type = request.form.get('mime_type', 'audio/mp4')
    conversation_history_str = request.form.get('conversation_history', '[]')

    if not audio_file:
        return jsonify({"error": "오디오 파일이 없습니다."}), 400

    try:
        conversation_history = json.loads(conversation_history_str)
    except json.JSONDecodeError:
        conversation_history = []

    try:
        import time

        # 오디오 바이트 읽기
        audio_bytes = audio_file.read()

        # 분석가 프롬프트 생성 (음성용 — student_input 자리에 지시 추가)
        prompt_text = build_analyst_prompt_for_audio(TEST_SCENARIO, conversation_history)

        # Gemini에 오디오 + 프롬프트 함께 전달
        analyst_start = time.time()
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt_text
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                )
            )
        )

        raw_text = (response.text or "").strip()

        # JSON 파싱
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        if '{' in clean:
            clean = clean[clean.index('{'):]
        if '}' in clean:
            clean = clean[:clean.rindex('}') + 1]

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = {"parse_error": True, "raw": raw_text}

        analyst_latency = int((time.time() - analyst_start) * 1000)

        # 인식된 텍스트 추출
        transcribed_text = parsed.get("transcribed_text", "(인식 실패)")

        # 연기자 체인: DYN일 때만
        actor_line = None
        actor_latency = None

        if parsed.get("route") == "DYN":
            actor_start = time.time()
            actor_prompt = build_actor_prompt(
                TEST_SCENARIO, conversation_history, parsed, transcribed_text
            )
            actor_response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=actor_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    )
                )
            )
            actor_raw = (actor_response.text or "").strip()
            actor_line = actor_raw.strip('"').strip("'")
            actor_latency = int((time.time() - actor_start) * 1000)

        return jsonify({
            "success": True,
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "transcribed_text": transcribed_text,
            "raw_text": raw_text,
            "actor_line": actor_line,
            "actor_latency": actor_latency,
            "prompt_used": prompt_text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"음성 처리 실패: {str(e)}"}), 500


@app.route('/api/scenario-info')
def scenario_info():
    """현재 테스트 시나리오 정보 반환"""
    return jsonify(TEST_SCENARIO)

def build_actor_prompt(scenario, conversation_history, analyst_json, student_input):
    npc = scenario["npc"]

    # 대화 기록을 읽기 쉬운 텍스트로 변환
    history_text = ""
    if conversation_history:
        for turn in conversation_history:
            role = "손님" if turn.get("role") == "player" else "점원(나)"
            history_text += f"{role}: {turn.get('text', '')}\n"
    else:
        history_text = "(첫 번째 턴)"

    prompt = f"""너는 롤플레이 게임에서 NPC를 연기하는 "연기자"이다.
너는 분석가가 보내준 감정 가이드를 받아서, 그에 맞는 대사를 생성한다.

## 너의 캐릭터
- 이름: {npc['name']}
- 나이: {npc['age']}세
- 직업: {npc['job']}
- 성격: {npc['personality']}

## 현재 상황
{scenario['situation']}

## NPC 도메인 지식 (너는 이것을 알고 있다)
- 메뉴: 아메리카노(핫/아이스, 4500원), 카페라떼(핫/아이스, 5000원), 카푸치노(핫만, 5000원), 녹차라떼(핫/아이스, 5500원), 바닐라라떼(핫/아이스, 5500원)
- 사이즈: Regular(기본), Large(+500원). "Tall", "Grande" 같은 건 없음
- 결제: 카드, 현금, 카카오페이
- 와이파이: 비밀번호는 영수증 하단에 인쇄됨
- 화장실: 매장 안쪽 왼편
- 디카페인: 아메리카노, 카페라떼만 가능 (+500원)
- 오늘의 추천: 바닐라라떼 (신메뉴)

## 지금까지의 대화
{history_text}

## 손님(학생)이 방금 한 말
"{student_input}"

## 분석가의 감정 가이드 (반드시 따를 것)
{json.dumps(analyst_json, ensure_ascii=False)}

## 연기 규칙 (매우 중요)

1. **audio tags를 대사 안에 자연스럽게 삽입하라.**
   분석가가 제공한 audio_tags를 대사 텍스트 안에 넣어라.
   예: "[laughing] 아 네, 카푸치노는 원래 따뜻한 거예요. [warmly] 맛있게 드세요!"

2. **1~2문장으로 짧게.** 진짜 대화처럼 짧게 말하라. 길게 설명하지 마라.

3. **캐릭터를 유지하라.** 김수진은 25세 카페 점원이다. 격식체("~요")를 쓰되 자연스럽게.

4. **NPC 도메인 지식을 활용하라.** 카페 점원이 당연히 아는 정보는 자연스럽게 사용하라.
   예: "카푸치노요? 카푸치노는 따뜻한 것만 있어요~"

5. **direction을 충실히 따르되, 대사는 네가 직접 만들어라.** direction은 지시일 뿐, 그대로 읽지 마라.

## 출력
대사 텍스트만 출력하라. 따옴표, 설명, JSON 등 다른 것은 일절 금지.
audio tags가 포함된 순수 대사 텍스트만."""

    return prompt