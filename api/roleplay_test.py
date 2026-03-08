from flask import Flask, request, jsonify, render_template, session
from google import genai
from google.genai import types
import os
import json
import pathlib
import traceback
import requests as http_requests
import base64
import time
import psycopg2
import psycopg2.extras

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'test-secret-key-change-me')

# ============================================================
# Gemini 클라이언트
# ============================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ [test] Gemini 클라이언트 로드 완료")
    except Exception as e:
        print(f"🚨 [test] 클라이언트 로드 실패: {e}")
else:
    print("⚠️ [test] GEMINI_API_KEY 미설정")

# ============================================================
# DB 연결
# ============================================================
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"🚨 [test] DB 연결 오류: {e}")
        return None

# ============================================================
# ElevenLabs TTS
# ============================================================
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_MODEL_ID = "eleven_v3"

def call_elevenlabs_tts(text, voice_id=None):
    if not ELEVENLABS_API_KEY:
        print("⚠️ ELEVENLABS_API_KEY 미설정 — TTS 건너뜀")
        return None

    voice_id = voice_id or "xi3rF0t7dg7uN2M0WUhr"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    payload = {"text": text, "model_id": ELEVENLABS_MODEL_ID, "language_code": "ko"}

    try:
        resp = http_requests.post(url, headers=headers, json=payload,
                                  params={"output_format": "mp3_44100_128"}, timeout=15)
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"🚨 ElevenLabs 오류 {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"🚨 ElevenLabs 요청 실패: {e}")
        return None


# ============================================================
# DB 로드 헬퍼 (프로덕션 roleplay.py와 동일)
# ============================================================
def load_scenario_from_db(scenario_id, conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM rp_scenarios WHERE id = %s", (scenario_id,))
        sc = cur.fetchone()
        if not sc:
            return None

        cur.execute("""
            SELECT DISTINCT category, guide_text 
            FROM rp_pre_recordings 
            WHERE scenario_id = %s
        """, (scenario_id,))
        pre_rows = cur.fetchall()
        pre_categories = {}
        for row in pre_rows:
            pre_categories[row['category']] = row['guide_text'] or ''

        npc_knowledge = sc.get('npc_knowledge')
        if isinstance(npc_knowledge, str):
            try:
                npc_knowledge = json.loads(npc_knowledge)
            except:
                npc_knowledge = {}

        return {
            "id": sc['id'],
            "npc": {
                "name": sc.get('npc_name', ''),
                "age": sc.get('npc_age', 0),
                "job": sc.get('npc_job', ''),
                "personality": sc.get('npc_personality', ''),
                "current_state": sc.get('npc_current_state', ''),
                "knowledge": npc_knowledge or {}
            },
            "situation": sc.get('situation', ''),
            "speech_style": sc.get('speech_style', '비격식 존댓말'),
            "voice_id": sc.get('npc_voice_id'),
            "temperature": sc.get('temperature', 0.3),
            "thinking_level": sc.get('thinking_level', 'LOW'),
            "pre_categories": pre_categories
        }


def load_goal_data(goal_id, conn):
    if not goal_id:
        return None
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT conversation_goal, npc_guidelines FROM rp_goals WHERE id = %s", (goal_id,))
        return cur.fetchone()


# ============================================================
# 프롬프트: 분석가 — 프로덕션 동기화
# ============================================================
def build_analyst_prompt(scenario, conversation_history, student_input):
    npc = scenario["npc"]
    pre_cats = scenario["pre_categories"]
    pre_list = "\n".join([f'  - "{key}": {desc}' for key, desc in pre_cats.items()])

    history_text = ""
    if conversation_history:
        for turn in conversation_history:
            role = "손님" if turn.get("role") == "player" else f"{npc['name']}(NPC)"
            history_text += f"{role}: {turn.get('text', '')}\n"
    else:
        history_text = "(첫 번째 턴)"

    prompt = f"""너는 롤플레이 게임의 "분석가"이다. 너의 역할은 플레이어(한국어 학습 중인 이탈리아 학생)의 발화를 분석하고, NPC가 어떻게 반응해야 하는지 판단하는 것이다.

## NPC 정보
- 이름: {npc['name']}
- 나이: {npc['age']}세
- 직업: {npc['job']}
- 성격: {npc['personality']}

## 현재 상황
{scenario['situation']}

## NPC 행동 방침 (반드시 따를 것)
{scenario.get('npc_guidelines', '') if scenario.get('npc_guidelines') else '(없음)'}

## NPC 도메인 지식 (PRE 판단 시 반드시 참고)
{json.dumps(scenario['npc'].get('knowledge', {}), ensure_ascii=False, indent=2) if scenario['npc'].get('knowledge') else '(없음)'}
※ 도메인 지식과 PRE 카테고리가 충돌하면 PRE를 사용하지 마라. DYN으로 처리하라.

## 대화 목표
{scenario.get('conversation_goal', '')}

## NPC 행동 방침 (direction 결정 시 반드시 참고)
{scenario.get('npc_guidelines', '') if scenario.get('npc_guidelines') else '(없음)'}
※ 행동 방침이 있으면, 대화 기록을 보고 NPC가 이미 해당 행동을 했는지 판단한 뒤 direction에 반영하라.

## 사용 가능한 PRE(사전녹음) 카테고리
아래 목록에 해당하는 상황이면 PRE를 우선 사용하라. 레이턴시 절약에 매우 중요하다.
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
  - 의미를 전혀 파악 불가 → 형식2 DYN (되묻기, intended=null)
  - 표현이 어색/부정확하지만 대충 이해 가능 → 형식2 DYN (확인, intended="추정 표현")
  - 완전히 이해 가능 → 2단계로
2단계: PRE 웨이포인트에 해당하는가?
  - 해당함 → 형식1 PRE + category
  - 해당하지 않음 → 3단계로
3단계: 형식3 DYN + 감정 분석

## 대화 기록
{history_text}

## 학생의 현재 발화
"{student_input}"

## boundary 판단 (매 턴 반드시 포함)
너는 이 NPC의 입장에서 판단한다.
boundary = 0: NPC가 자연스럽게 받아들일 수 있는 말
boundary = 1: NPC가 당황하거나 불편해하거나 이해할 수 없는 말

## 목적 달성 판단 (매 턴 반드시 포함)
대화 목표: "{scenario.get('conversation_goal', '')}"
goal_achieved = true: 목표 달성 완료
goal_achieved = false: 아직 미달성

## 출력 형식 (3가지 중 하나):

형식1 - PRE:
{{"route":"PRE","category":"카테고리명","boundary":0또는1,"goal_achieved":false}}

형식2 - DYN 부분 이해:
{{"route":"DYN","understood":"partial","heard":"학생이 쓴 표현","intended":"추정 표현 또는 null","direction":"방향","boundary":0또는1,"goal_achieved":false}}

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","boundary":0또는1,"goal_achieved":false}}

JSON만 출력하라. 설명, 마크다운, 줄바꿈 금지."""

    return prompt


# ============================================================
# 프롬프트: 연기자 — 프로덕션 동기화
# ============================================================
def build_actor_prompt(scenario, conversation_history, analyst_json, student_input):
    npc = scenario["npc"]

    history_text = ""
    if conversation_history:
        for turn in conversation_history:
            role = "손님" if turn.get("role") == "player" else f"{npc['name']}(나)"
            history_text += f"{role}: {turn.get('text', '')}\n"
    else:
        history_text = "(첫 번째 턴)"

    knowledge = npc.get('knowledge', {})
    knowledge_text = json.dumps(knowledge, ensure_ascii=False, indent=2) if isinstance(knowledge, dict) and knowledge else "(없음)"

    prompt = f"""너는 롤플레이 게임에서 NPC를 연기하는 "연기자"이다.
너는 분석가가 보내준 감정 가이드를 받아서, 그에 맞는 대사를 생성한다.

## 너의 캐릭터
- 이름: {npc['name']}
- 나이: {npc['age']}세
- 직업: {npc['job']}
- 성격: {npc['personality']}
- 현재 상태: {npc.get('current_state', '')}

## 현재 상황
{scenario['situation']}

## NPC 행동 방침 (반드시 따를 것)
{scenario.get('npc_guidelines', '') if scenario.get('npc_guidelines') else '(없음)'}

## NPC 도메인 지식 (너는 이것을 알고 있다)
{knowledge_text}

## 지금까지의 대화
{history_text}

## 손님(학생)이 방금 한 말
"{student_input}"

## 분석가의 감정 가이드 (반드시 따를 것)
{json.dumps(analyst_json, ensure_ascii=False)}

## 연기 규칙 (매우 중요)
1. **audio tags를 대사 안에 자연스럽게 삽입하라.**
2. **1~2문장으로 짧게.** 진짜 대화처럼 짧게 말하라.
3. **어휘 수준은 TOPIK 3급 이하로 제한하라.**
4. **캐릭터를 유지하라.**
5. **NPC 도메인 지식을 반드시 확인하고 정확히 따르라.**
7. **말투 규칙: {scenario.get('speech_style', '비격식 존댓말')}을 사용하라.**
8. **부분 이해 시 확인 패턴.** intended 값이 있으면 자연스럽게 확인하라.

## 출력
대사 텍스트만 출력하라. 따옴표, 설명, JSON 등 금지. audio tags 포함된 순수 대사 텍스트만."""

    return prompt


# ============================================================
# JSON 파싱 / Gemini / TTS 헬퍼
# ============================================================
def parse_gemini_json(raw_text):
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    if '{' in clean:
        clean = clean[clean.index('{'):]
    if '}' in clean:
        clean = clean[:clean.rindex('}') + 1]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw_text}


def run_analyst(scenario, conversation_history, student_input):
    prompt = build_analyst_prompt(scenario, conversation_history, student_input)
    start = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3, max_output_tokens=2048,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        )
    )
    raw = (response.text or "").strip()
    latency = int((time.time() - start) * 1000)
    return parse_gemini_json(raw), latency, prompt


def run_actor(scenario, conversation_history, analyst_json, student_input):
    prompt = build_actor_prompt(scenario, conversation_history, analyst_json, student_input)
    start = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.6, max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        )
    )
    raw = (response.text or "").strip()
    latency = int((time.time() - start) * 1000)
    return raw.strip('"').strip("'"), latency, prompt


def run_tts(text, voice_id=None):
    start = time.time()
    tts_bytes = call_elevenlabs_tts(text, voice_id)
    latency = int((time.time() - start) * 1000)
    if tts_bytes:
        return base64.b64encode(tts_bytes).decode('utf-8'), latency
    return None, latency


# ============================================================
# 페이지 라우트
# ============================================================
@app.route('/roleplay-test')
def roleplay_test_page():
    return render_template('roleplay/roleplay_test.html')


# ============================================================
# API: 시나리오 목록
# ============================================================
@app.route('/api/test/scenarios', methods=['GET'])
def test_get_scenarios():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, title, npc_name, npc_job, speech_style FROM rp_scenarios ORDER BY id")
            return jsonify({"scenarios": cur.fetchall()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API: 목표 목록
# ============================================================
@app.route('/api/test/goals', methods=['GET'])
def test_get_goals():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, title, target_expression, conversation_goal, npc_guidelines FROM rp_goals ORDER BY id")
            return jsonify({"goals": cur.fetchall()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API: 시나리오+목표 병합 로드
# ============================================================
@app.route('/api/test/load-config', methods=['GET'])
def test_load_config():
    scenario_id = request.args.get('scenario_id', type=int)
    goal_id = request.args.get('goal_id', type=int)
    if not scenario_id:
        return jsonify({"error": "scenario_id 필수"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500
    try:
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오 없음"}), 404
        if goal_id:
            goal_data = load_goal_data(goal_id, conn)
            if goal_data:
                if goal_data.get('conversation_goal'):
                    scenario['conversation_goal'] = goal_data['conversation_goal']
                if goal_data.get('npc_guidelines'):
                    scenario['npc_guidelines'] = goal_data['npc_guidelines']

        # first_speaker + opening PRE 로드
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT first_speaker FROM rp_scenarios WHERE id = %s", (scenario_id,))
            row = cur.fetchone()
            first_speaker = row['first_speaker'] if row else 'player'

            opening_transcript = None
            if first_speaker == 'npc':
                cur.execute("""
                    SELECT transcript FROM rp_pre_recordings
                    WHERE scenario_id = %s AND category = 'opening'
                    ORDER BY RANDOM() LIMIT 1
                """, (scenario_id,))
                pre_row = cur.fetchone()
                if pre_row:
                    opening_transcript = pre_row['transcript']

        return jsonify({
            "success": True,
            "scenario": scenario,
            "first_speaker": first_speaker,
            "opening_transcript": opening_transcript
        })    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# 메인: 3단계 체인 (텍스트)
# ============================================================
@app.route('/api/analyst-test', methods=['POST'])
def analyst_test():
    if not gemini_client:
        return jsonify({"error": "Gemini 미설정"}), 500

    data = request.get_json(silent=True) or {}
    student_input = data.get('student_input', '').strip()
    conversation_history = data.get('conversation_history', [])
    scenario_id = data.get('scenario_id')
    goal_id = data.get('goal_id')

    if not student_input:
        return jsonify({"error": "입력이 비어있습니다."}), 400
    if not scenario_id:
        return jsonify({"error": "시나리오를 선택해주세요."}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오 없음"}), 404
        if goal_id:
            goal_data = load_goal_data(goal_id, conn)
            if goal_data:
                if goal_data.get('conversation_goal'):
                    scenario['conversation_goal'] = goal_data['conversation_goal']
                if goal_data.get('npc_guidelines'):
                    scenario['npc_guidelines'] = goal_data['npc_guidelines']
        conn.close()
        conn = None

        # STAGE 1: 귀 (텍스트 패스스루)
        stt_text = student_input
        stt_latency = 0

        # STAGE 2: 분석가
        analyst_json, analyst_latency, analyst_prompt = run_analyst(scenario, conversation_history, stt_text)

        # STAGE 3: 연기자 (DYN만)
        actor_line = None
        actor_latency = None
        actor_prompt = None
        if analyst_json.get("route") == "DYN":
            actor_line, actor_latency, actor_prompt = run_actor(scenario, conversation_history, analyst_json, stt_text)

        # TTS
        tts_audio_b64 = None
        tts_latency = None
        if actor_line:
            tts_audio_b64, tts_latency = run_tts(actor_line, scenario.get('voice_id'))

        return jsonify({
            "success": True,
            "stt_text": stt_text, "stt_latency": stt_latency,
            "analyst_response": analyst_json, "analyst_latency": analyst_latency, "analyst_prompt": analyst_prompt,
            "actor_line": actor_line, "actor_latency": actor_latency, "actor_prompt": actor_prompt,
            "tts_audio_base64": tts_audio_b64, "tts_latency": tts_latency,
            "total_latency": stt_latency + analyst_latency + (actor_latency or 0) + (tts_latency or 0)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"처리 실패: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

# ============================================================
# 메인: 3단계 체인 (음성)
# ============================================================
@app.route('/api/analyst-test-audio', methods=['POST'])
def analyst_test_audio():
    if not gemini_client:
        return jsonify({"error": "Gemini 미설정"}), 500

    audio_file = request.files.get('audio_file')
    mime_type = request.form.get('mime_type', 'audio/mp4')
    conversation_history_str = request.form.get('conversation_history', '[]')
    scenario_id = request.form.get('scenario_id', type=int)
    goal_id = request.form.get('goal_id', type=int)

    if not audio_file:
        return jsonify({"error": "오디오 파일 없음"}), 400
    if not scenario_id:
        return jsonify({"error": "시나리오 미선택"}), 400

    try:
        conversation_history = json.loads(conversation_history_str)
    except json.JSONDecodeError:
        conversation_history = []

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오 없음"}), 404
        if goal_id:
            goal_data = load_goal_data(goal_id, conn)
            if goal_data:
                if goal_data.get('conversation_goal'):
                    scenario['conversation_goal'] = goal_data['conversation_goal']
                if goal_data.get('npc_guidelines'):
                    scenario['npc_guidelines'] = goal_data['npc_guidelines']
        conn.close()
        conn = None

        audio_bytes = audio_file.read()

        # ========== STAGE 1: 귀 (STT 전용) ==========
        stt_prompt = """너는 음성 인식 전문가이다. 첨부된 오디오를 듣고 한글로 전사하라.

절대 규칙:
- 들리는 소리를 있는 그대로 한글로 적어라.
- 문맥 기반 자동 교정 금지. 문법이 틀려도 들린 그대로.
- 외국어도 한글로 음차하라.
- 침묵/잡음만 있으면 빈 문자열을 반환하라.

출력: 전사된 텍스트만. 따옴표, 설명, JSON 금지. 순수 텍스트만."""

        stt_start = time.time()
        stt_response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                stt_prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256
            )
        )
        stt_text = (stt_response.text or "").strip().strip('"').strip("'")
        stt_latency = int((time.time() - stt_start) * 1000)

        # STT 실패 체크
        if not stt_text:
            return jsonify({
                "success": True,
                "stt_text": "",
                "stt_latency": stt_latency,
                "analyst_response": {"route": "DYN", "understood": False, "direction": "침묵/인식 실패"},
                "analyst_latency": 0,
                "actor_line": None, "actor_latency": None,
                "tts_audio_base64": None, "tts_latency": None,
                "total_latency": stt_latency
            })

        # ========== STAGE 2: 분석가 ==========
        analyst_json, analyst_latency, analyst_prompt = run_analyst(
            scenario, conversation_history, stt_text
        )

        # ========== STAGE 3: 연기자 (DYN만) ==========
        actor_line = None
        actor_latency = None
        actor_prompt = None
        if analyst_json.get("route") == "DYN":
            actor_line, actor_latency, actor_prompt = run_actor(
                scenario, conversation_history, analyst_json, stt_text
            )

        # ========== TTS ==========
        tts_audio_b64 = None
        tts_latency = None
        if actor_line:
            tts_audio_b64, tts_latency = run_tts(actor_line, scenario.get('voice_id'))

        return jsonify({
            "success": True,
            "stt_text": stt_text, "stt_latency": stt_latency,
            "analyst_response": analyst_json, "analyst_latency": analyst_latency, "analyst_prompt": analyst_prompt,
            "actor_line": actor_line, "actor_latency": actor_latency, "actor_prompt": actor_prompt,
            "tts_audio_base64": tts_audio_b64, "tts_latency": tts_latency,
            "total_latency": stt_latency + analyst_latency + (actor_latency or 0) + (tts_latency or 0)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"음성 처리 실패: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()