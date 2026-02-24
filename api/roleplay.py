"""
롤플레이 프로덕션 대화 엔진 (roleplay.py)
- 세션/팀 컨텍스트 기반 AI 대화
- 분석가 → 연기자 → TTS 체인
- 매 턴 conversation_logs 기록
- 시나리오/PRE를 DB에서 로드
"""
import os
import json
import pathlib
import traceback
import time
import base64
from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect

import psycopg2
import psycopg2.extras
from google import genai
from google.genai import types
import requests as http_requests

# ============================================================
# Flask 앱 설정
# ============================================================
BASE_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-prod')

DATABASE_URL = os.environ.get('POSTGRES_URL')

# ============================================================
# Gemini 클라이언트
# ============================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ [roleplay.py] Gemini 클라이언트 로드 완료")
    except Exception as e:
        print(f"🚨 [roleplay.py] Gemini 클라이언트 실패: {e}")

# ============================================================
# ElevenLabs TTS
# ============================================================
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_MODEL_ID = "eleven_v3"

def call_elevenlabs_tts(text, voice_id=None):
    """ElevenLabs TTS → MP3 bytes. 실패 시 None."""
    if not ELEVENLABS_API_KEY:
        print("⚠️ ELEVENLABS_API_KEY 미설정")
        return None

    voice_id = voice_id or "xi3rF0t7dg7uN2M0WUhr"  # 기본 음성
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        resp = http_requests.post(
            url,
            headers={"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY},
            json={"text": text, "model_id": ELEVENLABS_MODEL_ID, "language_code": "ko"},
            params={"output_format": "mp3_44100_128"},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"🚨 ElevenLabs {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"🚨 ElevenLabs 요청 실패: {e}")
        return None

# ============================================================
# DB / Auth 헬퍼
# ============================================================
def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"🚨 DB 연결 오류: {e}")
        return None

def player_required(f):
    """세션에 로그인한 학생만 허용"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "로그인 필요"}), 401
        return f(*args, **kwargs)
    return wrapper

# ============================================================
# DB에서 시나리오 + PRE 로드
# ============================================================
def load_scenario_from_db(scenario_id, conn):
    """DB에서 시나리오 정보를 roleplay_test.py 형식으로 변환"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM rp_scenarios WHERE id = %s", (scenario_id,))
        sc = cur.fetchone()
        if not sc:
            return None

        # PRE 카테고리 로드
        cur.execute("""
            SELECT DISTINCT category, guide_text 
            FROM rp_pre_recordings 
            WHERE scenario_id = %s
        """, (scenario_id,))
        pre_rows = cur.fetchall()
        pre_categories = {}
        for row in pre_rows:
            pre_categories[row['category']] = row['guide_text'] or ''

        # npc_knowledge 파싱
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
            "conversation_goal": sc.get('conversation_goal', ''),
            "voice_id": sc.get('npc_voice_id'),
            "temperature": sc.get('temperature', 0.3),
            "thinking_level": sc.get('thinking_level', 'LOW'),
            "pre_categories": pre_categories
        }

def load_conversation_history(team_id, scenario_id, conn):
    """DB에서 이 팀+시나리오의 대화 기록 로드"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT speaker, message_text, actor_line
            FROM rp_conversation_logs
            WHERE team_id = %s AND scenario_id = %s
            ORDER BY turn_number ASC
        """, (team_id, scenario_id))
        rows = cur.fetchall()

    history = []
    for row in rows:
        if row['speaker'] == 'player':
            history.append({"role": "player", "text": row['message_text'] or ''})
        elif row['speaker'] == 'npc':
            history.append({"role": "npc", "text": row['actor_line'] or row['message_text'] or ''})
    return history

def get_current_turn(team_id, scenario_id, conn):
    """현재 턴 번호 조회 (player 턴 기준)"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM rp_conversation_logs
            WHERE team_id = %s AND scenario_id = %s AND speaker = 'player'
        """, (team_id, scenario_id))
        return cur.fetchone()[0]

def save_turn(conn, team_id, scenario_id, turn_number, speaker, 
              message_text=None, player_user_id=None, audio_url=None,
              analyst_json=None, actor_line=None,
              tts_audio_base64=None, pre_audio_url=None):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rp_conversation_logs 
            (team_id, scenario_id, turn_number, speaker, player_user_id,
             message_text, audio_url, analyst_json, actor_line,
             tts_audio_base64, pre_audio_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            team_id, scenario_id, turn_number, speaker, player_user_id,
            message_text, audio_url,
            json.dumps(analyst_json, ensure_ascii=False) if analyst_json else None,
            actor_line, tts_audio_base64, pre_audio_url
        ))
    conn.commit()

# ============================================================
# 팀/세션 검증
# ============================================================
def validate_player_session(user_id, session_id, conn):
    """
    이 학생이 이 세션의 팀 멤버인지 확인.
    반환: {"team_id": int, "team_code": str, "session_status": str} 또는 None
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT t.id as team_id, t.team_code, s.status as session_status
            FROM rp_session_members m
            JOIN rp_session_teams t ON m.team_id = t.id
            JOIN rp_sessions s ON t.session_id = s.id
            WHERE s.id = %s AND m.user_id = %s
        """, (session_id, user_id))
        return cur.fetchone()

# ============================================================
# 프롬프트 빌더 (roleplay_test.py에서 복사 + DB 시나리오 호환)
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

## NPC 도메인 지식 (PRE 판단 시 반드시 참고)
{json.dumps(scenario['npc'].get('knowledge', {}), ensure_ascii=False, indent=2) if scenario['npc'].get('knowledge') else '(없음)'}
※ 도메인 지식과 PRE 카테고리가 충돌하면 PRE를 사용하지 마라. DYN으로 처리하라.
예: 메뉴에 "온도":["아이스"]만 있는 음료를 주문했으면, cold_or_hot PRE를 사용하지 말고 다음 단계로 넘어가라.

## 대화 목표
{scenario.get('conversation_goal', '')}

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
  - 완전히 이해 불가 → PRE "not_understood"
  - 부분적으로 이해 → DYN (되묻기)
  - 이해 가능 → 2단계로
2단계: PRE 웨이포인트에 해당하는가?
  - 해당함 → PRE + category
  - 해당하지 않음 → 3단계로
3단계: DYN + 감정 분석

## 대화 기록
{history_text}

## 학생의 현재 발화
"{student_input}"

## boundary 판단 (매 턴 반드시 포함)

너는 이 NPC의 입장에서 판단한다.
이 NPC가 지금 이 상황에서 이 말을 듣고 당황하거나 불편한가?

boundary = 0: NPC가 자연스럽게 받아들일 수 있는 말
boundary = 1: NPC가 당황하거나 불편해하거나 이해할 수 없는 말

판단 시 고려할 것:
- NPC의 성격과 직업
- 현재 대화 상황과 관계
- 대화의 목적 (위 "대화 목표" 참조)
- 외국어만 사용하는 경우 → 반드시 boundary=1
- 한국어에 흡수된 외래어 (아메리카노, 컴퓨터 등) → boundary=0

## 목적 달성 판단 (매 턴 반드시 포함)
대화 목표: "{scenario.get('conversation_goal', '')}"
※ 형식4(음성 인식 실패/외국어)를 사용하는 경우: goal_achieved는 무조건 false. 내용을 이해하거나 번역하려 하지 마라.
위 경우가 아닐 때만, 대화 기록 전체를 보고 학생이 대화 목표를 실질적으로 달성했는지 판단하라.
goal_achieved = true: 학생이 목표를 달성한 대화가 이번 턴에서 완성됨
goal_achieved = false: 아직 목표 미달성
주의: 목표에 근접했더라도 핵심 행위가 완료되지 않았으면 false.
예: "카페에서 음료 주문"이 목표라면, 실제로 음료를 말해야 true. "안녕하세요"만으로는 false.

## 출력 형식 (3가지 중 하나):

형식1 - PRE:
{{"route":"PRE","category":"카테고리명","boundary":0, "goal_achieved":false}}

형식2 - DYN 부분 이해:
{{"route":"DYN","understood":"partial","heard":"들린 부분","direction":"되묻기 방향","boundary":0또는1, "goal_achieved":false}}

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","boundary":0또는1, "goal_achieved":false}}

JSON만 출력하라. 설명, 마크다운, 줄바꿈 금지."""
    
    return prompt


def build_analyst_prompt_for_audio(scenario, conversation_history):
    """음성 입력용 — 텍스트 버전에 STT 지시 추가"""
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

## 🎤 중요: 음성 입력 (이 규칙은 절대적이다)
첨부된 오디오 파일은 학생이 직접 말한 음성이다.
1. 먼저 음성을 듣고 한국어인지 판별하라.
2. 한국어가 아닌 경우 (영어, 이탈리아어, 기타 외국어): 형식4(음성 인식 실패)로 처리하라. 절대로 한국어로 추측하지 마라.
3. 한국어인 경우: 텍스트로 변환하여 "transcribed_text"에 포함하라.
4. 그 텍스트를 기반으로 아래 분석을 수행하라.
※ 학생은 한국어 학습자이므로 발음이 부정확할 수 있다. 관대하게 인식하되, 한국어가 전혀 들리지 않으면 추측하지 마라.
※ 음성이 너무 짧거나(1초 미만), 잡음만 있거나, 한국어가 아닌 경우 → 형식4를 사용하라.
※ 판단 기준: 음성에서 한국어 단어가 1개라도 명확히 들리면 한국어로 처리. 한국어 단어가 전혀 안 들리면 무조건 형식4.

## NPC 정보
- 이름: {npc['name']}
- 나이: {npc['age']}세
- 직업: {npc['job']}
- 성격: {npc['personality']}

## 현재 상황
{scenario['situation']}

## NPC 도메인 지식 (PRE 판단 시 반드시 참고)
{json.dumps(scenario['npc'].get('knowledge', {}), ensure_ascii=False, indent=2) if scenario['npc'].get('knowledge') else '(없음)'}
※ 도메인 지식과 PRE 카테고리가 충돌하면 PRE를 사용하지 마라. DYN으로 처리하라.
예: 메뉴에 "온도":["아이스"]만 있는 음료를 주문했으면, cold_or_hot PRE를 사용하지 말고 다음 단계로 넘어가라.

## 대화 목표
{scenario.get('conversation_goal', '')}

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
{history_text}

## boundary 판단 (매 턴 반드시 포함)

너는 이 NPC의 입장에서 판단한다.
이 NPC가 지금 이 상황에서 이 말을 듣고 당황하거나 불편한가?

boundary = 0: NPC가 자연스럽게 받아들일 수 있는 말
boundary = 1: NPC가 당황하거나 불편해하거나 이해할 수 없는 말

판단 시 고려할 것:
- NPC의 성격과 직업
- 현재 대화 상황과 관계
- 대화의 목적 (위 "대화 목표" 참조)
- 외국어만 사용하는 경우 → 반드시 boundary=1
- 한국어에 흡수된 외래어 (아메리카노, 컴퓨터 등) → boundary=0

## 목적 달성 판단 (매 턴 반드시 포함)
대화 목표: "{scenario.get('conversation_goal', '')}"
이 대화 기록 전체를 보고, 학생이 대화 목표를 실질적으로 달성했는지 판단하라.
goal_achieved = true: 학생이 목표를 달성한 대화가 이번 턴에서 완성됨
goal_achieved = false: 아직 목표 미달성
주의: 목표에 근접했더라도 핵심 행위가 완료되지 않았으면 false.
예: "카페에서 음료 주문"이 목표라면, 실제로 음료를 말해야 true. "안녕하세요"만으로는 false.

## 출력 형식 (4가지 중 하나):

형식1 - PRE:
{{"route":"PRE","category":"카테고리명","transcribed_text":"인식된 텍스트", "boundary":0, "goal_achieved":false}}

형식2 - DYN 부분 이해:
{{"route":"DYN","understood":"partial","heard":"들린 부분","direction":"되묻기 방향","transcribed_text":"인식된 텍스트", "boundary":0또는1, "goal_achieved":false}}

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","transcribed_text":"인식된 텍스트", "boundary":0또는1, "goal_achieved":false}}

형식4 - 음성 인식 실패:
{{"route":"PRE","category":"not_understood","transcribed_text":"","boundary":1,"goal_achieved":false}}

JSON만 출력하라. 설명, 마크다운, 줄바꿈 금지."""

    return prompt


def build_actor_prompt(scenario, conversation_history, analyst_json, student_input):
    npc = scenario["npc"]

    history_text = ""
    if conversation_history:
        for turn in conversation_history:
            role = "손님" if turn.get("role") == "player" else f"{npc['name']}(나)"
            history_text += f"{role}: {turn.get('text', '')}\n"
    else:
        history_text = "(첫 번째 턴)"

    # NPC 도메인 지식 텍스트화
    knowledge = npc.get('knowledge', {})
    if isinstance(knowledge, dict) and knowledge:
        knowledge_text = json.dumps(knowledge, ensure_ascii=False, indent=2)
    else:
        knowledge_text = "(없음)"

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
   분석가가 제공한 audio_tags를 대사 텍스트 안에 넣어라.
   예: "[laughing] 아 네, 카푸치노는 원래 따뜻한 거예요. [warmly] 맛있게 드세요!"

2. **1~2문장으로 짧게.** 진짜 대화처럼 짧게 말하라. 길게 설명하지 마라.

3. **캐릭터를 유지하라.** {npc['name']}은(는) {npc['age']}세 {npc['job']}이다. 자연스러운 말투를 쓰라.

4. **NPC 도메인 지식을 활용하라.** {npc['job']}이(가) 당연히 아는 정보는 자연스럽게 사용하라.

5. **direction을 충실히 따르되, 대사는 네가 직접 만들어라.** direction은 지시일 뿐, 그대로 읽지 마라.

## 출력
대사 텍스트만 출력하라. 따옴표, 설명, JSON 등 다른 것은 일절 금지.
audio tags가 포함된 순수 대사 텍스트만. 설명, 마크다운, 줄바꿈 금지."""

    return prompt


# ============================================================
# AI 체인 실행
# ============================================================
def run_analyst(scenario, conversation_history, student_input):
    """분석가 호출 (텍스트 입력)"""
    prompt = build_analyst_prompt(scenario, conversation_history, student_input)

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
    analyst_latency = int((time.time() - analyst_start) * 1000)

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

    return parsed, analyst_latency, prompt


def run_analyst_audio(scenario, conversation_history, audio_bytes, mime_type):
    """분석가 호출 (음성 입력)"""
    prompt_text = build_analyst_prompt_for_audio(scenario, conversation_history)

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
    analyst_latency = int((time.time() - analyst_start) * 1000)

    clean = raw_text.replace("```json", "").replace("```", "").strip()
    if '{' in clean:
        clean = clean[clean.index('{'):]
    if '}' in clean:
        clean = clean[:clean.rindex('}') + 1]

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {"parse_error": True, "raw": raw_text}

    return parsed, analyst_latency, prompt_text


def run_actor(scenario, conversation_history, analyst_json, student_input):
    """연기자 호출"""
    actor_prompt = build_actor_prompt(scenario, conversation_history, analyst_json, student_input)

    actor_start = time.time()
    actor_response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=actor_prompt,
        config=types.GenerateContentConfig(
            temperature=scenario.get('temperature', 0.5),
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(
                thinking_level=getattr(types.ThinkingLevel, scenario.get('thinking_level', 'LOW'), types.ThinkingLevel.LOW)
            )
        )
    )
    actor_raw = (actor_response.text or "").strip()
    actor_line = actor_raw.strip('"').strip("'")
    actor_latency = int((time.time() - actor_start) * 1000)

    return actor_line, actor_latency


def run_tts(text, voice_id=None):
    """TTS 호출 → base64 반환"""
    tts_start = time.time()
    tts_bytes = call_elevenlabs_tts(text, voice_id)
    tts_latency = int((time.time() - tts_start) * 1000)

    if tts_bytes:
        return base64.b64encode(tts_bytes).decode('utf-8'), tts_latency
    return None, tts_latency


# ============================================================
# PRE 오디오 URL 조회
# ============================================================
def get_pre_audio_url(scenario_id, category, conn):
    """PRE 카테고리의 랜덤 변형 오디오 URL 반환"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT cloudflare_url, transcript FROM rp_pre_recordings
            WHERE scenario_id = %s AND category = %s AND cloudflare_url IS NOT NULL
            ORDER BY RANDOM() LIMIT 1
        """, (scenario_id, category))
        row = cur.fetchone()
        if row:
            return row['cloudflare_url'], row['transcript']

        # URL 없으면 transcript만이라도
        cur.execute("""
            SELECT transcript FROM rp_pre_recordings
            WHERE scenario_id = %s AND category = %s
            ORDER BY RANDOM() LIMIT 1
        """, (scenario_id, category))
        row = cur.fetchone()
        if row:
            return None, row['transcript']

    return None, None

# ============================================================
# violations 계산
# ============================================================

def get_total_violations(team_id, scenario_id, conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT analyst_json FROM rp_conversation_logs
            WHERE team_id=%s AND scenario_id=%s AND speaker='player'
            ORDER BY turn_number ASC
        """, (team_id, scenario_id))
        rows = cur.fetchall()
    
    total = 0
    for row in rows:
        aj = row[0]
        if isinstance(aj, str):
            try: aj = json.loads(aj)
            except: continue
        if aj and aj.get('boundary') == 1:
            total += 1
    return total

def get_boundary_pre(conn):
    """공통 Boundary PRE 풀에서 랜덤 1개 반환"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT cloudflare_url, transcript FROM rp_pre_recordings
            WHERE category = 'boundary_pre'
            ORDER BY RANDOM() LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            return row['cloudflare_url'], row['transcript']
        return None, "네?"

def handle_npc_response(conn, scenario, conversation_history,
                        parsed, student_input, team_id, scenario_id, new_turn):
    """분석가 결과 → boundary 체크 → NPC 응답 결정 (공통 로직)"""

    actor_line = None
    actor_latency = None
    tts_audio_b64 = None
    tts_latency = None
    pre_audio_url = None
    pre_transcript = None
    is_exit = False
    npc_name = scenario['npc']['name']

    # ── Boundary 체크 ──
    boundary = parsed.get('boundary', 0)

    if boundary == 1:
        total_violations = get_total_violations(team_id, scenario_id, conn)

        if total_violations >= 4:
            # Exit DYN — 종료 대사
            parsed['direction'] = f"boundary Exit: 학생이 {total_violations}회 이탈. 대화를 끝내는 대사를 하라. NPC 성격에 맞게."
            parsed['main_emotion'] = '불쾌'
            parsed['audio_tags'] = '[sigh][frustrated]'

            actor_line, actor_latency = run_actor(
                scenario, conversation_history, parsed, student_input)
            voice_id = scenario.get('voice_id')
            if actor_line:
                tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)
            save_turn(conn, team_id, scenario_id, new_turn, 'npc',
                      message_text="[EXIT]",
                      actor_line=actor_line, tts_audio_base64=tts_audio_b64)
            is_exit = True


        elif total_violations >= 3:
            # Boundary DYN — 맥락 참조 대사
            parsed['direction'] = f"boundary DYN: 학생이 {total_violations}회 이탈. 되묻기/저의확인/목표환기 중 상황에 맞게. 불쾌한 감정으로."
            parsed['main_emotion'] = '불쾌'
            parsed['audio_tags'] = '[frustrated][sigh]'

            actor_line, actor_latency = run_actor(
                scenario, conversation_history, parsed, student_input)
            voice_id = scenario.get('voice_id')
            if actor_line:
                tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)

            save_turn(conn, team_id, scenario_id, new_turn, 'npc',
                      actor_line=actor_line, tts_audio_base64=tts_audio_b64)

        else:
            # Boundary PRE — "네?" "뭐요?" 즉각 반환
            pre_audio_url, pre_transcript = get_pre_audio_url(
                scenario_id, "boundary_pre", conn)

            if not pre_audio_url:
                pre_audio_url, pre_transcript = get_boundary_pre(conn)

            save_turn(conn, team_id, scenario_id, new_turn, 'npc',
                      message_text="[BOUNDARY_PRE]",
                      actor_line=pre_transcript or "네?",
                      pre_audio_url=pre_audio_url)

        return {
            "actor_line": actor_line, "actor_latency": actor_latency,
            "tts_audio_b64": tts_audio_b64, "tts_latency": tts_latency,
            "pre_audio_url": pre_audio_url, "pre_transcript": pre_transcript,
            "is_exit": is_exit, "npc_name": npc_name
        }

    # ── 정상 흐름 (boundary=0) ──
    total_violations = get_total_violations(team_id, scenario_id, conn)
    if total_violations > 0:
        aftereffect = ""
        if total_violations >= 3:
            aftereffect = "직전에 불쾌한 상황이 있었다. 불쾌하고 사무적인 톤으로. [sigh] [flatly] 활용."
        elif total_violations >= 1:
            aftereffect = "직전에 당황스러운 상황이 있었다. 약간 머뭇거리는 톤으로. [hesitates] [pause] 활용."

        if aftereffect and parsed.get('direction'):
            parsed['direction'] = aftereffect + " " + parsed['direction']
        elif aftereffect:
            parsed['direction'] = aftereffect

    # ── Goal Achievement 체크 ──
    goal_achieved = parsed.get('goal_achieved', False)
    if goal_achieved is True or goal_achieved == 'true':
        # direction에 마무리 인사 지시 추가
        farewell_direction = "대화 목표가 달성되었다. 자연스러운 마무리 인사를 하라. NPC 성격에 맞게 따뜻하게 마무리."
        if parsed.get('direction'):
            parsed['direction'] = farewell_direction + " " + parsed['direction']
        else:
            parsed['direction'] = farewell_direction
        # PRE인 경우에도 DYN으로 전환 (마무리 대사가 필요하므로)
        parsed['route'] = 'DYN'
        if not parsed.get('audio_tags'):
            parsed['audio_tags'] = '[warmly]'

    if parsed.get("route") == "PRE":
        pre_audio_url, pre_transcript = get_pre_audio_url(
            scenario_id, parsed.get("category", ""), conn)
        save_turn(conn, team_id, scenario_id, new_turn, 'npc',
                  message_text=f"[PRE:{parsed.get('category','')}]",
                  actor_line=pre_transcript, pre_audio_url=pre_audio_url)

    elif parsed.get("route") == "DYN":
        actor_line, actor_latency = run_actor(
            scenario, conversation_history, parsed, student_input)
        voice_id = scenario.get('voice_id')
        if actor_line:
            tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)
        
        # [GOAL_ACHIEVED] 마커 저장
        npc_message_text = "[GOAL_ACHIEVED]" if parsed.get('goal_achieved', False) in (True, 'true') else None
        save_turn(conn, team_id, scenario_id, new_turn, 'npc',
                  message_text=npc_message_text,
                  actor_line=actor_line, tts_audio_base64=tts_audio_b64)

    return {
        "actor_line": actor_line, "actor_latency": actor_latency,
        "tts_audio_b64": tts_audio_b64, "tts_latency": tts_latency,
        "pre_audio_url": pre_audio_url, "pre_transcript": pre_transcript,
        "is_exit": False, "npc_name": npc_name,
        "goal_achieved": parsed.get('goal_achieved', False) in (True, 'true')        
    }

# ============================================================
# 페이지 라우트
# ============================================================
@app.route('/roleplay-play')
def roleplay_play_page():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('roleplay/roleplay_play.html')


# ============================================================
# API: 세션 정보 로드
# ============================================================
@app.route('/api/rp-play/session-info', methods=['GET'])
@player_required
def session_info():
    """세션+팀+시나리오 정보 반환"""
    session_id = request.args.get('session_id')
    user_id = session.get('user_id')

    if not session_id:
        return jsonify({"error": "session_id 필수"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500
    try:
        # 플레이어 검증
        player = validate_player_session(user_id, session_id, conn)
        if not player:
            return jsonify({"error": "이 세션의 팀 멤버가 아닙니다"}), 403
        if player['session_status'] != 'active':
            return jsonify({"error": f"세션 상태: {player['session_status']}"}), 400

        # 시나리오 목록 (순서대로)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ss.scenario_id, ss.order_num, sc.title, sc.npc_name
                FROM rp_session_scenarios ss
                JOIN rp_scenarios sc ON ss.scenario_id = sc.id
                WHERE ss.session_id = %s
                ORDER BY ss.order_num
            """, (session_id,))
            scenarios = cur.fetchall()

        # 각 시나리오별 현재 턴
        for sc in scenarios:
            sc['current_turn'] = get_current_turn(player['team_id'], sc['scenario_id'], conn)

        return jsonify({
            "team_id": player['team_id'],
            "team_code": player['team_code'],
            "session_status": player['session_status'],
            "scenarios": scenarios,
            "max_turns": 8
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API: 텍스트 입력 (디버깅 겸용)
# ============================================================
@app.route('/api/rp-play/send-text', methods=['POST'])
@player_required
def send_text():
    """텍스트 입력 → 분석가 → 연기자 → TTS → 로그 저장"""
    if not gemini_client:
        return jsonify({"error": "Gemini 미설정"}), 500

    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    scenario_id = int(data.get('scenario_id', 0))
    student_input = data.get('student_input', '').strip()

    if not all([session_id, scenario_id, student_input]):
        return jsonify({"error": "session_id, scenario_id, student_input 필수"}), 400

    user_id = session.get('user_id')
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        # 1. 플레이어 검증
        player = validate_player_session(user_id, session_id, conn)
        if not player:
            return jsonify({"error": "권한 없음"}), 403
        if player['session_status'] != 'active':
            return jsonify({"error": "세션이 활성 상태가 아닙니다"}), 400

        team_id = player['team_id']

        # 2. 턴 제한 확인
        current_turn = get_current_turn(team_id, scenario_id, conn)
        if current_turn >= 8:
            return jsonify({"error": "이 시나리오의 턴이 모두 소진되었습니다 (8턴)", "turn_limit_reached": True}), 400

        # 3. 시나리오 로드
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오를 찾을 수 없습니다"}), 404

        # 4. 대화 기록 로드
        conversation_history = load_conversation_history(team_id, scenario_id, conn)

        # 5. 분석가 호출
        parsed, analyst_latency, prompt = run_analyst(scenario, conversation_history, student_input)

        # 6. 플레이어 턴 저장
        new_turn = current_turn + 1
        save_turn(conn, team_id, scenario_id, new_turn, 'player',
                  message_text=student_input, player_user_id=user_id,
                  analyst_json=parsed)

        # 7. 응답 생성
        result = handle_npc_response(
            conn, scenario, conversation_history,
            parsed, student_input, team_id, scenario_id, new_turn)

        return jsonify({
            "success": True,
            "turn_number": new_turn,
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "actor_line": result["actor_line"],
            "actor_latency": result["actor_latency"],
            "tts_audio_base64": result["tts_audio_b64"],
            "tts_latency": result["tts_latency"],
            "pre_audio_url": result["pre_audio_url"],
            "pre_transcript": result["pre_transcript"],
            "is_exit": result["is_exit"],
            "npc_name": result["npc_name"],
            "turns_remaining": 8 - new_turn,
            "goal_achieved": result.get("goal_achieved", False)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"처리 실패: {str(e)}"}), 500
    finally:
        conn.close()


# ============================================================
# API: 음성 입력 (메인)
# ============================================================
@app.route('/api/rp-play/send-audio', methods=['POST'])
@player_required
def send_audio():
    """음성 입력 → 분석가(STT+분석) → 연기자 → TTS → 로그 저장"""
    if not gemini_client:
        return jsonify({"error": "Gemini 미설정"}), 500

    session_id = request.form.get('session_id')
    scenario_id = int(request.form.get('scenario_id', 0))
    audio_file = request.files.get('audio_file')
    mime_type = request.form.get('mime_type', 'audio/mp4')

    if not all([session_id, scenario_id, audio_file]):
        return jsonify({"error": "session_id, scenario_id, audio_file 필수"}), 400

    user_id = session.get('user_id')
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        # 1. 플레이어 검증
        player = validate_player_session(user_id, session_id, conn)
        if not player:
            return jsonify({"error": "권한 없음"}), 403
        if player['session_status'] != 'active':
            return jsonify({"error": "세션이 활성 상태가 아닙니다"}), 400

        team_id = player['team_id']

        # 2. 턴 제한
        current_turn = get_current_turn(team_id, scenario_id, conn)
        if current_turn >= 8:
            return jsonify({"error": "턴 소진 (8턴)", "turn_limit_reached": True}), 400

        # 3. 시나리오 + 대화기록 로드
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오 없음"}), 404

        conversation_history = load_conversation_history(team_id, scenario_id, conn)

        # 4. 오디오 읽기 + 분석가 호출
        audio_bytes = audio_file.read()
        parsed, analyst_latency, prompt = run_analyst_audio(
            scenario, conversation_history, audio_bytes, mime_type)

        transcribed_text = parsed.get("transcribed_text", "")

        # 5. 플레이어 턴 저장
        new_turn = current_turn + 1
        save_turn(conn, team_id, scenario_id, new_turn, 'player',
                  message_text=transcribed_text, player_user_id=user_id,
                  analyst_json=parsed)

        # 6. 응답 생성
        result = handle_npc_response(
            conn, scenario, conversation_history,
            parsed, transcribed_text or "(인식 실패)",
            team_id, scenario_id, new_turn)

        return jsonify({
            "success": True,
            "turn_number": new_turn,
            "transcribed_text": transcribed_text,
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "actor_line": result["actor_line"],
            "actor_latency": result["actor_latency"],
            "tts_audio_base64": result["tts_audio_b64"],
            "tts_latency": result["tts_latency"],
            "pre_audio_url": result["pre_audio_url"],
            "pre_transcript": result["pre_transcript"],
            "is_exit": result["is_exit"],
            "npc_name": result["npc_name"],
            "turns_remaining": 8 - new_turn,
            "goal_achieved": result.get("goal_achieved", False)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"음성 처리 실패: {str(e)}"}), 500
    finally:
        conn.close()

# ============================================================
# API: 대화 기록 조회 (팀 동기화용)
# ============================================================
@app.route('/api/rp-play/history', methods=['GET'])
@player_required
def get_history():
    """팀의 현재 시나리오 대화 기록 반환 (폴링용)"""
    session_id = request.args.get('session_id')
    scenario_id = request.args.get('scenario_id')
    user_id = session.get('user_id')

    if not all([session_id, scenario_id]):
        return jsonify({"error": "session_id, scenario_id 필수"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        player = validate_player_session(user_id, session_id, conn)
        if not player:
            return jsonify({"error": "권한 없음"}), 403

        team_id = player['team_id']

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT turn_number, speaker, message_text, actor_line,
                       analyst_json, created_at::text as created_at,
                       tts_audio_base64, pre_audio_url
                FROM rp_conversation_logs
                WHERE team_id = %s AND scenario_id = %s
                ORDER BY turn_number ASC, id ASC
            """, (team_id, int(scenario_id)))
            logs = cur.fetchall()

        current_turn = get_current_turn(team_id, int(scenario_id), conn)

        return jsonify({
            "logs": logs,
            "current_turn": current_turn,
            "turns_remaining": 8 - current_turn
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()