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
            "speech_style": sc.get('speech_style', '비격식 존댓말'),
            "voice_id": sc.get('npc_voice_id'),
            "temperature": sc.get('temperature', 0.3),
            "thinking_level": sc.get('thinking_level', 'LOW'),
            "pre_categories": pre_categories
        }

def load_goal_data(goal_id, conn):
    """세션의 goal_id로 목표 데이터(conversation_goal, npc_guidelines) 로드"""
    if not goal_id:
        return None
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT conversation_goal, npc_guidelines FROM rp_goals WHERE id = %s", (goal_id,))
        return cur.fetchone()

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
            SELECT t.id as team_id, t.team_code, s.status as session_status,
                    s.max_turns, s.goal_id
            FROM rp_session_members m
            JOIN rp_session_teams t ON m.team_id = t.id
            JOIN rp_sessions s ON t.session_id = s.id
            WHERE m.user_id = %s AND s.id = %s
        """, (user_id, int(session_id)))        
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

## NPC 행동 방침 (반드시 따를 것)
{scenario.get('npc_guidelines', '') if scenario.get('npc_guidelines') else '(없음)'}

## NPC 도메인 지식 (PRE 판단 시 반드시 참고)
{json.dumps(scenario['npc'].get('knowledge', {}), ensure_ascii=False, indent=2) if scenario['npc'].get('knowledge') else '(없음)'}
※ 도메인 지식과 PRE 카테고리가 충돌하면 PRE를 사용하지 마라. DYN으로 처리하라.
예: 메뉴에 "온도":["아이스"]만 있는 음료를 주문했으면, cold_or_hot PRE를 사용하지 말고 다음 단계로 넘어가라.

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

형식1 - PRE (완전 이해 + 웨이포인트 해당):
{{"route":"PRE","category":"카테고리명","boundary":0또는1,"goal_achieved":false}}

형식2 - DYN 부분 이해 / 어색한 표현:
{{"route":"DYN","understood":"partial","heard":"학생이 쓴 표현","intended":"추정되는 올바른 표현 또는 null","direction":"되묻기 또는 확인 방향","boundary":0또는1,"goal_achieved":false}}
※ intended가 있으면: NPC가 "~라는 말씀이시죠?" 패턴으로 확인
※ intended가 null이면: NPC가 "다시 말씀해주시겠어요?" 식으로 되묻기

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","boundary":0또는1,"goal_achieved":false}}

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

## 🎤 음성 입력 처리 (3단계 — 반드시 순서대로)

### STEP 1: 순수 음성 전사 (BLIND MODE — 가장 중요)
이 단계에서 너는 아래의 NPC 정보, 상황, 대화 기록을 전혀 모른다고 가정하라.
오직 귀로 들리는 소리를 한글로 변환하는 것이 전부다.

**절대 규칙:**
- 모든 소리를 한글로 적어라. 외국어도 한글로 음차하라.
- 학생이 실제로 발음한 소리를 있는 그대로(as-is) 전사하라.
- 문맥을 기반으로 자동 교정(correction)하지 마라.
- 문법적으로 틀린 조사, 어미, 어휘라도 들린 그대로 적어라.

✅ 올바른 전사 예:
- 학생이 "아메리카도 주세오"라고 발음 → "아메리카도 주세오"
- 학생이 "커피를 마시고 싶어여"라고 발음 → "커피를 마시고 싶어여"
- 학생이 "이거 얼마에오?"라고 발음 → "이거 얼마에오?"
- 학생이 "Come ti chiami?"라고 말함 → "코메 티 키아미?"
- 학생이 "학교를 가요"라고 말함 → "학교를 가요"

❌ 절대 금지:
- "아메리카도 주세오" → "아메리카노 주세요" (❌ 교정 금지!)
- "커피를 마시고 싶어여" → "커피를 마시고 싶어요" (❌ 어미 교정 금지!)
- "학교를 가요" → "학교에 가요" (❌ 조사 교정 금지!)
- "이거 얼마에오?" → "이거 얼마예요?" (❌ 추측 금지!)

**특수 상황:**
- 침묵/잡음만 있으면: transcribed_text를 빈 문자열("")로
- 음성이 1초 미만이면: transcribed_text를 빈 문자열("")로

### STEP 2: 언어 판별
STEP 1에서 전사한 텍스트를 보고 판별하라:
- 한국어 단어가 1개라도 포함 → 한국어로 판단, STEP 3으로
- 한국어 단어가 전혀 없음 (외국어만) → 형식4 사용
- 빈 문자열 (침묵/잡음) → 형식4 사용

### STEP 3: 분석
이제 아래의 NPC 정보와 상황을 참고하여 PRE/DYN 판단을 수행하라.
단, transcribed_text는 STEP 1의 결과를 절대 수정하지 마라.

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

## NPC 행동 방침 (direction 결정 시 반드시 참고)
{scenario.get('npc_guidelines', '') if scenario.get('npc_guidelines') else '(없음)'}
※ 행동 방침이 있으면, 대화 기록을 보고 NPC가 이미 해당 행동을 했는지 판단한 뒤 direction에 반영하라.

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
  - 한국어이지만 의미를 전혀 파악 불가 → 형식2 DYN (되묻기, intended=null)
  - 한국어이지만 표현이 어색/부정확 → 형식2 DYN (확인, intended="추정 표현")
  - 완전히 이해 가능 → 2단계로
2단계: PRE 웨이포인트에 해당하는가?
  - 해당함 → 형식1 PRE + category
  - 해당하지 않음 → 3단계로
3단계: 형식3 DYN + 감정 분석

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

형식1 - PRE (완전 이해 + 웨이포인트 해당):
{{"route":"PRE","category":"카테고리명","transcribed_text":"STEP1 전사 결과","boundary":0또는1,"goal_achieved":false}}

형식2 - DYN 부분 이해 / 어색한 표현:
{{"route":"DYN","understood":"partial","heard":"들린 부분","intended":"추정되는 올바른 표현 또는 null","direction":"되묻기 또는 확인 방향","transcribed_text":"STEP1 전사 결과","boundary":0또는1,"goal_achieved":false}}
※ intended가 있으면: NPC가 "~라는 말씀이시죠?" 패턴으로 확인할 수 있다
※ intended가 null이면: NPC가 "다시 말씀해주시겠어요?" 식으로 되묻는다

형식3 - DYN 완전 이해:
{{"route":"DYN","understood":true,"main_emotion":"감정","intensity":강도,"sub_emotion":"보조감정또는null","sub_intensity":강도또는null,"audio_tags":"[태그1][태그2]","direction":"반응 방향","transcribed_text":"STEP1 전사 결과","boundary":0또는1,"goal_achieved":false}}

형식4 - 외국어 / 인식 실패 (한국어가 아니거나 소리 없음):
{{"route":"DYN","understood":false,"is_korean":false,"transcribed_text":"STEP1 전사 결과 또는 빈문자열","direction":"외국어라서 못 알아듣겠다는 자연스러운 반응","boundary":1,"goal_achieved":false}}

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
   분석가가 제공한 audio_tags를 대사 텍스트 안에 넣어라.
   예: "[laughing] 아 네, 카푸치노는 원래 따뜻한 거예요. [warmly] 맛있게 드세요!"

2. **1~2문장으로 짧게.** 진짜 대화처럼 짧게 말하라. 길게 설명하지 마라.

3. **어휘 수준은 TOPIK 3급 이하로 제한하라.** 학생은 한국어를 배우는 외국인이다. 쉬운 단어와 기본 문형을 사용하라. 어려운 관용구, 사자성어, 축약어, 신조어는 절대 쓰지 마라.

4. **캐릭터를 유지하라.** {npc['name']}은(는) {npc['age']}세 {npc['job']}이다. 자연스러운 말투를 쓰라.

5. **NPC 도메인 지식을 반드시 확인하고 정확히 따르라.** 메뉴, 가격, 옵션 정보가 도메인 지식에 있으면 반드시 그대로 사용하라. 도메인 지식에 없는 선택지를 만들어내지 마라. 예: 메뉴에 "온도":["아이스"]만 있으면 핫/아이스를 묻지 말고 바로 아이스로 진행하라.

7. **말투 규칙: {scenario.get('speech_style', '비격식 존댓말')}을 사용하라.**
   - "격식 존댓말": 합쇼체(-습니다, -습니까)를 사용. 예: "주문하시겠습니까?", "감사합니다."
   - "비격식 존댓말": 해요체(-아/어요)를 사용. 예: "주문하시겠어요?", "감사해요~"
   - "반말": 해체(-아/어)를 사용. 예: "뭐 마실 거야?", "잠깐만~"
   학생이 어떤 말투를 쓰든, NPC는 위 지정된 말투를 일관되게 유지하라.

8. **부분 이해 시 확인 패턴.** 분석가가 "understood":"partial"이고 "intended" 값이 있으면, NPC는 intended의 내용을 활용하여 "~라는 말씀이시죠?", "~요?", "혹시 ~(이)요?" 패턴으로 자연스럽게 확인하라.
   예: intended가 "아메리카노 주세요"이면 → "아메리카노 주세요...라는 말씀이시죠?"
   예: intended가 "카드로 할게요"이면 → "카드로 하신다는 거죠?"

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
                thinking_level=getattr(types.ThinkingLevel, scenario.get('thinking_level', 'LOW'), types.ThinkingLevel.LOW)
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

    parsed = None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # 마지막 '}' 에서 실패 시, 첫 번째 완전한 JSON 객체를 찾는다
        import re
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # 그래도 실패하면, 중첩 깊이로 첫 번째 완전한 JSON을 추출
        if not parsed:
            depth = 0
            start = clean.index('{')
            for i in range(start, len(clean)):
                if clean[i] == '{': depth += 1
                elif clean[i] == '}': depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(clean[start:i+1])
                        break
                    except json.JSONDecodeError:
                        continue

    if not parsed:
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
                thinking_level=getattr(types.ThinkingLevel, scenario.get('thinking_level', 'LOW'), types.ThinkingLevel.LOW)
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

    parsed = None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # 마지막 '}' 에서 실패 시, 첫 번째 완전한 JSON 객체를 찾는다
        import re
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # 그래도 실패하면, 중첩 깊이로 첫 번째 완전한 JSON을 추출
        if not parsed:
            depth = 0
            start = clean.index('{')
            for i in range(start, len(clean)):
                if clean[i] == '{': depth += 1
                elif clean[i] == '}': depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(clean[start:i+1])
                        break
                    except json.JSONDecodeError:
                        continue

    if not parsed:
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
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            )
        )
    )
    actor_raw = (actor_response.text or "").strip()
    actor_line = actor_raw.strip('"').strip("'")
    actor_latency = int((time.time() - actor_start) * 1000)

    return actor_line, actor_latency

def convert_korean_numbers(text):
    """TTS 전처리: 한국어 맥락의 숫자를 한글로 변환"""
    import re
    
    # 가격 패턴: 숫자+원 → 한글+원
    def price_to_korean(match):
        num = int(match.group(1))
        if num >= 10000:
            man = num // 10000
            remainder = num % 10000
            if remainder >= 1000:
                cheon = remainder // 1000
                rest = remainder % 1000
                if rest > 0:
                    return f"{_sino(man)}만 {_sino(cheon)}천{_sino_hundreds(rest)}원"
                return f"{_sino(man)}만 {_sino(cheon)}천원"
            elif remainder > 0:
                return f"{_sino(man)}만 {_sino_hundreds(remainder)}원"
            return f"{_sino(man)}만원"
        elif num >= 1000:
            cheon = num // 1000
            rest = num % 1000
            if rest > 0:
                return f"{_sino(cheon)}천{_sino_hundreds(rest)}원"
            return f"{_sino(cheon)}천원"
        return f"{num}원"
    
    def _sino(n):
        """1~9 → 한글 (한자어 수)"""
        sino = {1:'일',2:'이',3:'삼',4:'사',5:'오',6:'육',7:'칠',8:'팔',9:'구'}
        return sino.get(n, str(n))
    
    def _sino_hundreds(n):
        """100~999 부분을 한글로"""
        result = ''
        if n >= 100:
            h = n // 100
            result += f"{_sino(h)}백"
            n %= 100
        if n >= 10:
            t = n // 10
            result += f"{_sino(t)}십"
            n %= 10
        if n > 0:
            result += _sino(n)
        return result
    
    # 가격 변환: 숫자+원
    text = re.sub(r'(\d+)원', price_to_korean, text)
    
    return text

def run_tts(text, voice_id=None):
    """TTS 호출 → base64 반환 (숫자→한글 전처리 포함)"""
    tts_start = time.time()
    processed_text = convert_korean_numbers(text)
    tts_bytes = call_elevenlabs_tts(processed_text, voice_id)
    tts_latency = int((time.time() - tts_start) * 1000)

# ============================================================
# PRE 오디오 URL 조회
# ============================================================
def get_pre_audio_url(scenario_id, category, conn, team_id=None):
    """PRE 카테고리의 랜덤 변형 오디오 URL 반환 (사용한 URL 제외)"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 미사용 URL 우선
        if team_id:
            cur.execute("""
                SELECT cloudflare_url, transcript FROM rp_pre_recordings
                WHERE scenario_id = %s AND category = %s AND cloudflare_url IS NOT NULL
                AND cloudflare_url NOT IN (
                    SELECT pre_audio_url FROM rp_conversation_logs
                    WHERE team_id = %s AND scenario_id = %s AND pre_audio_url IS NOT NULL
                )
                ORDER BY RANDOM() LIMIT 1
            """, (scenario_id, category, team_id, scenario_id))
            row = cur.fetchone()
            if row:
                return row['cloudflare_url'], row['transcript']

        # 폴백: 전부 소진 시 반복 허용
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

def get_boundary_pre(conn, team_id=None, scenario_id=None):
    """공통 Boundary PRE 풀에서 랜덤 1개 반환 (사용한 URL 제외)"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if team_id and scenario_id:
            cur.execute("""
                SELECT cloudflare_url, transcript FROM rp_pre_recordings
                WHERE category = 'boundary_pre' AND cloudflare_url IS NOT NULL
                AND cloudflare_url NOT IN (
                    SELECT pre_audio_url FROM rp_conversation_logs
                    WHERE team_id = %s AND scenario_id = %s AND pre_audio_url IS NOT NULL
                )
                ORDER BY RANDOM() LIMIT 1
            """, (team_id, scenario_id))
            row = cur.fetchone()
            if row:
                return row['cloudflare_url'], row['transcript']

        # 폴백: 전부 소진 시 반복 허용
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
            if not student_input.strip():
                parsed['direction'] = f"boundary Exit: 학생이 {total_violations}회 계속 한국어가 아닌 말을 했다. 더 이상 대화할 수 없다며 대화를 끝내는 대사를 하라. NPC 성격에 맞게."
            else:
                parsed['direction'] = f"boundary Exit: 학생이 \"{student_input}\"라고 했는데 {total_violations}회 계속 엉뚱한 말을 한다. 대화를 끝내는 대사를 하라. NPC 성격에 맞게."

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
            if not student_input.strip():
                parsed['direction'] = f"boundary DYN: 학생이 {total_violations}회 한국어가 아닌 말을 했다. 한국어로 말해달라고 요청하면서, 원래 대화 목표로 돌아가게 유도하라. 불쾌한 감정으로."
            else:
                parsed['direction'] = f"boundary DYN: 학생이 \"{student_input}\"라고 했는데 상황에 맞지 않는 말이다. 되묻기/저의확인/목표환기 중 상황에 맞게. 불쾌한 감정으로."

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
                scenario_id, "boundary_pre", conn, team_id)

            if not pre_audio_url:
                pre_audio_url, pre_transcript = get_boundary_pre(conn, team_id, scenario_id)

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
            scenario_id, parsed.get("category", ""), conn, team_id)
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
                SELECT ss.scenario_id, ss.order_num, sc.title, sc.npc_name,
                       sc.illustration_url, sc.speech_style, sc.npc_knowledge,
                       sc.situation
                FROM rp_session_scenarios ss                        
                JOIN rp_scenarios sc ON ss.scenario_id = sc.id
                WHERE ss.session_id = %s
                ORDER BY ss.order_num
            """, (session_id,))
            scenarios = cur.fetchall()

            # 세션의 goal에서 objective_it 로드
            cur.execute("""
                SELECT g.objective_it FROM rp_sessions s
                JOIN rp_goals g ON s.goal_id = g.id
                WHERE s.id = %s
            """, (session_id,))
            goal_row = cur.fetchone()
            objective_it = goal_row['objective_it'] if goal_row else ''

            # 모든 시나리오에 objective_it 추가
            for sc in scenarios:
                sc['objective_it'] = objective_it

        # 팀별 랜덤 순서
        import random
        rng = random.Random(player['team_id'])
        rng.shuffle(scenarios)

        # 각 시나리오별 현재 턴 + 완료 여부
        for sc in scenarios:
            sc['current_turn'] = get_current_turn(player['team_id'], sc['scenario_id'], conn)
            # 완료 여부 확인 ([GOAL_ACHIEVED] 또는 [EXIT] 마커 존재 여부)
            with conn.cursor() as cur2:
                cur2.execute("""
                    SELECT 1 FROM rp_conversation_logs
                    WHERE team_id = %s AND scenario_id = %s
                      AND speaker = 'npc'
                      AND message_text IN ('[GOAL_ACHIEVED]', '[EXIT]')
                    LIMIT 1
                """, (player['team_id'], sc['scenario_id']))
                sc['is_completed'] = cur2.fetchone() is not None

        return jsonify({
            "team_id": player['team_id'],
            "team_code": player['team_code'],
            "session_status": player['session_status'],
            "scenarios": scenarios,
            "max_turns": player.get('max_turns', 8)
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
        max_turns = player.get('max_turns', 8)

        # 2. 턴 제한 확인
        current_turn = get_current_turn(team_id, scenario_id, conn)
        if current_turn >= max_turns:
            return jsonify({"error": f"이 시나리오의 턴이 모두 소진되었습니다 ({max_turns}턴)", "turn_limit_reached": True}), 400
        
        # 3. 시나리오 로드
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오를 찾을 수 없습니다"}), 404

        # 3-1. 목표 데이터 병합 (goal의 conversation_goal/npc_guidelines 우선)
        goal_data = load_goal_data(player.get('goal_id'), conn)
        if goal_data:
            if goal_data.get('conversation_goal'):
                scenario['conversation_goal'] = goal_data['conversation_goal']
            if goal_data.get('npc_guidelines'):
                scenario['npc_guidelines'] = goal_data['npc_guidelines']        

        # 4. 대화 기록 로드
        conversation_history = load_conversation_history(team_id, scenario_id, conn)

        # 5. 한글 체크 → 외국어면 분석가 스킵
        import re
        if not re.search('[가-힣]', student_input):
            parsed = {"route": "PRE", "category": "not_understood", "boundary": 1, "goal_achieved": False}
            analyst_latency = 0
            prompt = None
        else:
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
            "turns_remaining": max_turns - new_turn,
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
        max_turns = player.get('max_turns', 8)

        # 2. 턴 제한
        current_turn = get_current_turn(team_id, scenario_id, conn)
        if current_turn >= max_turns:
            return jsonify({"error": f"턴 소진 ({max_turns}턴)", "turn_limit_reached": True}), 400
        
        # 3. 시나리오 + 대화기록 로드
        scenario = load_scenario_from_db(scenario_id, conn)
        if not scenario:
            return jsonify({"error": "시나리오 없음"}), 404

        # 3-1. 목표 데이터 병합
        goal_data = load_goal_data(player.get('goal_id'), conn)
        if goal_data:
            if goal_data.get('conversation_goal'):
                scenario['conversation_goal'] = goal_data['conversation_goal']
            if goal_data.get('npc_guidelines'):
                scenario['npc_guidelines'] = goal_data['npc_guidelines']

        conversation_history = load_conversation_history(team_id, scenario_id, conn)

        # 4. 오디오 읽기 + 분석가 호출
        audio_bytes = audio_file.read()
        parsed, analyst_latency, prompt = run_analyst_audio(
            scenario, conversation_history, audio_bytes, mime_type)

        transcribed_text = parsed.get("transcribed_text", "")
        
        # parse_error 발생 시 raw에서 transcribed_text 복구 시도
        if not transcribed_text and parsed.get("parse_error"):
            import re
            raw = parsed.get("raw", "")
            m = re.search(r'"transcribed_text"\s*:\s*"([^"]*)"', raw)
            if m:
                transcribed_text = m.group(1)

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

            "turns_remaining": max_turns - new_turn,

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

        # max_turns 조회
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur2:
            cur2.execute("""
                SELECT s.max_turns FROM rp_session_teams t
                JOIN rp_sessions s ON t.session_id = s.id
                WHERE t.id = %s
            """, (team_id,))
            sess_row = cur2.fetchone()
            max_turns = sess_row['max_turns'] if sess_row else 8

        return jsonify({
            "logs": logs,
            "current_turn": current_turn,
            "turns_remaining": max_turns - current_turn,
            "max_turns": max_turns
        })    

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()