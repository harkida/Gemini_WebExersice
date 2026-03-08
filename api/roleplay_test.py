"""
롤플레이 테스트/디버깅 엔진 (roleplay_test.py)
- 프로덕션 roleplay.py의 모든 로직을 미러링
- DB 저장(save_turn) 없음 — 프론트엔드가 상태 관리
- 3단계 체인: 귀(STT) → 분석가 → 연기자 각각 분리 디버깅
"""
import os
import json
import re
import pathlib
import traceback
import time
import base64
from flask import Flask, render_template, jsonify, request

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
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'test-secret-key-change-me')

DATABASE_URL = os.environ.get('DATABASE_URL')

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
        print(f"🚨 [test] Gemini 클라이언트 실패: {e}")

# ============================================================
# ElevenLabs TTS — 프로덕션 동일
# ============================================================
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_MODEL_ID = "eleven_v3"

def call_elevenlabs_tts(text, voice_id=None):
    if not ELEVENLABS_API_KEY:
        print("⚠️ ELEVENLABS_API_KEY 미설정")
        return None
    voice_id = voice_id or "xi3rF0t7dg7uN2M0WUhr"
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
# DB 연결
# ============================================================
def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"🚨 [test] DB 연결 오류: {e}")
        return None

# ============================================================
# DB 로드 헬퍼 — 프로덕션 roleplay.py 그대로
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
# PRE 오디오 URL 조회 — 프로덕션 roleplay.py 그대로
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


# ============================================================
# 프롬프트: 분석가 (텍스트) — 프로덕션 roleplay.py 그대로
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


# ============================================================
# 프롬프트: 연기자 — 프로덕션 roleplay.py 그대로
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
# STT 전용 프롬프트 (BLIND MODE)
# ============================================================
STT_PROMPT = """너는 음성 전사 기계이다. 오디오를 듣고 글자로 바꾸는 것이 유일한 역할이다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🎯 순수 음성 인식 (BLIND MODE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**절대 규칙:**
- 당신은 지금 이 오디오의 "맥락"을 전혀 모른다.
- 어떤 상황인지, 무엇을 말해야 하는지, 정답이 무엇인지 모른다.
- **오직 귀로 들리는 소리를 텍스트로 변환하는 것이 전부다.**

**인식 기준:**
✅ **허용:** 학생이 실제로 발음한 소리 그대로
   - 예: "그 남자 맛있어요" → "그 남자 맛있어요"
   - 예: "저기 문 다주세요" → "저기 문 다주세요" (발음 오류 포함)
   - 예: "아메리카도 주세오" → "아메리카도 주세오"
   - 예: "커피를 마시고 시퍼여" → "커피를 마시고 시퍼여"
   - 예: "이거 얼마에오" → "이거 얼마에오"
   - 예: "너무 보내고 시픈데" → "너무 보내고 시픈데"

❌ **금지:** 문맥 기반 자동 수정
   - "그 남자 맛있어요" → "그 남자 멋있어요" (❌ 절대 안 됨!)
   - "문 다주세요" → "문 닫아 주세요" (❌ 발음 교정 금지!)
   - "아메리카도 주세오" → "아메리카노 주세요" (❌ 금지!)
   - "커피를 마시고 시퍼여" → "커피를 마시고 싶어요" (❌ 금지!)
   - "이거 얼마에오" → "이거 얼마예요" (❌ 금지!)
   - "너무 보내고 시픈데" → "너무 보내고 싶은데" (❌ 금지!)

### ⚠️ 조사 인식 특별 주의사항 (Critical!)

한국어 학습자들은 조사를 매우 자주 틀린다. 절대로 문법적으로 "올바른" 조사로 자동 보정하지 마라!

❌ 절대 금지:
- 학생: "영화를 재미있어요" → "영화가 재미있어요" (❌)
- 학생: "학교를 가요" → "학교에 가요" (❌)
- 학생: "친구가 만났어요" → "친구를 만났어요" (❌)
- 학생: "책이 읽었어요" → "책을 읽었어요" (❌)

✅ 올바른 인식:
- "영화를 재미있어요" → "영화를 재미있어요" (그대로!)
- "학교를 가요" → "학교를 가요" (그대로!)
- "커피를 좋아해요" → "커피를 좋아해요" (그대로!)

**조사 보정 금지 체크리스트:**
- 은/는, 이/가, 을/를 — 학생이 말한 그대로 적었는가?
- 에/에서/로 — 문맥상 틀려도 학생 발음 그대로 적었는가?
- 와/과, 하고 — 보정 없이 들린 그대로 적었는가?

**특수 상황:**
- 침묵/소음만 있으면 빈 문자열 반환
- 외국어가 들리면 한글로 음차: "Come ti chiami?" → "코메 티 키아미?"
- 극도로 불명확하면 들린 부분만 적어라

## ⚠️ 왜 교정하면 안 되는가 (중요)
이 전사 결과는 한국어 학습자의 발음/문법 오류를 평가하는 데 사용된다.
당신이 교정하면, 학생의 실제 오류를 파악할 수 없게 되어 평가가 불가능해진다.
따라서 교정은 평가 시스템을 망가뜨리는 행위이다. 절대 교정하지 마라.

**출력: 반드시 아래 JSON 형식으로만 출력하라.**
{"transcribed_text": "여기에 전사 결과"}"""

# ============================================================
# JSON 파싱 — 프로덕션 roleplay.py 강화 버전
# ============================================================
def parse_gemini_json(raw_text):
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    if '{' in clean:
        clean = clean[clean.index('{'):]
    if '}' in clean:
        clean = clean[:clean.rindex('}') + 1]

    parsed = None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

        if not parsed and '{' in clean:
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

    return parsed


# ============================================================
# TTS 전처리: 한국어 숫자 변환 — 프로덕션 그대로
# ============================================================
def convert_korean_numbers(text):
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
        sino = {1:'일',2:'이',3:'삼',4:'사',5:'오',6:'육',7:'칠',8:'팔',9:'구'}
        return sino.get(n, str(n))

    def _sino_hundreds(n):
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

    text = re.sub(r'(\d+)원', price_to_korean, text)
    return text


# ============================================================
# AI 체인 실행 — 프로덕션 roleplay.py 미러링
# ============================================================
def run_analyst(scenario, conversation_history, student_input):
    prompt = build_analyst_prompt(scenario, conversation_history, student_input)
    start = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level=getattr(types.ThinkingLevel,
                    scenario.get('thinking_level', 'LOW'), types.ThinkingLevel.LOW)
            )
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
            temperature=scenario.get('temperature', 0.5),
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        )
    )
    raw = (response.text or "").strip()
    latency = int((time.time() - start) * 1000)
    return raw.strip('"').strip("'"), latency, prompt


def run_tts(text, voice_id=None):
    start = time.time()
    processed_text = convert_korean_numbers(text)
    tts_bytes = call_elevenlabs_tts(processed_text, voice_id)
    latency = int((time.time() - start) * 1000)
    if tts_bytes:
        return base64.b64encode(tts_bytes).decode('utf-8'), latency
    return None, latency


def run_stt(audio_bytes, mime_type):
    """귀: STT 전용 Gemini 호출"""
    start = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            STT_PROMPT
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    raw = (response.text or "").strip()

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = json.loads(raw.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            parsed = None

    if isinstance(parsed, dict):
        stt_text = parsed.get("transcribed_text", "")
    elif isinstance(parsed, list):
        stt_text = parsed[0] if parsed else ""
    elif parsed is None:
        m = re.search(r"transcribed_text['\"]?\s*:\s*['\"](.+?)['\"]", raw)
        stt_text = m.group(1) if m else raw
    else:
        stt_text = str(parsed)

    # 문자열 보장
    if not isinstance(stt_text, str):
        stt_text = str(stt_text) if stt_text else ""

    # 불필요한 따옴표 제거
    stt_text = re.sub(r'^[\s"\'\u201c\u201d\u2018\u2019]+|[\s"\'\u201c\u201d\u2018\u2019]+$', '', stt_text)

    latency = int((time.time() - start) * 1000)
    return stt_text, latency

# ============================================================
# handle_npc_response — 프로덕션 roleplay.py 미러링
# (DB 저장 제거, PRE 조회는 유지)
# ============================================================
def handle_npc_response(conn, scenario, conversation_history,
                        parsed, student_input, boundary_count):
    """분석가 결과 → boundary 체크 → NPC 응답 결정"""

    actor_line = None
    actor_latency = None
    actor_prompt = None
    tts_audio_b64 = None
    tts_latency = None
    pre_audio_url = None
    pre_transcript = None
    is_exit = False
    npc_name = scenario['npc']['name']
    voice_id = scenario.get('voice_id')
    scenario_id = scenario.get('id')

    # ── Boundary 체크 ──
    boundary = parsed.get('boundary', 0)

    if boundary == 1:
        total_violations = boundary_count + 1  # 현재 턴 포함

        if total_violations >= 4:
            # EXIT
            if not student_input.strip():
                parsed['direction'] = f"boundary Exit: 학생이 {total_violations}회 계속 한국어가 아닌 말을 했다. 더 이상 대화할 수 없다며 대화를 끝내는 대사를 하라. NPC 성격에 맞게."
            else:
                parsed['direction'] = f"boundary Exit: 학생이 \"{student_input}\"라고 했는데 {total_violations}회 계속 엉뚱한 말을 한다. 대화를 끝내는 대사를 하라. NPC 성격에 맞게."
            parsed['main_emotion'] = '불쾌'
            parsed['audio_tags'] = '[sigh][frustrated]'

            actor_line, actor_latency, actor_prompt = run_actor(
                scenario, conversation_history, parsed, student_input)
            if actor_line:
                tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)
            is_exit = True

        elif total_violations >= 3:
            # Boundary DYN
            if not student_input.strip():
                parsed['direction'] = f"boundary DYN: 학생이 {total_violations}회 한국어가 아닌 말을 했다. 한국어로 말해달라고 요청하면서, 원래 대화 목표로 돌아가게 유도하라. 불쾌한 감정으로."
            else:
                parsed['direction'] = f"boundary DYN: 학생이 \"{student_input}\"라고 했는데 상황에 맞지 않는 말이다. 되묻기/저의확인/목표환기 중 상황에 맞게. 불쾌한 감정으로."
            parsed['main_emotion'] = '불쾌'
            parsed['audio_tags'] = '[frustrated][sigh]'

            actor_line, actor_latency, actor_prompt = run_actor(
                scenario, conversation_history, parsed, student_input)
            if actor_line:
                tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)

        else:
            # Boundary PRE
            if scenario_id and conn:
                pre_audio_url, pre_transcript = get_pre_audio_url(
                    scenario_id, "boundary_pre", conn)
            if not pre_audio_url and conn:
                pre_audio_url, pre_transcript = get_boundary_pre(conn)
            if not pre_transcript:
                pre_transcript = "네?"

        return {
            "actor_line": actor_line, "actor_latency": actor_latency,
            "actor_prompt": actor_prompt,
            "tts_audio_b64": tts_audio_b64, "tts_latency": tts_latency,
            "pre_audio_url": pre_audio_url, "pre_transcript": pre_transcript,
            "is_exit": is_exit, "npc_name": npc_name,
            "boundary_action": "exit" if is_exit else ("dyn" if total_violations >= 3 else "pre"),
            "total_violations": total_violations,
            "goal_achieved": False
        }

    # ── 정상 흐름 (boundary=0) ──
    # Aftereffect: 이전 violations가 있으면 톤 영향
    if boundary_count > 0:
        aftereffect = ""
        if boundary_count >= 3:
            aftereffect = "직전에 불쾌한 상황이 있었다. 불쾌하고 사무적인 톤으로. [sigh] [flatly] 활용."
        elif boundary_count >= 1:
            aftereffect = "직전에 당황스러운 상황이 있었다. 약간 머뭇거리는 톤으로. [hesitates] [pause] 활용."
        if aftereffect and parsed.get('direction'):
            parsed['direction'] = aftereffect + " " + parsed['direction']
        elif aftereffect:
            parsed['direction'] = aftereffect

    # ── Goal Achievement 체크 ──
    goal_achieved = parsed.get('goal_achieved', False)
    if goal_achieved is True or goal_achieved == 'true':
        farewell_direction = "대화 목표가 달성되었다. 자연스러운 마무리 인사를 하라. NPC 성격에 맞게 따뜻하게 마무리."
        if parsed.get('direction'):
            parsed['direction'] = farewell_direction + " " + parsed['direction']
        else:
            parsed['direction'] = farewell_direction
        parsed['route'] = 'DYN'
        if not parsed.get('audio_tags'):
            parsed['audio_tags'] = '[warmly]'

    if parsed.get("route") == "PRE":
        if scenario_id and conn:
            pre_audio_url, pre_transcript = get_pre_audio_url(
                scenario_id, parsed.get("category", ""), conn)

    elif parsed.get("route") == "DYN":
        actor_line, actor_latency, actor_prompt = run_actor(
            scenario, conversation_history, parsed, student_input)
        if actor_line:
            tts_audio_b64, tts_latency = run_tts(actor_line, voice_id)

    return {
        "actor_line": actor_line, "actor_latency": actor_latency,
        "actor_prompt": actor_prompt,
        "tts_audio_b64": tts_audio_b64, "tts_latency": tts_latency,
        "pre_audio_url": pre_audio_url, "pre_transcript": pre_transcript,
        "is_exit": False, "npc_name": npc_name,
        "boundary_action": None,
        "total_violations": boundary_count,
        "goal_achieved": parsed.get('goal_achieved', False) in (True, 'true')
    }


# ============================================================
# 페이지 라우트
# ============================================================
@app.route('/roleplay-test')
def roleplay_test_page():
    return render_template('roleplay/roleplay_test.html')


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
    boundary_count = data.get('boundary_count', 0)

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

        # STAGE 1: 귀 (텍스트 패스스루)
        stt_text = student_input
        stt_latency = 0

        # 한글 체크 — 프로덕션 동일
        if not re.search('[가-힣]', stt_text):
            parsed = {"route": "PRE", "category": "not_understood", "boundary": 1, "goal_achieved": False}
            analyst_latency = 0
            analyst_prompt = None
        else:
            # STAGE 2: 분석가
            parsed, analyst_latency, analyst_prompt = run_analyst(
                scenario, conversation_history, stt_text)

        # STAGE 3: handle_npc_response — 프로덕션 미러링
        result = handle_npc_response(
            conn, scenario, conversation_history,
            parsed, stt_text, boundary_count)

        return jsonify({
            "success": True,
            # Stage 1
            "stt_text": stt_text, "stt_latency": stt_latency,
            # Stage 2
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "analyst_prompt": analyst_prompt,
            # Stage 3 (handle_npc_response 결과)
            "actor_line": result["actor_line"],
            "actor_latency": result["actor_latency"],
            "actor_prompt": result.get("actor_prompt"),
            "tts_audio_base64": result["tts_audio_b64"],
            "tts_latency": result["tts_latency"],
            "pre_audio_url": result["pre_audio_url"],
            "pre_transcript": result["pre_transcript"],
            "is_exit": result["is_exit"],
            "npc_name": result["npc_name"],
            "boundary_action": result.get("boundary_action"),
            "total_violations": result.get("total_violations", boundary_count),
            "goal_achieved": result.get("goal_achieved", False),
            # 합산
            "total_latency": stt_latency + analyst_latency + (result["actor_latency"] or 0) + (result["tts_latency"] or 0)
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
    boundary_count = int(request.form.get('boundary_count', 0))

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

        audio_bytes = audio_file.read()

        # ========== STAGE 1: 귀 (STT 전용) ==========
        stt_text, stt_latency = run_stt(audio_bytes, mime_type)

        # STT 실패
        if not stt_text:
            return jsonify({
                "success": True,
                "stt_text": "", "stt_latency": stt_latency,
                "analyst_response": {"route": "DYN", "understood": False,
                                     "direction": "침묵/인식 실패", "boundary": 1},
                "analyst_latency": 0, "analyst_prompt": None,
                "actor_line": None, "actor_latency": None, "actor_prompt": None,
                "tts_audio_base64": None, "tts_latency": None,
                "pre_audio_url": None, "pre_transcript": None,
                "is_exit": False, "npc_name": scenario['npc']['name'],
                "boundary_action": None, "total_violations": boundary_count,
                "goal_achieved": False,
                "total_latency": stt_latency
            })

        # 한글 체크
        if not re.search('[가-힣]', stt_text):
            parsed = {"route": "PRE", "category": "not_understood", "boundary": 1, "goal_achieved": False}
            analyst_latency = 0
            analyst_prompt = None
        else:
            # ========== STAGE 2: 분석가 ==========
            parsed, analyst_latency, analyst_prompt = run_analyst(
                scenario, conversation_history, stt_text)

        # ========== STAGE 3: handle_npc_response ==========
        result = handle_npc_response(
            conn, scenario, conversation_history,
            parsed, stt_text, boundary_count)

        return jsonify({
            "success": True,
            "stt_text": stt_text, "stt_latency": stt_latency,
            "analyst_response": parsed,
            "analyst_latency": analyst_latency,
            "analyst_prompt": analyst_prompt,
            "actor_line": result["actor_line"],
            "actor_latency": result["actor_latency"],
            "actor_prompt": result.get("actor_prompt"),
            "tts_audio_base64": result["tts_audio_b64"],
            "tts_latency": result["tts_latency"],
            "pre_audio_url": result["pre_audio_url"],
            "pre_transcript": result["pre_transcript"],
            "is_exit": result["is_exit"],
            "npc_name": result["npc_name"],
            "boundary_action": result.get("boundary_action"),
            "total_violations": result.get("total_violations", boundary_count),
            "goal_achieved": result.get("goal_achieved", False),
            "total_latency": stt_latency + analyst_latency + (result["actor_latency"] or 0) + (result["tts_latency"] or 0)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"음성 처리 실패: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()