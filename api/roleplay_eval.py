"""
롤플레이 평가 엔진 (roleplay_eval.py)
- 자동 채점 (Gemini 호출)
- 교사 대시보드 조회
- 학생 대시보드 조회
"""
import os
import json
import pathlib
import traceback
import time
from functools import wraps
from flask import Flask, jsonify, request, session, redirect

import psycopg2
import psycopg2.extras

from google import genai
from google.genai import types

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
        print("✅ [roleplay_eval.py] Gemini 클라이언트 로드 완료")
    except Exception as e:
        print(f"🚨 [roleplay_eval.py] Gemini 클라이언트 실패: {e}")

# ============================================================
# 공통 유틸
# ============================================================
def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"🚨 DB 연결 오류: {e}")
        return None

def teacher_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('is_teacher'):
            return f(*args, **kwargs)
        return redirect('/teacher-login')
    return wrapper

def student_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "로그인 필요"}), 401
        return f(*args, **kwargs)
    return wrapper

def extract_first_json_block(text):
    if not text:
        return None
    t = text.replace("```json", "```").strip()
    if "```" in t:
        parts = t.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                return chunk
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return None

def get_rating_details(score):
    try:
        score = float(score)
    except (ValueError, TypeError):
        score = 0.0
    if score >= 8.5: return {"category": "Eccellente", "color": "#00cc9f"}
    if score >= 7.0: return {"category": "Buono", "color": "#00cc29"}
    if score >= 5.5: return {"category": "Sufficiente", "color": "#cccc00"}
    if score >= 4.0: return {"category": "Da migliorare", "color": "#cc6400"}
    return {"category": "Riprova", "color": "#cc0000"}


# ============================================================
# 평가 프롬프트
# ============================================================
ROLEPLAY_EVALUATION_PROMPT = """
당신은 한국어 교육 전문가입니다. 이탈리아 대학생의 한국어 롤플레이 대화를 평가합니다.

## 시나리오 정보
- 제목: {scenario_title}
- 상황: {situation}
- 대화 목표: {conversation_goal}
- NPC: {npc_name} ({npc_job})

## 전체 대화 기록
{conversation_log}

## 평가 대상
- 팀 전체 대화를 평가하시오.
- 개별 학생이 아닌 팀의 협력적 대화 수행을 기준으로 채점하라.

## 평가 기준 (총 10.0점)

### 1. 목표 달성 기여도 (3.0점)
- 대화 목표를 향해 적절한 발화를 했는가?
- 불필요한 이탈 없이 목적에 맞게 진행했는가?
- 목표 달성에 결정적 기여를 했는가?

### 2. 어휘/표현 적절성 (3.0점)
- 상황에 맞는 한국어 어휘를 사용했는가?
- 존댓말/반말 사용이 상황에 적합한가?
- 다양한 표현을 시도했는가?

### 3. 문법 정확성 (2.0점)
- 문장 구조가 올바른가?
- 조사, 어미 사용이 정확한가?

### 4. 대화 자연스러움 (2.0점)
- NPC 응답에 적절히 반응했는가?
- 대화 흐름이 자연스러운가?
- 맥락에 맞지 않는 발화가 있었는가?

## 출력 형식 (반드시 JSON만 출력, 마크다운 금지)
{{
    "score": 7.5,
    "goal_contribution": {{
        "score": 2.5,
        "detail": "주문 목표를 적절히 수행했으나 Turn 3에서 불필요한 이탈이 있었음"
    }},
    "vocabulary": {{
        "score": 2.0,
        "detail": "기본 주문 어휘 사용. '아이스 아메리카노', '카드' 등. 다양성 부족",
        "used_expressions": ["아이스 아메리카노 주세요", "카드로 할게요"],
        "missed_opportunities": ["사이즈 관련 표현 미사용", "포장/매장 표현 미사용"]
    }},
    "grammar": {{
        "score": 1.5,
        "detail": "기본 문형은 정확. 조사 '을/를' 누락 1건",
        "errors": ["아메리카노 주세요 → 아메리카노를 주세요"]
    }},
    "naturalness": {{
        "score": 1.5,
        "detail": "대체로 자연스러운 흐름. Turn 5에서 NPC 질문에 엉뚱한 답변"
    }},
    "summary_for_teacher": "전반적으로 기본적인 주문 수행 가능. 어휘 다양성 확대 필요. 조사 정확성 연습 권장.",
    "boundary_violations": 1
}}
"""


# ============================================================
# API 1: 자동 채점 (프론트엔드에서 goal_achieved 시 호출)
# ============================================================
@app.route('/api/rp/evaluate', methods=['POST'])
def evaluate_roleplay():
    """롤플레이 자동 채점"""
    if not gemini_client:
        return jsonify({"error": "Gemini 미설정"}), 500

    team_id = request.args.get('team_id')
    scenario_id = request.args.get('scenario_id')

    if not all([team_id, scenario_id]):
        return jsonify({"error": "team_id, scenario_id 필수"}), 400

    team_id = int(team_id)
    scenario_id = int(scenario_id)

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── 1. 중복 채점 방지 ──
            cur.execute("""
                SELECT id FROM rp_evaluations
                WHERE team_id = %s AND scenario_id = %s LIMIT 1
            """, (team_id, scenario_id))
            if cur.fetchone():
                return jsonify({"success": True, "message": "이미 채점됨"})

            # ── 2. 팀 정보 ──
            cur.execute("""
                SELECT t.team_code, t.session_id, s.class_name
                FROM rp_session_teams t
                JOIN rp_sessions s ON t.session_id = s.id
                WHERE t.id = %s
            """, (team_id,))
            team_info = cur.fetchone()
            if not team_info:
                return jsonify({"error": "팀 정보 없음"}), 404

            # ── 3. 팀 멤버 ──
            cur.execute("""
                SELECT m.user_id, u.full_name
                FROM rp_session_members m
                LEFT JOIN users u ON m.user_id = u.id
                WHERE m.team_id = %s
            """, (team_id,))
            members = cur.fetchall()
            member_names = ', '.join(
                m['full_name'] or str(m['user_id']) for m in members
            )

            # ── 4. 시나리오 정보 ──
            cur.execute("""
                SELECT title, situation, conversation_goal, npc_name, npc_job
                FROM rp_scenarios WHERE id = %s
            """, (scenario_id,))
            scenario = cur.fetchone()
            if not scenario:
                return jsonify({"error": "시나리오 없음"}), 404

            # ── 5. 대화 기록 ──
            cur.execute("""
                SELECT turn_number, speaker, message_text, actor_line
                FROM rp_conversation_logs
                WHERE team_id = %s AND scenario_id = %s
                ORDER BY turn_number ASC, id ASC
            """, (team_id, scenario_id))
            logs = cur.fetchall()

            conversation_log = ""
            for log in logs:
                if log['speaker'] == 'player':
                    text = log['message_text'] or '(음성)'
                    conversation_log += f"[Turn {log['turn_number']}] 학생: {text}\n"
                elif log['speaker'] == 'npc':
                    text = log['actor_line'] or log['message_text'] or ''
                    if text not in ('[EXIT]', '[GOAL_ACHIEVED]', '[BOUNDARY_PRE]'):
                        conversation_log += f"[Turn {log['turn_number']}] NPC: {text}\n"

            # ── 6. Gemini 호출 ──
            prompt = ROLEPLAY_EVALUATION_PROMPT.format(
                scenario_title=scenario['title'],
                situation=scenario['situation'],
                conversation_goal=scenario['conversation_goal'],
                npc_name=scenario['npc_name'],
                npc_job=scenario['npc_job'] or '',
                conversation_log=conversation_log
            )

            eval_start = time.time()
            response = gemini_client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )
            eval_latency = int((time.time() - eval_start) * 1000)

            raw_text = response.text
            json_str = extract_first_json_block(raw_text)
            if not json_str:
                print(f"🚨 평가 JSON 파싱 실패: {raw_text}")
                return jsonify({"error": "평가 파싱 실패"}), 500

            eval_result = json.loads(json_str)
            score = round(float(eval_result.get('score', 0)), 1)

            # ── 7. 팀원 전원에게 동일 점수 INSERT ──
            for member in members:
                cur.execute("""
                    INSERT INTO rp_evaluations
                    (student_id, scenario_id, session_id, team_id,
                     team_code, class_name, scenario_title, team_members,
                     score, feedback_json, conversation_log)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (student_id, team_id, scenario_id) DO NOTHING
                """, (
                    member['user_id'], scenario_id,
                    team_info['session_id'], team_id,
                    team_info['team_code'], team_info['class_name'],
                    scenario['title'], member_names,
                    score,
                    json.dumps(eval_result, ensure_ascii=False),
                    conversation_log
                ))

            conn.commit()
            print(f"✅ 평가 완료: team {team_id}, scenario {scenario_id}, "
                  f"score {score}, {len(members)}명, {eval_latency}ms")
            return jsonify({"success": True, "score": score})

    except Exception as e:
        conn.rollback()
        print(f"🚨 평가 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API 2: 교사 — 평가 결과 조회
# ============================================================
@app.route('/api/rp-admin/evaluations', methods=['GET'])
@teacher_required
def get_evaluations():
    """교사: 롤플레이 평가 결과 목록"""
    class_name = request.args.get('class_name', 'all')

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 팀 단위로 그룹핑 (같은 team_id+scenario_id는 같은 점수)
            if class_name == 'all':
                cur.execute("""
                    SELECT DISTINCT ON (team_id, scenario_id)
                        id, session_id, team_id, team_code, class_name,
                        scenario_id, scenario_title, team_members,
                        score, feedback_json, created_at
                    FROM rp_evaluations
                    ORDER BY team_id, scenario_id, id
                """)
            else:
                cur.execute("""
                    SELECT DISTINCT ON (team_id, scenario_id)
                        id, session_id, team_id, team_code, class_name,
                        scenario_id, scenario_title, team_members,
                        score, feedback_json, created_at
                    FROM rp_evaluations
                    WHERE class_name = %s
                    ORDER BY team_id, scenario_id, id
                """, (class_name,))

            evals = cur.fetchall()
            for e in evals:
                e['created_at'] = e['created_at'].isoformat() if e.get('created_at') else None

            return jsonify({"evaluations": evals})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API 3: 학생 — 평가 통계 (Overview용)
# ============================================================
@app.route('/api/rp-student/eval-stats', methods=['GET'])
@student_required
def student_eval_stats():
    """학생: 롤플레이 평균 점수 + 횟수"""
    user_id = session.get('user_id')

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT AVG(score) as avg, COUNT(*) as count
                FROM rp_evaluations
                WHERE student_id = %s
            """, (user_id,))
            result = cur.fetchone()

            avg = round(float(result['avg']), 1) if result['avg'] else 0.0
            count = result['count'] or 0
            color = get_rating_details(avg)['color']

            return jsonify({
                "avg": avg,
                "count": count,
                "color": color
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# API 4: 학생 — 평가 기록 (Cronologia용)
# ============================================================
@app.route('/api/rp-student/eval-history', methods=['GET'])
@student_required
def student_eval_history():
    """학생: 롤플레이 채점 기록 목록 (점수+시나리오+날짜만)"""
    user_id = session.get('user_id')

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            cur.execute("""
                SELECT id, scenario_title, team_code, team_members,
                       conversation_log, created_at
                FROM rp_evaluations
                WHERE student_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            evals = cur.fetchall()

            for e in evals:
                e['created_at'] = e['created_at'].strftime('%Y-%m-%d %H:%M') if e.get('created_at') else ''

            return jsonify({"evaluations": evals})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()