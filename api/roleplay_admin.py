"""
롤플레이 관리 API (roleplay_admin.py)
- 시나리오 CRUD
- 목표 CRUD
- PRE 녹음 관리
- 교사 인증 필수
"""
import os
import json
import pathlib
import traceback
from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect

import psycopg2
import psycopg2.extras

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

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"🚨 DB 연결 오류: {e}")
        return None

def teacher_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('is_teacher'):
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# ============================================================
# 페이지 라우트
# ============================================================
@app.route('/roleplay-admin')
def roleplay_admin_page():
    if not session.get('is_teacher'):
        return redirect('/teacher-login')
    return render_template('roleplay/roleplay_admin.html')

@app.route('/roleplay-session')
def roleplay_session_page():
    if not session.get('is_teacher'):
        return redirect('/teacher-login')
    return render_template('roleplay/roleplay_session.html')

# ============================================================
# 시나리오 API
# ============================================================

@app.route('/api/rp-admin/scenarios', methods=['GET'])
@teacher_required
def get_scenarios():
    """시나리오 목록 조회"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM rp_scenarios ORDER BY id DESC")
            scenarios = cur.fetchall()
            # JSON 필드 직렬화
            for s in scenarios:
                if s.get('npc_knowledge') and isinstance(s['npc_knowledge'], str):
                    try: s['npc_knowledge'] = json.loads(s['npc_knowledge'])
                    except: pass
            return jsonify({"scenarios": scenarios})
    except Exception as e:
        print(f"🚨 시나리오 조회 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/scenarios', methods=['POST'])
@teacher_required
def create_scenario():
    """시나리오 생성"""
    data = request.get_json()
    if not data or not data.get('title'):
        return jsonify({"error": "제목은 필수입니다"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            # npc_knowledge를 JSON 문자열로 변환
            npc_knowledge = data.get('npc_knowledge')
            if npc_knowledge and isinstance(npc_knowledge, dict):
                npc_knowledge = json.dumps(npc_knowledge, ensure_ascii=False)
            elif npc_knowledge and isinstance(npc_knowledge, str):
                # 유효한 JSON인지 확인
                try:
                    json.loads(npc_knowledge)
                except:
                    npc_knowledge = None

            cur.execute("""
                INSERT INTO rp_scenarios (
                    title, situation,
                    illustration_url, speech_style,
                    npc_name, npc_age, npc_job,
                    npc_personality, npc_current_state, npc_knowledge,
                    npc_voice_id, temperature, thinking_level
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                data.get('title'),
                data.get('situation'),
                data.get('illustration_url'),
                data.get('speech_style', '비격식 존댓말'),
                data.get('npc_name'),
                data.get('npc_age'),
                data.get('npc_job'),
                data.get('npc_personality'),
                data.get('npc_current_state'),
                npc_knowledge,
                data.get('npc_voice_id'),
                data.get('temperature', 0.3),
                data.get('thinking_level', 'LOW')
            ))

            
            new_id = cur.fetchone()[0]
            conn.commit()
            return jsonify({"success": True, "id": new_id})
    except Exception as e:
        conn.rollback()
        print(f"🚨 시나리오 생성 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/scenarios/<int:scenario_id>', methods=['DELETE'])
@teacher_required
def delete_scenario(scenario_id):
    """시나리오 삭제 (CASCADE로 연결된 PRE도 삭제)"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rp_scenarios WHERE id = %s", (scenario_id,))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/scenarios/<int:scenario_id>', methods=['PUT'])
@teacher_required
def update_scenario(scenario_id):
    """시나리오 수정"""
    data = request.get_json()
    if not data: return jsonify({"error": "데이터 없음"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            npc_knowledge = data.get('npc_knowledge')
            if npc_knowledge and isinstance(npc_knowledge, dict):
                npc_knowledge = json.dumps(npc_knowledge, ensure_ascii=False)
            elif npc_knowledge and isinstance(npc_knowledge, str):
                try: json.loads(npc_knowledge)
                except: npc_knowledge = None

            cur.execute("""
                UPDATE rp_scenarios SET
                    title=%s, situation=%s,
                    illustration_url=%s, speech_style=%s,
                    npc_name=%s, npc_age=%s, npc_job=%s,
                    npc_personality=%s, npc_current_state=%s, npc_knowledge=%s,
                    npc_voice_id=%s, temperature=%s, thinking_level=%s
                WHERE id=%s
            """, (
                data.get('title'), data.get('situation'),
                data.get('illustration_url'), data.get('speech_style', '비격식 존댓말'),
                data.get('npc_name'), data.get('npc_age'), data.get('npc_job'),
                data.get('npc_personality'), data.get('npc_current_state'), npc_knowledge,
                data.get('npc_voice_id'), data.get('temperature', 0.3), data.get('thinking_level', 'LOW'),
                scenario_id
            ))


            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        print(f"🚨 시나리오 수정 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# 목표 API
# ============================================================

@app.route('/api/rp-admin/goals', methods=['GET'])
@teacher_required
def get_goals():
    """목표 목록 조회"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM rp_goals ORDER BY id DESC")
            goals = cur.fetchall()
            return jsonify({"goals": goals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/goals', methods=['POST'])
@teacher_required
def create_goal():
    """목표 생성"""
    data = request.get_json()
    if not data or not data.get('title'):
        return jsonify({"error": "제목은 필수입니다"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            
            cur.execute("""
                INSERT INTO rp_goals (title, target_expression, target_grammar, target_vocabulary, class_name, conversation_goal, npc_guidelines, objective_it)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                data.get('title'),
                data.get('target_expression'),
                data.get('target_grammar'),
                data.get('target_vocabulary'),
                data.get('class_name'),
                data.get('conversation_goal'),
                data.get('npc_guidelines'),
                data.get('objective_it')
            ))

            new_id = cur.fetchone()[0]
            conn.commit()
            return jsonify({"success": True, "id": new_id})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/goals/<int:goal_id>', methods=['DELETE'])
@teacher_required
def delete_goal(goal_id):
    """목표 삭제"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rp_goals WHERE id = %s", (goal_id,))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/goals/<int:goal_id>', methods=['PUT'])
@teacher_required
def update_goal(goal_id):
    """목표 수정"""
    data = request.get_json()
    if not data: return jsonify({"error": "데이터 없음"}), 400
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE rp_goals SET title=%s, target_expression=%s, target_grammar=%s, target_vocabulary=%s, class_name=%s,
                    conversation_goal=%s, npc_guidelines=%s, objective_it=%s
                WHERE id=%s
            """, (
                data.get('title'), data.get('target_expression'),
                data.get('target_grammar'), data.get('target_vocabulary'),
                data.get('class_name'),
                data.get('conversation_goal'), data.get('npc_guidelines'),
                data.get('objective_it'),
                goal_id
            ))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# PRE 녹음 API
# ============================================================

@app.route('/api/rp-admin/pre-recordings/<int:scenario_id>', methods=['GET'])
@teacher_required
def get_pre_recordings(scenario_id):
    """특정 시나리오의 PRE 목록 조회"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM rp_pre_recordings 
                WHERE scenario_id = %s 
                ORDER BY category, variant
            """, (scenario_id,))
            recordings = cur.fetchall()
            return jsonify({"recordings": recordings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/pre-recordings', methods=['POST'])
@teacher_required
def create_pre_recording():
    """PRE 녹음 등록"""
    data = request.get_json()
    required = ['scenario_id', 'category', 'variant', 'transcript']
    if not all(data.get(k) for k in required):
        return jsonify({"error": f"필수 필드: {', '.join(required)}"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rp_pre_recordings (scenario_id, category, variant, guide_text, transcript, cloudflare_url)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                data['scenario_id'],
                data['category'],
                data['variant'],
                data.get('guide_text'),
                data['transcript'],
                data.get('cloudflare_url')
            ))
            new_id = cur.fetchone()[0]
            conn.commit()
            return jsonify({"success": True, "id": new_id})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "이미 존재하는 조합입니다 (scenario_id + category + variant)"}), 409
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/pre-recordings/<int:recording_id>', methods=['DELETE'])
@teacher_required
def delete_pre_recording(recording_id):
    """PRE 녹음 삭제"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rp_pre_recordings WHERE id = %s", (recording_id,))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/pre-recordings/<int:recording_id>', methods=['PUT'])
@teacher_required
def update_pre_recording(recording_id):
    """PRE 녹음 수정"""
    data = request.get_json()
    if not data: return jsonify({"error": "데이터 없음"}), 400
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE rp_pre_recordings 
                SET category=%s, variant=%s, guide_text=%s, transcript=%s, cloudflare_url=%s
                WHERE id=%s
            """, (
                data.get('category'), data.get('variant'),
                data.get('guide_text'), data.get('transcript'),
                data.get('cloudflare_url'), recording_id
            ))
            conn.commit()
            return jsonify({"success": True})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "이미 존재하는 조합입니다 (scenario_id + category + variant)"}), 409
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# 세션 API
# ============================================================

@app.route('/api/rp-admin/sessions', methods=['GET'])
@teacher_required
def get_sessions():
    """세션 목록 조회 (목표/시나리오 정보 포함)"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT s.*, g.title as goal_title
                FROM rp_sessions s
                LEFT JOIN rp_goals g ON s.goal_id = g.id
                ORDER BY s.id DESC
            """)
            sessions = cur.fetchall()

            for sess in sessions:
                cur.execute("""
                    SELECT sc.title, sc.id as scenario_id
                    FROM rp_session_scenarios ss
                    JOIN rp_scenarios sc ON ss.scenario_id = sc.id
                    WHERE ss.session_id = %s ORDER BY ss.order_num
                """, (sess['id'],))
                sess['scenarios'] = cur.fetchall()

                cur.execute("""
                    SELECT t.id, t.team_code, COUNT(m.id) as member_count
                    FROM rp_session_teams t
                    LEFT JOIN rp_session_members m ON m.team_id = t.id
                    WHERE t.session_id = %s
                    GROUP BY t.id, t.team_code
                    ORDER BY t.team_code
                """, (sess['id'],))
                sess['teams'] = cur.fetchall()

            return jsonify({"sessions": sessions})
    except Exception as e:
        print(f"🚨 세션 조회 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/sessions', methods=['POST'])
@teacher_required
def create_session():
    """세션 생성 + 팀 자동 생성"""
    data = request.get_json()
    class_name = data.get('class_name')
    goal_id = data.get('goal_id')
    scenario_ids = data.get('scenario_ids', [])
    team_count = data.get('team_count', 1)
    max_turns = data.get('max_turns', 8)
    if not class_name or not scenario_ids:
        return jsonify({"error": "반, 시나리오는 필수입니다"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO rp_sessions (class_name, goal_id, team_count, max_turns, status)
                VALUES (%s, %s, %s, %s, 'waiting') RETURNING id
            """, (class_name, goal_id if goal_id else None, team_count, max_turns))            
            session_id = cur.fetchone()[0]

            for idx, sc_id in enumerate(scenario_ids):
                cur.execute("""
                    INSERT INTO rp_session_scenarios (session_id, scenario_id, order_num)
                    VALUES (%s, %s, %s)
                """, (session_id, sc_id, idx + 1))

            for i in range(1, team_count + 1):
                team_code = f"A{i}"
                cur.execute("""
                    INSERT INTO rp_session_teams (session_id, team_code)
                    VALUES (%s, %s)
                """, (session_id, team_code))

            conn.commit()
            return jsonify({"success": True, "id": session_id, "team_count": team_count})
    except Exception as e:
        conn.rollback()
        print(f"🚨 세션 생성 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/sessions/<int:session_id>/start', methods=['PUT'])
@teacher_required
def start_session(session_id):
    """세션 시작 (waiting → active)"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE rp_sessions SET status='active', started_at=CURRENT_TIMESTAMP
                WHERE id=%s AND status='waiting'
            """, (session_id,))
            if cur.rowcount == 0:
                return jsonify({"error": "시작할 수 없는 세션입니다 (이미 시작됨 또는 종료됨)"}), 400
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/sessions/<int:session_id>/complete', methods=['PUT'])
@teacher_required
def complete_session(session_id):
    """세션 종료 (active → completed)"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE rp_sessions SET status='completed', completed_at=CURRENT_TIMESTAMP
                WHERE id=%s AND status IN ('waiting','active')
            """, (session_id,))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-admin/sessions/<int:session_id>', methods=['DELETE'])
@teacher_required
def delete_session(session_id):
    """세션 삭제 (CASCADE로 팀/멤버도 삭제)"""
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rp_sessions WHERE id = %s", (session_id,))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# 교사용 — 팀 대화 관찰 API
# ============================================================

@app.route('/api/rp-admin/team-history', methods=['GET'])
@teacher_required
def teacher_team_history():
    """교사: 특정 팀의 대화 기록 조회 (관찰용)"""
    team_id = request.args.get('team_id')
    scenario_id = request.args.get('scenario_id')

    if not all([team_id, scenario_id]):
        return jsonify({"error": "team_id, scenario_id 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT turn_number, speaker, message_text, actor_line,
                       analyst_json, created_at::text as created_at,
                       player_user_id
                FROM rp_conversation_logs
                WHERE team_id = %s AND scenario_id = %s
                ORDER BY turn_number ASC, id ASC
            """, (int(team_id), int(scenario_id)))
            logs = cur.fetchall()

            # 현재 턴
            cur.execute("""
                SELECT COUNT(*) FROM rp_conversation_logs
                WHERE team_id = %s AND scenario_id = %s AND speaker = 'player'
            """, (int(team_id), int(scenario_id)))
            current_turn = cur.fetchone()['count']

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur2:
            cur2.execute("""
                SELECT s.max_turns FROM rp_session_teams t
                JOIN rp_sessions s ON t.session_id = s.id
                WHERE t.id = %s
            """, (int(team_id),))
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


@app.route('/api/rp-admin/team-scenarios', methods=['GET'])
@teacher_required
def teacher_team_scenarios():
    """교사: 특정 세션의 팀+시나리오 목록"""
    session_id = request.args.get('session_id')
    team_id = request.args.get('team_id')

    if not all([session_id, team_id]):
        return jsonify({"error": "session_id, team_id 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 시나리오 목록
            cur.execute("""
                SELECT ss.scenario_id, sc.title, sc.npc_name
                FROM rp_session_scenarios ss
                JOIN rp_scenarios sc ON ss.scenario_id = sc.id
                WHERE ss.session_id = %s
                ORDER BY ss.order_num
            """, (int(session_id),))
            scenarios = cur.fetchall()

            # ★ 학생 측과 동일한 셔플 적용
            import random
            rng = random.Random(int(team_id))
            rng.shuffle(scenarios)
            
            # 팀 멤버
            cur.execute("""
                SELECT m.user_id, u.full_name
                FROM rp_session_members m
                LEFT JOIN users u ON m.user_id = u.id
                WHERE m.team_id = %s
            """, (int(team_id),))
            members = cur.fetchall()

        return jsonify({
            "scenarios": scenarios,
            "members": members
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# 학생용 — 로비 페이지 & API
# ============================================================

def student_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "로그인 필요"}), 401
        return f(*args, **kwargs)
    return wrapper

@app.route('/roleplay-lobby')
def roleplay_lobby_page():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('roleplay/roleplay_lobby.html')

@app.route('/api/rp-student/sessions', methods=['GET'])
@student_required
def student_get_sessions():
    """학생: 자기 반의 활성 세션 목록 조회"""
    class_name = request.args.get('class_name')
    if not class_name:
        return jsonify({"error": "class_name 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # waiting 또는 active 세션만
            cur.execute("""
                SELECT s.id, s.class_name, s.status, s.goal_id, s.team_count,
                       s.created_at, g.title as goal_title
                FROM rp_sessions s
                LEFT JOIN rp_goals g ON s.goal_id = g.id
                WHERE s.class_name = %s AND s.status IN ('waiting', 'active')
                ORDER BY s.created_at DESC
            """, (class_name,))
            sessions = cur.fetchall()

            for sess in sessions:
                # 시나리오 목록
                cur.execute("""
                    SELECT sc.title FROM rp_session_scenarios ss
                    JOIN rp_scenarios sc ON ss.scenario_id = sc.id
                    WHERE ss.session_id = %s ORDER BY ss.order_num
                """, (sess['id'],))
                sess['scenarios'] = [r['title'] for r in cur.fetchall()]

                # 팀 목록 + 인원수
                cur.execute("""
                    SELECT t.id, t.team_code, COUNT(m.id) as member_count
                    FROM rp_session_teams t
                    LEFT JOIN rp_session_members m ON m.team_id = t.id
                    WHERE t.session_id = %s
                    GROUP BY t.id, t.team_code
                    ORDER BY t.team_code
                """, (sess['id'],))
                sess['teams'] = cur.fetchall()

            return jsonify({"sessions": sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-student/join-team', methods=['POST'])
@student_required
def student_join_team():
    """학생: 팀에 합류"""
    data = request.get_json()
    team_id = data.get('team_id')
    user_id = session.get('user_id')

    if not team_id:
        return jsonify({"error": "team_id 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 팀이 속한 세션이 유효한지 확인
            cur.execute("""
                SELECT t.id, t.session_id, s.status
                FROM rp_session_teams t
                JOIN rp_sessions s ON t.session_id = s.id
                WHERE t.id = %s
            """, (team_id,))
            team_info = cur.fetchone()
            if not team_info:
                return jsonify({"error": "존재하지 않는 팀"}), 404
            if team_info['status'] == 'completed':
                return jsonify({"error": "이미 종료된 세션입니다"}), 400

            # 이미 이 세션의 다른 팀에 들어있는지 확인
            cur.execute("""
                SELECT m.id, t.team_code FROM rp_session_members m
                JOIN rp_session_teams t ON m.team_id = t.id
                WHERE t.session_id = %s AND m.user_id = %s
            """, (team_info['session_id'], user_id))
            existing = cur.fetchone()
            if existing:
                return jsonify({"error": f"이미 팀 {existing['team_code']}에 합류했습니다"}), 409

            # 팀 인원 제한 (5명)
            cur.execute("SELECT COUNT(*) as cnt FROM rp_session_members WHERE team_id = %s", (team_id,))
            count = cur.fetchone()['cnt']
            if count >= 5:
                return jsonify({"error": "팀이 가득 찼습니다 (최대 5명)"}), 400

            # 합류
            cur.execute("""
                INSERT INTO rp_session_members (team_id, user_id)
                VALUES (%s, %s)
            """, (team_id, user_id))
            conn.commit()
            return jsonify({"success": True, "session_id": team_info['session_id']})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "이미 이 팀에 합류했습니다"}), 409
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-student/my-status', methods=['GET'])
@student_required
def student_my_status():
    """학생: 특정 세션에서 내 팀 상태 확인 (폴링용)"""
    session_id = request.args.get('session_id')
    user_id = session.get('user_id')
    if not session_id:
        return jsonify({"error": "session_id 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 세션 상태
            cur.execute("SELECT status FROM rp_sessions WHERE id = %s", (session_id,))
            sess = cur.fetchone()
            if not sess:
                return jsonify({"error": "세션 없음"}), 404

            # 내 팀 정보
            cur.execute("""
                SELECT t.team_code, t.id as team_id
                FROM rp_session_members m
                JOIN rp_session_teams t ON m.team_id = t.id
                WHERE t.session_id = %s AND m.user_id = %s
            """, (session_id, user_id))
            my_team = cur.fetchone()

            # 내 팀 멤버 목록
            members = []
            if my_team:
                cur.execute("""
                    SELECT u.full_name FROM rp_session_members m
                    JOIN users u ON m.user_id = u.id
                    WHERE m.team_id = %s
                """, (my_team['team_id'],))
                members = [r['full_name'] for r in cur.fetchall()]

            return jsonify({
                "session_status": sess['status'],
                "my_team": my_team,
                "members": members
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/rp-student/leave-team', methods=['POST'])
@student_required
def student_leave_team():
    """학생: 팀 퇴장"""
    data = request.get_json()
    session_id = data.get('session_id')
    user_id = session.get('user_id')

    if not session_id:
        return jsonify({"error": "session_id 필수"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM rp_session_members
                WHERE user_id = %s AND team_id IN (
                    SELECT id FROM rp_session_teams WHERE session_id = %s
                )
            """, (user_id, session_id))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()