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
                if s.get('boundary_strategies') and isinstance(s['boundary_strategies'], str):
                    try: s['boundary_strategies'] = json.loads(s['boundary_strategies'])
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

            # boundary_strategies를 JSON 문자열로 변환
            boundary_strategies = data.get('boundary_strategies', '["되묻기","저의확인","목표환기"]')
            if isinstance(boundary_strategies, list):
                boundary_strategies = json.dumps(boundary_strategies, ensure_ascii=False)

            cur.execute("""
                INSERT INTO rp_scenarios (
                    title, situation, conversation_goal,
                    boundary_tolerance, boundary_strategies,
                    illustration_url,
                    npc_name, npc_age, npc_job,
                    npc_personality, npc_current_state, npc_knowledge,
                    npc_voice_id, temperature, thinking_level
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                data.get('title'),
                data.get('situation'),
                data.get('conversation_goal'),
                data.get('boundary_tolerance', 'low'),
                boundary_strategies,
                data.get('illustration_url'),
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

            boundary_strategies = data.get('boundary_strategies', '["되묻기","저의확인","목표환기"]')
            if isinstance(boundary_strategies, list):
                boundary_strategies = json.dumps(boundary_strategies, ensure_ascii=False)

            cur.execute("""
                UPDATE rp_scenarios SET
                    title=%s, situation=%s, conversation_goal=%s,
                    boundary_tolerance=%s, boundary_strategies=%s,
                    illustration_url=%s,
                    npc_name=%s, npc_age=%s, npc_job=%s,
                    npc_personality=%s, npc_current_state=%s, npc_knowledge=%s,
                    npc_voice_id=%s, temperature=%s, thinking_level=%s
                WHERE id=%s
            """, (
                data.get('title'), data.get('situation'), data.get('conversation_goal'),
                data.get('boundary_tolerance', 'low'), boundary_strategies,
                data.get('illustration_url'),
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
                INSERT INTO rp_goals (title, target_expression, target_grammar, target_vocabulary, class_name)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (
                data.get('title'),
                data.get('target_expression'),
                data.get('target_grammar'),
                data.get('target_vocabulary'),
                data.get('class_name')
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
                UPDATE rp_goals SET title=%s, target_expression=%s, target_grammar=%s, target_vocabulary=%s, class_name=%s
                WHERE id=%s
            """, (
                data.get('title'), data.get('target_expression'),
                data.get('target_grammar'), data.get('target_vocabulary'),
                data.get('class_name'), goal_id
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