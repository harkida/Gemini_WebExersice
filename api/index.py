import os
import json
import pathlib
from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import psycopg2
import psycopg2.extras
import google.generativeai as genai

# --- Flask 템플릿 경로 설정 ---
BASE_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-prod')
TEACHER_PASSWORD = os.environ.get('TEACHER_PASSWORD')

# --- AI 모델 설정 (API 키 없으면 명시적으로 비활성) ---
api_key = os.environ.get('GEMINI_API_KEY')
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
    except Exception as e:
        model = None
        print(f"🚨 Gemini AI 모델 설정 오류: {e}")
else:
    print("⚠️ GEMINI_API_KEY 미설정: 채점 기능이 비활성화됩니다.")

# --- 데이터베이스 설정 ---
DATABASE_URL = os.environ.get('POSTGRES_URL')

def get_db_connection():
    try:
        # 필요 시 연결 문자열에 sslmode=require 포함 여부 확인
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"🚨 데이터베이스 연결 오류: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS exercises (
                        id SERIAL PRIMARY KEY,
                        korean_sentence TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS submissions (
                        id SERIAL PRIMARY KEY,
                        exercise_id INTEGER REFERENCES exercises(id),
                        student_id VARCHAR(255) NOT NULL,
                        student_answer TEXT,
                        score NUMERIC(3, 1),
                        ai_analysis_json JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                print("✅ 데이터베이스 테이블이 성공적으로 확인/생성되었습니다.")
        except Exception as e:
            print(f"🚨 테이블 생성 오류: {e}")
        finally:
            conn.close()

init_db()

# --- 안전한 JSON 본문 추출 유틸 ---
def extract_first_json_block(text: str):
    if not text:
        return None
    # 코드블럭 마커 제거
    t = text.replace("```json", "```").strip()
    if "```" in t:
        # 첫 번째 코드블럭만 취함
        parts = t.split("```")
        # parts는 ["서문", "json?", "후문"] 구조일 수 있음
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                return chunk
    # 코드블럭이 아니면 중괄호 범위 스캔
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end+1]
    return None

# --- 채점 프롬프트 ---
EVALUATION_PROMPT = """
당신은 이탈리아 학생에게 한국어를 가르치는, 매우 엄격하고 공정한 AI 언어 교사입니다.
당신의 임무는, 주어진 한국어 원문과 학생이 제출한 이탈리아어 번역 답안을 비교하여, 학생의 이해도를 10.0점 만점으로 채점하고 심층적인 분석을 제공하는 것입니다.

[채점 기준]
- 의미의 정확성
- 문법 및 어휘
- 점수는 반드시 0.0~10.0, 소수점 한 자리

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[출력 형식]
JSON ONLY:
{
  "score": "10.0 형식의 숫자 문자열",
  "analysis": {
    "original_korean_question": "채점의 기준이 된 한국어 원문",
    "student_answer_original": "학생 이탈리아어 원문",
    "student_answer_korean_translation": "학생 답안을 자연스러운 한국어로 번역",
    "score": "동일 점수",
    "key_phrases_italian": ["..."],
    "key_phrases_korean_translation": ["..."]
  }
}
"""

# --- 공용 라우트(학생용) ---
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/quiz')
def quiz_page():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return "데이터베이스에 연결할 수 없습니다.", 500

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, korean_sentence FROM exercises ORDER BY id;")
            exercises = cur.fetchall()

        return render_template('index.html', exercises=exercises)
    except Exception as e:
        print(f"🚨 /quiz 페이지 로딩 중 오류 발생: {e}")
        return "퀴즈를 불러오는 중 오류가 발생했습니다.", 500
    finally:
        if conn:
            conn.close()

# --- 학생 답안 제출 API ---
@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')

    if not all([student_id, student_answer, exercise_id]):
        return jsonify({"error": "필수 정보(학생 ID, 답안, 문제 ID)가 누락되었습니다."}), 400

    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "DB 연결 실패"}), 500

        # 문제 한국어 원문 조회
        with conn.cursor() as cur:
            cur.execute("SELECT korean_sentence FROM exercises WHERE id = %s;", (exercise_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "해당 ID의 문제를 찾을 수 없습니다."}), 404
            korean_question = row[0]

        if not model:
            return jsonify({"error": "AI 모델이 설정되지 않았습니다. GEMINI_API_KEY 확인"}), 500

        # JSON만 반환하도록 강제
        prompt_text = EVALUATION_PROMPT.format(
            Korean_Question=korean_question,
            Student_Answer=student_answer
        )
        response = model.generate_content(
            prompt_text,
            generation_config={"response_mime_type": "application/json"}
        )

        # 응답 텍스트 확보
        raw_text = (getattr(response, "text", None) or "").strip()
        if not raw_text and hasattr(response, "candidates") and response.candidates:
            try:
                raw_text = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                raw_text = ""

        if not raw_text:
            print("🚨 AI 응답이 비어 있습니다.")
            return jsonify({"error": "AI 응답이 비어 있습니다."}), 502

        # JSON 파싱 (코드블럭 제거 후 시도)
        json_str = extract_first_json_block(raw_text) or raw_text
        try:
            ai_result = json.loads(json_str)
        except Exception as e:
            print(f"🚨 AI JSON 디코딩 실패: {e}\nRAW: {raw_text[:400]}")
            return jsonify({"error": "AI 응답 파싱 실패"}), 502

        # 점수 처리
        score = None
        score_raw = ai_result.get('score')
        try:
            # 쉼표 소수점 등 대비
            score = round(float(str(score_raw).strip().replace(',', '.')), 1)
        except Exception:
            score = None

        # 분석 필드
        analysis = ai_result.get('analysis') or {}
        if "original_korean_question" not in analysis:
            analysis["original_korean_question"] = korean_question

        # DB 저장(JSONB는 Json 어댑터로 안전 삽입)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submissions (exercise_id, student_id, student_answer, score, ai_analysis_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    exercise_id,
                    student_id,
                    student_answer,
                    score,
                    psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False))
                )
            )
            conn.commit()

        return jsonify({"success": True, "score": score})
    except Exception as e:
        print(f"🚨 /api/submit-answer 처리 중 오류 발생: {e}")
        return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500
    finally:
        if conn:
            conn.close()
            
# --------------------------
# 교사용 로그인/대시보드
# --------------------------
def teacher_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('is_teacher'):
            return f(*args, **kwargs)
        return redirect(url_for('teacher_login'))
    return wrapper

@app.route('/teacher-login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        pwd = request.form.get('password')
        if TEACHER_PASSWORD and pwd == TEACHER_PASSWORD:
            session['is_teacher'] = True
            return redirect(url_for('dashboard'))
        return render_template('teacher_login.html', error='비밀번호가 틀렸습니다.')
    return render_template('teacher_login.html')

@app.route('/teacher-logout')
def teacher_logout():
    session.clear()
    return redirect(url_for('teacher_login'))

@app.route('/dashboard')
@teacher_required
def dashboard():
    return render_template('dashboard.html')

# 교사용: 제출 목록 API(자동 갱신용 - 폴링)
@app.route('/api/submissions', methods=['GET'])
def api_submissions():
    if not session.get('is_teacher'):
        return jsonify({"error": "unauthorized"}), 401

    since_id = request.args.get('since_id', default=0, type=int)
    limit = request.args.get('limit', default=50, type=int)

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    s.id, s.exercise_id, s.student_id, s.student_answer,
                    s.score, s.ai_analysis_json, s.created_at,
                    e.korean_sentence
                FROM submissions s
                JOIN exercises e ON e.id = s.exercise_id
                WHERE s.id > %s
                ORDER BY s.id ASC
                LIMIT %s
            """, (since_id, limit))
            rows = cur.fetchall()

        items = []
        for r in rows:
            analysis = r.get('ai_analysis_json') or {}
            original_ko = analysis.get("original_korean_question") or r.get("korean_sentence")
            s_val = r.get("score")
            try:
                s_num = float(s_val) if s_val is not None else None
            except Exception:
                s_num = None

            items.append({
                "id": r["id"],
                "student_id": r["student_id"],
                "student_answer": r["student_answer"],
                "score": s_num,
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "original_korean_question": original_ko,
                "student_answer_original": analysis.get("student_answer_original"),
                "student_answer_korean_translation": analysis.get("student_answer_korean_translation"),
                "key_phrases_italian": analysis.get("key_phrases_italian"),
                "key_phrases_korean_translation": analysis.get("key_phrases_korean_translation"),
            })
        return jsonify({"items": items})
    finally:
        if conn:
            conn.close()