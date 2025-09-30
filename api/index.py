import os
import json
import pathlib
import traceback
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

# --- AI 모델 설정 ---
api_key = os.environ.get('GEMINI_API_KEY')
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 모델 이름은 환경에 맞게 조정 가능
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
                # ★★★ 변경점: exercises 테이블 생성 시 class_name 컬럼 추가 ★★★
                cur.execute("CREATE TABLE IF NOT EXISTS exercises (id SERIAL PRIMARY KEY, korean_sentence TEXT NOT NULL, class_name VARCHAR(50), created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                # ★★★ 변경점: submissions 테이블 생성 시 class_name 컬럼 추가 ★★★
                cur.execute("CREATE TABLE IF NOT EXISTS submissions (id SERIAL PRIMARY KEY, exercise_id INTEGER REFERENCES exercises(id), student_id VARCHAR(255) NOT NULL, student_answer TEXT, score NUMERIC(3, 1), ai_analysis_json JSONB, class_name VARCHAR(50), created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                conn.commit()
                print("✅ 데이터베이스 테이블이 성공적으로 확인/생성되었습니다.")
        except Exception as e:
            print(f"🚨 테이블 생성 오류: {e}")
        finally:
            conn.close()
init_db()

def extract_first_json_block(text: str):
    if not text: return None
    t = text.replace("```json", "```").strip()
    if "```" in t:
        parts = t.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"): return chunk
    start = t.find("{"); end = t.rfind("}")
    if start != -1 and end != -1 and end > start: return t[start:end+1]
    return None

# --- 채점 프롬프트 (변경 없음) ---
EVALUATION_PROMPT = """
... (기존 프롬프트 내용과 동일) ...
"""

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')
    # ★★★ 추가: 프론트엔드에서 class_name을 받아옵니다 ★★★
    class_name = data.get('class_name')

    if not all([student_id, student_answer, exercise_id, class_name]):
        return jsonify({"error": "필수 정보 누락 (student_id, answer, exercise_id, class_name)"}), 400

    conn = None
    try:
        conn = get_db_connection()
        if conn is None: return jsonify({"error": "DB 연결 실패"}), 500

        with conn.cursor() as cur:
            cur.execute("SELECT korean_sentence FROM exercises WHERE id = %s;", (exercise_id,))
            row = cur.fetchone()
            if not row: return jsonify({"error": "문제 ID 없음"}), 404
            korean_question = row[0]

        if not model: return jsonify({"error": "AI 모델 미설정"}), 500

        response = None
        try:
            prompt_text = EVALUATION_PROMPT.format(Korean_Question=korean_question, Student_Answer=student_answer)
            response = model.generate_content(
                prompt_text,
                generation_config={"response_mime_type": "application/json"}
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "AI 모델 호출 중 오류 발생"}), 500

        # ... (AI 응답 처리 로직은 기존과 동일) ...
        raw_text = getattr(response, 'text', '').strip()
        if not raw_text: return jsonify({"error": "AI로부터 빈 응답"}), 502
        json_str = extract_first_json_block(raw_text) or raw_text
        try:
            ai_result = json.loads(json_str)
        except json.JSONDecodeError:
            return jsonify({"error": "AI 응답 JSON 해석 실패"}), 502

        score_raw = ai_result.get('score')
        score = None
        try:
            if score_raw is not None: score = round(float(str(score_raw).strip().replace(',', '.')), 1)
        except (ValueError, TypeError): pass
        analysis = ai_result.get('analysis', {})
        if 'original_korean_question' not in analysis: analysis['original_korean_question'] = korean_question
        if 'student_answer_original' not in analysis: analysis['student_answer_original'] = student_answer
        if 'score' not in analysis and score is not None: analysis['score'] = str(score)

        with conn.cursor() as cur:
            # ★★★ 변경점: INSERT 쿼리에 class_name을 추가하여 저장합니다 ★★★
            cur.execute(
                "INSERT INTO submissions (exercise_id, student_id, student_answer, score, ai_analysis_json, class_name) VALUES (%s, %s, %s, %s, %s, %s)",
                (exercise_id, student_id, student_answer, score, psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False)), class_name)
            )
            conn.commit()

        return jsonify({"success": True, "score": score})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "서버 내부 오류"}), 500
    finally:
        if conn: conn.close()

# --- 페이지 라우팅 로직 (대규모 변경) ---

def teacher_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('is_teacher'): return f(*args, **kwargs)
        return redirect(url_for('teacher_login'))
    return wrapper

# ★★★ 삭제: 기존의 @app.route('/')는 더 이상 사용하지 않습니다 ★★★
# @app.route('/')
# def login(): return render_template('login.html')

# ★★★ 신규: 모든 학생은 이제 이 주소로 접속합니다 ★★★
@app.route('/')
@app.route('/class/<class_name>')
def student_login(class_name=None):
    if not class_name:
        # 만약 아무 반 이름 없이 접속하면, 기본값으로 설정하거나 에러 페이지를 보여줄 수 있습니다.
        # 여기서는 'siena-3'를 기본값으로 설정하겠습니다.
        class_name = 'siena-3'
    
    # 세션에 학생이 어느 반으로 접속했는지 기록합니다.
    session['class_name'] = class_name
    return render_template('login.html', class_name=class_name)


@app.route('/quiz')
def quiz_page():
    # ★★★ 변경점: 세션에서 class_name을 가져옵니다 ★★★
    class_name = session.get('class_name')
    if not class_name:
        # 만약 비정상적인 접근으로 class_name이 없다면, 로그인 페이지로 돌려보냅니다.
        return redirect(url_for('student_login'))

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # ★★★ 변경점: 해당 반에 맞는 문제만 선택(SELECT)합니다 ★★★
            cur.execute("SELECT id, korean_sentence FROM exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            exercises = cur.fetchall()
        
        # ★★★ 변경점: 템플릿에 class_name을 전달하여, 프론트엔드가 알 수 있도록 합니다 ★★★
        return render_template('index.html', exercises=exercises, class_name=class_name)
    finally:
        if conn: conn.close()

@app.route('/teacher-login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        if TEACHER_PASSWORD and request.form.get('password') == TEACHER_PASSWORD:
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
def dashboard(): return render_template('dashboard.html')

# index.py 파일에서 아래 함수를 찾아서 교체해주세요.

@app.route('/api/submissions')
@teacher_required
def api_submissions():
    if not session.get('is_teacher'): return jsonify({"error": "unauthorized"}), 401
    
    since_id = request.args.get('since_id', 0, type=int)
    # ★★★ 추가: 대시보드로부터 class_name 필터 값을 받습니다. ★★★
    class_name_filter = request.args.get('class_name', 'all')

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ★★★ 변경점: class_name 필터 값에 따라 SQL 쿼리를 동적으로 구성합니다. ★★★
            
            # 기본 쿼리문
            query = "SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, s.created_at, e.korean_sentence, s.class_name FROM submissions s JOIN exercises e ON e.id = s.exercise_id WHERE s.id > %s"
            params = [since_id]
            
            # '전체 보기'가 아닌 특정 반 필터가 선택된 경우
            if class_name_filter != 'all':
                query += " AND s.class_name = %s"
                params.append(class_name_filter)
            
            query += " ORDER BY s.id ASC LIMIT 50"
            
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            items.append(r)
        return jsonify({"items": items})
    finally:
        if conn: conn.close()