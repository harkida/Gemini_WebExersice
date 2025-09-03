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
                cur.execute("CREATE TABLE IF NOT EXISTS exercises (id SERIAL PRIMARY KEY, korean_sentence TEXT NOT NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                cur.execute("CREATE TABLE IF NOT EXISTS submissions (id SERIAL PRIMARY KEY, exercise_id INTEGER REFERENCES exercises(id), student_id VARCHAR(255) NOT NULL, student_answer TEXT, score NUMERIC(3, 1), ai_analysis_json JSONB, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
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

# --- 채점 프롬프트 (★★★ 핵심 수정 부분 ★★★) ---
EVALUATION_PROMPT = """
당신은 이탈리아 학생에게 한국어를 가르치는, 매우 엄격하고 공정한 AI 언어 교사입니다. 당신의 임무는 학생의 답안을 채점하고, 교사를 위한 심층 분석 자료를 생성하는 것입니다.

[채점 기준]
- 의미의 정확성, 문법, 어휘를 종합하여 10.0점 만점으로 채점합니다.

[핵심 어휘 추출 규칙]
- 학생의 이탈리아어 답안에서, 유럽언어기준(CEFR) B1 레벨 이상의 학습 가치가 있는 핵심 이탈리아어 어휘를 추출합니다. (개수 제한 없음)
- 만약 B1 레벨 이상의 어휘가 없다면, 대신 가장 중요하다고 판단되는 A2 레벨의 명사 또는 동사를 추출합니다.
- 각 어휘는 반드시 동사 원형(infinito)이나 명사의 단수형 등 기본 형태로 변환하여 `key_vocabularies_italian` 배열에 추가합니다.
- 각 어휘에 대한 한국어 뜻을 `key_vocabularies_korean_translation` 배열에 추가합니다.
- 추출할 어휘가 아예 없는 경우에만 두 배열 모두 빈 상태 `[]`로 둡니다.

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[출력 형식]
JSON ONLY. 다른 설명 없이 JSON 객체만 반환해야 합니다.
{{
  "score": "10.0 형식의 숫자 문자열",
  "analysis": {{
    "original_korean_question": "채점의 기준이 된 한국어 원문",
    "student_answer_original": "학생이 제출한 이탈리아어 답안 원문",
    "student_answer_korean_translation": "학생의 이탈리아어 답안을 자연스러운 한국어로 번역한 결과",
    "score": "채점된 점수와 동일한 값",
    "key_vocabularies_italian": ["추출된 이탈리아어 어휘 기본형"],
    "key_vocabularies_korean_translation": ["위 이탈리아어 어휘의 한국어 뜻"]
  }}
}}
"""

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')

    if not all([student_id, student_answer, exercise_id]):
        return jsonify({"error": "필수 정보 누락"}), 400

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
            print(f"🚨🚨🚨 AI 모델 호출(generate_content) 자체에서 심각한 오류 발생! 🚨🚨🚨")
            print(f"오류 타입: {type(e)}")
            print(f"오류 메시지: {e}")
            traceback.print_exc()
            return jsonify({"error": "AI 모델 호출 중 심각한 오류가 발생했습니다."}), 500

        if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            print(f"🚨 AI 프롬프트가 차단되었습니다. 이유: {block_reason}")
            return jsonify({"error": f"AI가 유해성 등의 이유로 응답을 거부했습니다: {block_reason}"}), 503

        raw_text = getattr(response, 'text', '').strip()
        if not raw_text:
            print("🚨 AI 응답이 비어 있습니다. 전체 응답 객체를 확인합니다.")
            print(f"AI 응답 객체 전문: {response}")
            return jsonify({"error": "AI로부터 빈 응답을 받았습니다."}), 502

        print(f"✅ AI로부터 받은 RAW 응답: {raw_text[:500]}")

        json_str = extract_first_json_block(raw_text) or raw_text
        try:
            ai_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"🚨 AI JSON 디코딩 실패: {e}\nRAW TEXT: {raw_text[:500]}")
            return jsonify({"error": "AI 응답을 JSON으로 해석하는데 실패했습니다."}), 502

        score_raw = ai_result.get('score')
        score = None
        try:
            if score_raw is not None:
                score = round(float(str(score_raw).strip().replace(',', '.')), 1)
        except (ValueError, TypeError) as e:
            print(f"⚠️ 'score' 값 '{score_raw}'을(를) 숫자로 변환하는 데 실패했습니다. 오류: {e}")

        analysis = ai_result.get('analysis', {})
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO submissions (exercise_id, student_id, student_answer, score, ai_analysis_json) VALUES (%s, %s, %s, %s, %s)",
                (exercise_id, student_id, student_answer, score, psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False)))
            )
            conn.commit()

        return jsonify({"success": True, "score": score})

    except Exception as e:
        print(f"🚨 /api/submit-answer 의 예측하지 못한 위치에서 오류 발생: {e}")
        traceback.print_exc()
        return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500
    finally:
        if conn: conn.close()
        
# --- 나머지 라우트 ---
def teacher_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('is_teacher'): return f(*args, **kwargs)
        return redirect(url_for('teacher_login'))
    return wrapper

@app.route('/')
def login(): return render_template('login.html')

@app.route('/quiz')
def quiz_page():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, korean_sentence FROM exercises ORDER BY id;")
            exercises = cur.fetchall()
        return render_template('index.html', exercises=exercises)
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

@app.route('/api/submissions')
@teacher_required
def api_submissions():
    if not session.get('is_teacher'): return jsonify({"error": "unauthorized"}), 401
    since_id = request.args.get('since_id', 0, type=int)
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, s.created_at, e.korean_sentence FROM submissions s JOIN exercises e ON e.id = s.exercise_id WHERE s.id > %s ORDER BY s.id ASC LIMIT 50", (since_id,))
            rows = cur.fetchall()
        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            items.append(r)
        return jsonify({"items": items})
    finally:
        if conn: conn.close()