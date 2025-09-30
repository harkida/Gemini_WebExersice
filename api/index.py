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
        model = genai.GenerativeModel('gemini-2.5-flash')
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

# ★★★ [핵심 수정] 데이터베이스 초기화 로직을 '반별 기능'에 맞게 전면 수정합니다. ★★★
def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # exercises 테이블에 class_name 컬럼이 없으면 추가합니다.
                cur.execute("ALTER TABLE exercises ADD COLUMN IF NOT EXISTS class_name VARCHAR(50);")
                # submissions 테이블에 class_name 컬럼이 없으면 추가합니다.
                cur.execute("ALTER TABLE submissions ADD COLUMN IF NOT EXISTS class_name VARCHAR(50);")
                
                # 테이블이 존재하지 않을 경우를 대비한 생성 구문 (기존 구조와 호환)
                cur.execute("CREATE TABLE IF NOT EXISTS exercises (id SERIAL PRIMARY KEY, korean_sentence TEXT NOT NULL, class_name VARCHAR(50), created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                cur.execute("CREATE TABLE IF NOT EXISTS submissions (id SERIAL PRIMARY KEY, exercise_id INTEGER REFERENCES exercises(id), student_id VARCHAR(255) NOT NULL, student_answer TEXT, score NUMERIC(3, 1), ai_analysis_json JSONB, class_name VARCHAR(50), created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);")
                
                conn.commit()
                print("✅ 데이터베이스 테이블이 '반별 기능'에 맞게 성공적으로 확인/수정되었습니다.")
        except Exception as e:
            print(f"🚨 테이블 생성/수정 오류: {e}")
            conn.rollback()
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

# --- 채점 프롬프트 (교수님 지시대로 축약) ---
EVALUATION_PROMPT = """

너는 한국어와 이탈리아어에 모두 능통한 언어 평가 전문가이다. 너의 유일한 임무는 '한국어 원문'을 듣고 학생이 작성한 '이탈리아어 답안'이 원문의 의미를 얼마나 정확하게 이해하고 반영했는지를 평가하는 것이다.

[핵심 원칙]
이것은 이탈리아어 작문 시험이 아니다. 학생의 이탈리아어 문법이 다소 어색하거나 사소한 오류가 있더라도, 원문의 의미를 이해했다고 판단되면 절대 감점하지 마라. 평가는 오직 '의미의 정확성' 하나만을 기준으로 한다.

[채점 기준: 의미의 정확성 (Semantic Accuracy) - 100%]
1.  만점(10.0)에서 시작한다.
2.  **점수는 반드시 소수점 첫째 자리까지 평가해야 한다 (예: 9.6, 8.1, 7.3). 정수(7, 8, 9)로만 점수를 매기는 것은 허용되지 않는다.**
3.  아래 기준에 따라 오류를 발견할 때마다 점수를 차감한다.
    -   **완전한 오역 또는 의미 왜곡:** 원문의 핵심 의미를 완전히 잘못 이해하여 정반대의 의미나 전혀 다른 의미로 번역한 경우. (감점: -5.1 ~ -8.0점)
    -   **핵심 정보 누락/오류:** 문장의 주어, 목적어, 동사 등 핵심적인 구성 요소나 정보를 빠뜨리거나 틀리게 번역한 경우. (감점: -2.6 ~ -5.0점)
    -   **사소한 의미 불일치:** 전체적인 의미는 맞지만, 특정 단어나 표현의 뉘앙스를 잘못 이해하여 약간의 의미 차이가 발생한 경우. (감점: -0.5 ~ -2.5점) # 감점 폭 미세 조정

4.  **뉘앙스 및 격식 (Nuance & Formality):**
    -   **이것은 절대 감점 요인이 아니다.** 관용구의 번역(예: '표를 끊다' -> 'comprare i biglietti')이나, 존댓말/반말, 어조, 단어 선택의 미묘한 차이는 '오류'로 간주해서는 안 되며, 절대로 감점의 근거가 될 수 없다.
    -   다만, 이러한 차이점이 교육적으로 의미가 있다고 판단될 경우, 반드시 'evaluation_feedback'에 **[교사용 참고]** 태그를 사용하여 그 차이점만 객관적으로 서술한다. (예: "[교사용 참고] 원문의 관용구 '표를 끊다'는 '표를 사다'는 의미로, 학생의 'comprare' 사용은 자연스럽고 올바른 번역입니다.")

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[출력 형식]
JSON ONLY. 다른 설명 없이 JSON 객체만 반환해야 합니다. 점수 계산 근거와 교육적 피드백을 'evaluation_feedback'에 상세히 서술해야 한다.
# ★★★ [핵심 수정 2] 출력 형식 예시에도 소수점 사용을 명확히 보여줍니다. ★★★
{{
  "score": "9.5, 8.0, 7.5 등과 같은 10.0 형식의 숫자 문자열",
  "analysis": {{
    "original_korean_question": "채점의 기준이 된 한국어 원문",
    "student_answer_original": "학생이 제출한 이탈리아어 답안 원문",
    "student_answer_korean_translation": "학생의 이탈리아어 답안을 자연스러운 한국어로 번역한 결과",
    "score": "채점된 점수와 동일한 값",
    "key_vocabularies_italian": ["추출된 이탈리아어 어휘 기본형"],
    "key_vocabularies_korean_translation": ["위 이탈리아어 어휘의 한국어 뜻"],
    "evaluation_feedback": "AI의 채점 근거와 교육적 피드백에 대한 상세한 서술. 어떤 오류 때문에 몇 점이 감점되었는지 명확히 설명하고, 뉘앙스 차이는 [교사용 참고] 태그를 붙여 보고한다."
  }}
}}
"""

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')
    # ★★★ [핵심 수정] 요청 본문에서 class_name을 가져옵니다. ★★★
    class_name = data.get('class_name')

    # ★★★ [핵심 수정] class_name도 필수 정보로 확인합니다. ★★★
    if not all([student_id, student_answer, exercise_id, class_name]):
        return jsonify({"error": "필수 정보 누락 (반 정보 포함)"}), 400

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

        prompt_text = EVALUATION_PROMPT.format(Korean_Question=korean_question, Student_Answer=student_answer)
        response = model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
        raw_text = getattr(response, 'text', '').strip()
        json_str = extract_first_json_block(raw_text) or raw_text
        ai_result = json.loads(json_str)
        score_raw = ai_result.get('score')
        score = None
        if score_raw is not None:
            score = round(float(str(score_raw).strip().replace(',', '.')), 1)
        analysis = ai_result.get('analysis', {})
        if 'original_korean_question' not in analysis: analysis['original_korean_question'] = korean_question
        if 'student_answer_original' not in analysis: analysis['student_answer_original'] = student_answer
        if 'score' not in analysis and score is not None: analysis['score'] = str(score)
        
        with conn.cursor() as cur:
            # ★★★ [핵심 수정] INSERT 구문에 class_name을 추가하여 저장합니다. ★★★
            cur.execute(
                "INSERT INTO submissions (exercise_id, student_id, student_answer, score, ai_analysis_json, class_name) VALUES (%s, %s, %s, %s, %s, %s)",
                (exercise_id, student_id, student_answer, score, psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False)), class_name)
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

# ★★★ [핵심 수정] /quiz 라우트가 반 별로 문제를 필터링합니다. ★★★
@app.route('/quiz')
def quiz_page():
    class_name = request.args.get('class_name')
    if not class_name:
        return redirect(url_for('login'))

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT id, korean_sentence FROM exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            exercises = cur.fetchall()
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

@app.route('/api/submissions')
@teacher_required
def api_submissions():
    if not session.get('is_teacher'): return jsonify({"error": "unauthorized"}), 401
    since_id = request.args.get('since_id', 0, type=int)
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ★★★ [핵심 수정] submissions 테이블에서 class_name도 함께 가져옵니다. ★★★
            cur.execute("SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, s.created_at, e.korean_sentence, s.class_name FROM submissions s JOIN exercises e ON e.id = s.exercise_id WHERE s.id > %s ORDER BY s.id ASC LIMIT 50", (since_id,))
            rows = cur.fetchall()
        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            items.append(r)
        return jsonify({"items": items})
    finally:
        if conn: conn.close()