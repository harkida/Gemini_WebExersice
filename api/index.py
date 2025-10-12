import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import pathlib
import traceback
from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import psycopg2
import psycopg2.extras
import google.generativeai as genai

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-prod')
TEACHER_PASSWORD = os.environ.get('TEACHER_PASSWORD')

api_key = os.environ.get('GEMINI_API_KEY')
flash_model = None
pro_model = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        flash_model = genai.GenerativeModel('gemini-2.5-flash')
        pro_model = genai.GenerativeModel('gemini-2.5-pro')
        print("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
        print("   📌 번역 퀴즈: gemini-2.5-flash (빠르고 경제적)")
        print("   📌 이해력 퀴즈: gemini-2.5-pro (정밀한 평가)")
    except Exception as e:
        flash_model = None
        pro_model = None
        print(f"🚨 Gemini AI 모델 설정 오류: {e}")
else:
    print("⚠️ GEMINI_API_KEY 미설정: 채점 기능이 비활성화됩니다.")

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
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS translation_exercises (
                        id SERIAL PRIMARY KEY,
                        korean_sentence TEXT NOT NULL,
                        class_name VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS translation_submissions (
                        id SERIAL PRIMARY KEY,
                        exercise_id INTEGER REFERENCES translation_exercises(id) ON DELETE SET NULL,
                        student_id VARCHAR(255) NOT NULL,
                        student_answer TEXT,
                        score NUMERIC(3, 1),
                        ai_analysis_json JSONB,
                        class_name VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS comprehension_exercises (
                        id SERIAL PRIMARY KEY,
                        korean_dialogue TEXT NOT NULL,
                        audio_file_path VARCHAR(255),
                        key_points JSONB,
                        class_name VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS comprehension_submissions (
                        id SERIAL PRIMARY KEY,
                        comprehension_exercise_id INTEGER REFERENCES comprehension_exercises(id) ON DELETE SET NULL,
                        student_id VARCHAR(255) NOT NULL,
                        class_name VARCHAR(50),
                        student_answer TEXT,
                        ai_analysis_json JSONB,
                        feedback_korean TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                cur.execute("""
                    ALTER TABLE comprehension_submissions 
                    ADD COLUMN IF NOT EXISTS feedback_korean TEXT;
                """)
                
                conn.commit()
                print("✅ 데이터베이스 테이블이 최종 블루프린트에 맞게 성공적으로 확인/생성되었습니다.")
        except Exception as e:
            print(f"🚨 테이블 구조 설정 중 심각한 오류 발생: {e}")
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

def translate_italian_to_korean(italian_text):
    """AI를 사용하여 이탈리아어 텍스트를 한국어로 번역"""
    if not flash_model or not italian_text:
        return "(번역 불가)"
    
    try:
        prompt = f"""다음 이탈리아어 텍스트를 자연스러운 한국어로 번역해주세요. 번역문만 출력하고 다른 설명은 하지 마세요.

이탈리아어 원문:
{italian_text}

한국어 번역:"""
        
        response = flash_model.generate_content(prompt)
        korean_translation = getattr(response, 'text', '').strip()
        return korean_translation if korean_translation else "(번역 실패)"
    except Exception as e:
        print(f"🚨 번역 오류: {e}")
        return "(번역 오류)"

EVALUATION_PROMPT = """
너는 한국어와 이탈리아어에 모두 능통한 언어 평가 전문가이다. 너의 유일한 임무는 '한국어 원문'을 듣고 학생이 작성한 '이탈리아어 답안'이 원문의 의미를 얼마나 정확하게 이해하고 반영했는지를 평가하는 것이다.

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[핵심 원칙]
이것은 이탈리아어 작문 시험이 아니다. 학생의 이탈리아어 문법이 다소 어색하거나 사소한 오류가 있더라도, 원문의 의미를 이해했다고 판단되면 절대 감점하지 마라. 평가는 오직 '의미의 정확성' 하나만을 기준으로 한다.

[채점 기준: 의미의 정확성 (Semantic Accuracy) - 100%]
    1.  만점(10.0)에서 시작한다.
    2.  **점수는 반드시 소수점 첫째 자리까지 평가해야 한다 (예: 9.6, 8.1, 7.3). 정수(7, 8, 9)로만 점수를 매기는 것은 허용되지 않는다.**
    3.  아래 기준에 따라 오류를 발견할 때마다 점수를 차감한다.
        -   **완전한 오역 또는 의미 왜곡:** 원문의 핵심 의미를 완전히 잘못 이해하여 정반대의 의미나 전혀 다른 의미로 번역한 경우. (감점: -5.1 ~ -8.0점)
        -   **핵심 정보 누락/오류:** 문장의 주어, 목적어, 동사 등 핵심적인 구성 요소나 정보를 빠뜨리거나 틀리게 번역한 경우. (감점: -2.6 ~ -5.0점)
        -   **사소한 의미 불일치:** 전체적인 의미는 맞지만, 특정 단어나 표현의 뉘앙스를 잘못 이해하여 약간의 의미 차이가 발생한 경우. (감점: -0.5 ~ -2.5점)

    4.  **뉘앙스 및 격식 (Nuance & Formality):**
        -   **이것은 절대 감점 요인이 아니다.** 관용구의 번역(예: '표를 끊다' -> 'comprare i biglietti')이나, 존댓말/반말, 어조, 단어 선택의 미묘한 차이는 '오류'로 간주해서는 안 되며, 절대로 감점의 근거가 될 수 없다.
        -   다만, 이러한 차이점이 교육적으로 의미가 있다고 판단될 경우, 반드시 'evaluation_feedback'에 **[교사용 참고]** 태그를 사용하여 그 차이점만 객관적으로 서술한다.

[출력 형식]
JSON ONLY. 다른 설명 없이 JSON 객체만 반환해야 합니다.
{{
  "score": "9.5, 8.0, 7.5 등과 같은 10.0 형식의 숫자 문자열",
  "analysis": {{
    "original_korean_question": "채점의 기준이 된 한국어 원문",
    "student_answer_original": "학생이 제출한 이탈리아어 답안 원문",
    "student_answer_korean_translation": "학생의 이탈리아어 답안을 자연스러운 한국어로 번역한 결과",
    "score": "채점된 점수와 동일한 값",
    "key_vocabularies_italian": ["추출된 이탈리아어 어휘 기본형"],
    "key_vocabularies_korean_translation": ["위 이탈리아어 어휘의 한국어 뜻"],
    "evaluation_feedback": "AI의 채점 근거와 교육적 피드백에 대한 상세한 서술."
  }}
}}
"""

COMPREHENSION_EVALUATION_PROMPT = """
You are an expert AI assistant specializing in Korean language education for Italian students. Your mission is to evaluate how well a student has understood a Korean dialogue based on specific scoring criteria (`key_points`) set by the professor.

[Input Information]
- **Original Korean Dialogue:** "{korean_dialogue}"
- **Student's Italian Answer:** "{student_answer}"
- **Professor's Scoring Criteria (key_points):** {key_points_json}

[Scoring Structure - Total: 10.0 points]

1. Target Vocabulary Assessment (목표 어휘 평가) - 30% (3.0 points)
   
   Calculate the vocabulary coverage ratio: 
    - vocabulary_score = (number of target vocabulary used / total target vocabulary) × 3.0
    - Valid synonyms count as "used"
    - If a student uses a different but semantically correct word, award full credit for that vocabulary item
    - Partial credit is NOT given per vocabulary item (it's either used correctly or not)

    **Examples:**
    - 4 target words, student used 3 correctly → (3/4) × 3.0 = 2.25 points
    - 2 target words, student used 2 correctly → (2/2) × 3.0 = 3.0 points
    - 6 target words, student used 4 correctly → (4/6) × 3.0 = 2.0 points

2. Meaning Points Coverage (핵심 의미 포괄도) - 60% (6.0 points)

    Evaluate each meaning_point individually, then calculate: 
        - meaning_score = (sum of individual meaning_point scores / total number of meaning_points) × 6.0
    For each meaning_point, assign a score from 0.0 to 1.0:
        - **1.0:** Fully covered (all aspects of the meaning_point are clearly present)
        - **0.5-0.7:** Partially covered (some aspects mentioned, but key details missing)
        - Example: A meaning_point states "기기는 옛날에는 자주 사용되었지만, 지금은 잘 사용되지 않는다"
        - Student only mentions "지금은 사용 안 함" → 0.5-0.6
        - Student mentions both past and present → 1.0
        - **0.0:** Not covered at all

    **Examples:**
        - 4 meaning_points, scores: [1.0, 0.6, 1.0, 0.0] → (2.6/4) × 6.0 = 3.9 points
        - 2 meaning_points, scores: [1.0, 1.0] → (2.0/2) × 6.0 = 6.0 points
        - 5 meaning_points, scores: [1.0, 0.7, 1.0, 0.5, 0.0] → (3.2/5) × 6.0 = 3.84 points

    **Critical Rule:** 
        - If meaning_points coverage is below 80% (sum/total < 0.8), the final score is CAPPED at 8.0
        - This ensures that superficial summaries cannot achieve top scores

3. Factual Accuracy (사실 정확성) - 10% (1.0 point baseline)

    Start with 1.0 points, then apply deductions:
        - **Over-inference (과잉 추론):** Student adds information NOT stated in the dialogue
            → Deduct 0.5-1.0 points per instance

        - **Factual error (사실 오류):** Student states incorrect information
            → Deduct 1.0-2.0 points per error

        - **Subject/object confusion (주체/객체 혼동):** Critical error
            → Deduct 1.5-2.0 points

    The accuracy score can go below 0.0 (resulting in negative contribution to total score)

    Critical Rule:
        - If there are ANY factual errors or over-inferences, final score is CAPPED at 7.5
        - This prevents students from writing verbose but inaccurate answers

4. Bonus Points (추가 정확한 정보) - Maximum +0.5 points

    If the student mentions accurate details from the dialogue NOT listed in `meaning_points`:
        - Award +0.1 to +0.3 per accurate additional fact
        - Maximum total bonus: +0.5 points

    Important: Bonus points are awarded ONLY if:
        - The information is explicitly stated in the dialogue
        - The information is factually correct
        - No accuracy deductions have been applied (errors disqualify bonus points)

[Evaluation Process]
1. Count total number of `target_vocabulary` items
2. Count how many the student used correctly → Calculate vocabulary_score
3. Count total number of `meaning_points`
4. Evaluate each meaning_point (0.0 to 1.0) → Calculate meaning_score
5. Start with accuracy_score = 1.0, apply deductions for errors
6. Check for bonus-worthy additional accurate information
7. Calculate preliminary score: vocabulary_score + meaning_score + accuracy_score + bonus
8. **Apply score caps:**
    - If meaning_points coverage < 80% → cap at 8.0
    - If factual errors exist → cap at 7.5
9. Round to one decimal place (e.g., 7.3, 8.5, 9.2)

[Output Format - JSON Only]

{{
  "score": 8.5,
  "student_answer_original": "학생이 제출한 이탈리아어 답안 원문",
  "student_answer_korean_translation": "학생의 이탈리아어 답안을 자연스러운 한국어로 번역한 결과",
  "key_vocabularies_italian": ["학생 답안에서 추출된 핵심 이탈리아어 어휘의 기본형"],
  "key_vocabularies_korean_translation": ["위 이탈리아어 어휘들의 한국어 뜻"],
  "evaluation": "(한국어로) 상세한 채점 근거",
  "feedback": "(이탈리아어로) 학생을 위한 격려와 건설적 피드백"
}}

Important:
- The evaluation field MUST show detailed calculations with actual numbers
- Clearly state the coverage percentage for meaning_points
- If a score cap is applied, explain why
"""

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')
    class_name = data.get('class_name')
    quiz_type = data.get('quiz_type')

    if not all([student_id, student_answer, exercise_id, class_name, quiz_type]):
        return jsonify({"error": "필수 정보 누락 (퀴즈 유형 포함)"}), 400

    conn = None
    korean_text = ""
    
    try:
        conn = get_db_connection()
        if conn is None: return jsonify({"error": "DB 연결 실패"}), 500
        
        if quiz_type == 'translation':
            selected_model = flash_model
            model_name = "Flash"
        elif quiz_type == 'comprehension':
            selected_model = pro_model
            model_name = "Pro"
        else:
            return jsonify({"error": "잘못된 퀴즈 유형"}), 400
        
        if not selected_model:
            return jsonify({"error": f"AI 모델 미설정 ({model_name})"}), 500

        with conn.cursor() as cur:
            if quiz_type == 'translation':
                cur.execute("SELECT korean_sentence FROM translation_exercises WHERE id = %s;", (exercise_id,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "문제 ID 없음"}), 404
                korean_question = row[0]
                korean_text = korean_question

                prompt_text = EVALUATION_PROMPT.format(Korean_Question=korean_question, Student_Answer=student_answer)
                response = selected_model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
                print(f"🤖 [번역 퀴즈] gemini-2.5-flash 사용 - 학생: {student_id}")
                
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text
                ai_result = json.loads(json_str)
                
                score_raw = ai_result.get('score')
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw else None
                analysis = ai_result.get('analysis', {})
                
                cur.execute(
                    "INSERT INTO translation_submissions (exercise_id, student_id, student_answer, score, ai_analysis_json, class_name) VALUES (%s, %s, %s, %s, %s, %s)",
                    (exercise_id, student_id, student_answer, score, psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False)), class_name)
                )
                
            elif quiz_type == 'comprehension':
                cur.execute("SELECT korean_dialogue, key_points FROM comprehension_exercises WHERE id = %s;", (exercise_id,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "문제 ID 없음"}), 404
                korean_dialogue, key_points = row[0], row[1]
                korean_text = korean_dialogue

                prompt_text = COMPREHENSION_EVALUATION_PROMPT.format(
                    korean_dialogue=korean_dialogue,
                    student_answer=student_answer, 
                    key_points_json=json.dumps(key_points, ensure_ascii=False)
                )

                response = selected_model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
                print(f"🤖 [이해력 퀴즈] gemini-2.5-pro 사용 - 학생: {student_id}")
                
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text
                ai_result = json.loads(json_str)
                
                score_raw = ai_result.get('score')
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw else None
                
                feedback_italian = ai_result.get('feedback', 'Nessun feedback disponibile.')
                if feedback_italian and feedback_italian != 'Nessun feedback disponibile.':
                    feedback_korean = translate_italian_to_korean(feedback_italian)
                    print(f"📝 피드백 번역 완료: {len(feedback_korean)}자")
                else:
                    feedback_korean = '(피드백 없음)'
                
                cur.execute(
                    """INSERT INTO comprehension_submissions 
                       (comprehension_exercise_id, student_id, student_answer, ai_analysis_json, feedback_korean, class_name) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (exercise_id, student_id, student_answer, 
                     psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False)), 
                     feedback_korean, class_name)
                )

            conn.commit()

        def get_rating_details(score):
            score = float(score) if score else 0
            if score >= 8.6: return {"category": "Eccellente", "color": "teal"}
            if score >= 7.1: return {"category": "Buono", "color": "lightgreen"}
            if score >= 5.6: return {"category": "Sufficiente", "color": "gold"}
            if score >= 4.1: return {"category": "Da migliorare", "color": "orange"}
            return {"category": "Riprova", "color": "red"}

        rating_info = get_rating_details(score)

        if quiz_type == 'translation':
            student_feedback = analysis.get('evaluation_feedback', 'Nessun feedback disponibile.')
        elif quiz_type == 'comprehension':
            student_feedback = ai_result.get('feedback', 'Nessun feedback disponibile.')
        else:
            student_feedback = 'Feedback non disponibile.'

        return jsonify({
            "success": True, 
            "score": score,
            "rating_category": rating_info["category"],
            "rating_color": rating_info["color"],
            "feedback": student_feedback,
            "korean_text": korean_text
        })    

    except Exception as e:
        print(f"🚨 /api/submit-answer 오류: {e}")
        traceback.print_exc()
        if conn: conn.rollback()
        return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500
    finally:
        if conn: conn.close()
        
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
    class_name = request.args.get('class_name')
    quiz_type = request.args.get('quiz_type')
    
    if not class_name or not quiz_type:
        return redirect(url_for('login'))

    conn = get_db_connection()
    if not conn:
        return "데이터베이스 연결에 실패했습니다.", 500
        
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if quiz_type == 'translation':
                cur.execute("SELECT id, korean_sentence AS question_text FROM translation_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            elif quiz_type == 'comprehension':
                cur.execute("SELECT id, korean_dialogue AS question_text, audio_file_path FROM comprehension_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            else:
                return "잘못된 퀴즈 유형입니다.", 400
            
            exercises = cur.fetchall()
        
        return render_template('index.html', exercises=exercises, class_name=class_name, quiz_type=quiz_type)
    except Exception as e:
        print(f"🚨 /quiz 페이지 로딩 오류: {e}")
        return "퀴즈를 불러오는 중 오류가 발생했습니다.", 500
    finally:
        if conn:
            conn.close()

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

@app.route('/api/get-translation-submissions')
@teacher_required
def api_translation_submissions():
    if not session.get('is_teacher'): return jsonify({"error": "unauthorized"}), 401
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, s.created_at, 
                       e.korean_sentence, s.class_name 
                FROM translation_submissions s 
                JOIN translation_exercises e ON e.id = s.exercise_id 
                ORDER BY s.id DESC LIMIT 100
            """)
            rows = cur.fetchall()
        
        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            items.append(r)
        return jsonify({"items": items, "quiz_type": "translation"})
    finally:
        if conn: conn.close()

@app.route('/api/get-comprehension-submissions')
@teacher_required
def api_comprehension_submissions():
    if not session.get('is_teacher'): return jsonify({"error": "unauthorized"}), 401
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT s.id, s.student_id, s.student_answer, s.ai_analysis_json, 
                       s.feedback_korean, s.created_at, 
                       e.korean_dialogue, e.key_points, s.class_name 
                FROM comprehension_submissions s 
                JOIN comprehension_exercises e ON e.id = s.comprehension_exercise_id 
                ORDER BY s.id DESC LIMIT 100
            """)
            rows = cur.fetchall()
        
        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            r['feedback_korean'] = r.get('feedback_korean') or '(피드백 없음)'
            items.append(r)
        return jsonify({"items": items, "quiz_type": "comprehension"})
    finally:
        if conn: conn.close()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)