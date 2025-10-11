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
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
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
                # 1. 기존 테이블 이름 변경 (오류 발생 시에도 계속 진행)
                try:
                    cur.execute("ALTER TABLE exercises RENAME TO translation_exercises;")
                    print("✅ 'exercises' 테이블을 'translation_exercises'로 변경했습니다.")
                except psycopg2.Error as e:
                    print(f"ℹ️ 'exercises' 테이블 이름 변경 건너뛰기: {e}")
                    conn.rollback() # 트랜잭션 리셋
                
                try:
                    cur.execute("ALTER TABLE submissions RENAME TO translation_submissions;")
                    print("✅ 'submissions' 테이블을 'translation_submissions'로 변경했습니다.")
                except psycopg2.Error as e:
                    print(f"ℹ️ 'submissions' 테이블 이름 변경 건너뛰기: {e}")
                    conn.rollback() # 트랜잭션 리셋

                # 2. '번역 퀴즈' 관련 테이블 생성 및 보강
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

                # 3. '이해력 퀴즈' 관련 테이블 생성
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
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
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

# --- 이해력(Comprehension) 퀴즈용 채점 프롬프트 (블루프린트 부록 반영) ---
COMPREHENSION_EVALUATION_PROMPT = """
You are an expert AI assistant specializing in Korean language education for Italian students. Your mission is to evaluate how well a student has understood a Korean dialogue based on specific scoring criteria (`key_points`) set by the professor.

[Input Information]
- Student's Italian Answer: "{student_answer}"
- Professor's Scoring Criteria (key_points): {key_points_json}

[Evaluation Guidelines]
1. **Vocabulary Assessment (1단계):** Check if the student's answer includes the Italian equivalents (or valid synonyms) of the words in `target_vocabulary` from `key_points`. Award basic points based on vocabulary usage.

2. **Contextual Assessment (2단계):** Evaluate if the overall meaning of the student's answer aligns with the core ideas described in `meaning_points` from `key_points`. Award additional points or deduct based on meaning accuracy.

3. **Core Scoring Principles (핵심 평가 원칙):**
   - **Synonyms (유의어):** If the student uses valid synonyms not present in `target_vocabulary`, and the context is correct, award high scores. Mention the original target vocabulary in `feedback`.
   - **Context Drift (문맥 이탈):** If the student uses key vocabulary but writes content unrelated to `meaning_points`, award low scores and guide them in `feedback`.
   - **Subject/Object Confusion (주체/객체 혼동):** Confusing the subject or object is a critical error. Award very low scores.
   - **Over-Inference (과잉 추론):** If the answer includes facts not present in the original dialogue (student's inference), consider it a failure to summarize key points. Award low scores.
   - **Sentence Structure Variation (문장 구조 변형):** If grammatical structure differs (e.g., active to passive) but meaning is perfectly preserved, full marks can be awarded.

4. **Scoring:** Synthesize the above assessments to assign a score out of 10.0 (e.g., 9.5, 8.0, 7.5). The score MUST have one decimal place.

5. **Output Format:** Your response MUST be ONLY a single JSON object. Do NOT add any explanatory text before or after the JSON.

[Required JSON Output Format]
```json
{{
  "score": 8.5,
  "evaluation": "(한국어로) 핵심 어휘 '복잡하다(difficile)'와 '찾다(trovare)' 사용. 핵심 의미 '키아라가 지하철역을 복잡하게 생각함'을 정확히 파악. 높은 점수 부여.",
  "feedback": "(이탈리아어로) Ottima comprensione! Hai capito il punto chiave della conversazione. Per una risposta perfetta, prova a usare il vocabolario target come 'stazione della metropolitana'. Continua così!"
}}

Important Notes:
score: A number (float) out of 10.0, with one decimal place.
evaluation: (In Korean) An objective summary of the scoring process for the professor's review, based strictly on key_points.
feedback: (In Italian) Encouraging and constructive feedback for the student.
"""


@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')
    class_name = data.get('class_name')
    quiz_type = data.get('quiz_type')  # ★★★ [핵심 추가] quiz_type 받기

    # ★★★ [핵심 수정] quiz_type도 필수 정보로 확인
    if not all([student_id, student_answer, exercise_id, class_name, quiz_type]):
        return jsonify({"error": "필수 정보 누락 (퀴즈 유형 포함)"}), 400

    conn = None
    try:
        conn = get_db_connection()
        if conn is None: return jsonify({"error": "DB 연결 실패"}), 500
        if not model: return jsonify({"error": "AI 모델 미설정"}), 500

        # ★★★ [핵심 분기] quiz_type에 따라 다른 테이블 조회 및 저장
        with conn.cursor() as cur:
            if quiz_type == 'translation':
                # 번역 퀴즈: translation_exercises에서 원문 조회
                cur.execute("SELECT korean_sentence FROM translation_exercises WHERE id = %s;", (exercise_id,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "문제 ID 없음"}), 404
                korean_question = row[0]

                # AI 채점 (번역용 프롬프트)
                prompt_text = EVALUATION_PROMPT.format(Korean_Question=korean_question, Student_Answer=student_answer)
                response = model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text
                ai_result = json.loads(json_str)
                
                score_raw = ai_result.get('score')
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw else None
                analysis = ai_result.get('analysis', {})
                
                # translation_submissions 테이블에 저장
                cur.execute(
                    "INSERT INTO translation_submissions (exercise_id, student_id, student_answer, score, ai_analysis_json, class_name) VALUES (%s, %s, %s, %s, %s, %s)",
                    (exercise_id, student_id, student_answer, score, psycopg2.extras.Json(analysis, dumps=lambda x: json.dumps(x, ensure_ascii=False)), class_name)
                )
                
            elif quiz_type == 'comprehension':
                # 이해력 퀴즈: comprehension_exercises에서 대화문과 key_points 조회
                cur.execute("SELECT korean_dialogue, key_points FROM comprehension_exercises WHERE id = %s;", (exercise_id,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "문제 ID 없음"}), 404
                korean_dialogue, key_points = row[0], row[1]

                # AI 채점 (이해력용 프롬프트 사용)
                prompt_text = COMPREHENSION_EVALUATION_PROMPT.format(student_answer=student_answer, key_points_json=json.dumps(key_points, ensure_ascii=False))
                response = model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text
                ai_result = json.loads(json_str)
                
                score_raw = ai_result.get('score')
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw else None
                
                # comprehension_submissions 테이블에 저장 (ai_analysis_json에 전체 결과 저장)
                cur.execute(
                    "INSERT INTO comprehension_submissions (comprehension_exercise_id, student_id, student_answer, ai_analysis_json, class_name) VALUES (%s, %s, %s, %s, %s)",
                    (exercise_id, student_id, student_answer, psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False)), class_name)
                )
            else:
                return jsonify({"error": "잘못된 퀴즈 유형"}), 400

            conn.commit()

        # ★★★ [핵심 추가] 교수님께서 정해주신 5단계 평가 기준 적용
        def get_rating_details(score):
            score = float(score) if score else 0
            if score >= 8.6: return {"category": "Eccellente", "color": "teal"}
            if score >= 7.1: return {"category": "Buono", "color": "lightgreen"}
            if score >= 5.6: return {"category": "Sufficiente", "color": "gold"}
            if score >= 4.1: return {"category": "Da migliorare", "color": "orange"}
            return {"category": "Riprova", "color": "red"}

        rating_info = get_rating_details(score)

        # 학생에게 보낼 피드백 추출
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
            "feedback": student_feedback
        })    

    except Exception as e:
        print(f"🚨 /api/submit-answer 오류: {e}")
        traceback.print_exc()
        if conn: conn.rollback()
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
    quiz_type = request.args.get('quiz_type')
    
    if not class_name or not quiz_type:
        # 필수 정보가 없으면 로그인 페이지로 돌려보냅니다.
        return redirect(url_for('login'))

    conn = get_db_connection()
    if not conn:
        return "데이터베이스 연결에 실패했습니다.", 500
        
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if quiz_type == 'translation':
                # 번역 퀴즈 문제 목록을 불러옵니다.
                cur.execute("SELECT id, korean_sentence AS question_text FROM translation_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            elif quiz_type == 'comprehension':
                # 이해력 퀴즈 문제 목록을 불러옵니다.
                cur.execute("SELECT id, korean_dialogue AS question_text FROM comprehension_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            else:
                # 잘못된 퀴즈 유형일 경우 에러를 표시합니다.
                return "잘못된 퀴즈 유형입니다.", 400
            
            exercises = cur.fetchall()
        
        # 퀴즈 유형(quiz_type)을 HTML 템플릿으로 함께 전달합니다.
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