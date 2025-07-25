import os
import google.generativeai as genai
from flask import Flask, render_template, jsonify, request # 'request'를 새로 추가했습니다.
import psycopg2
import psycopg2.extras
import json # AI의 응답(문자열)을 JSON으로 다루기 위해 추가했습니다.

# --- 기본 설정 ---
app = Flask(__name__)

# --- AI 모델 설정 ---
try:
    api_key = os.environ.get('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
except Exception as e:
    model = None
    print(f"🚨 Gemini AI 모델 설정 오류: {e}")

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

# --- 최종 프롬프트 (교수님 피드백 반영 v1.1) ---
EVALUATION_PROMPT = """
당신은 이탈리아 학생에게 한국어를 가르치는, 매우 엄격하고 공정한 AI 언어 교사입니다.
당신의 임무는, 주어진 한국어 원문과 학생이 제출한 이탈리아어 번역 답안을 비교하여, 학생의 이해도를 10.0점 만점으로 채점하고 심층적인 분석을 제공하는 것입니다.

[채점 기준]
- 의미의 정확성: 단순 직역이 아닌, 문맥적 의미와 뉘앙스를 얼마나 잘 살렸는지가 가장 중요합니다.
- 문법 및 어휘: 약간의 문법적 오류나 더 나은 단어 선택이 가능했다면 점수를 미세하게 조정하세요.
- 점수는 반드시 0.0에서 10.0 사이의 숫자여야 하며, 소수점 첫째 자리까지 표현해야 합니다.

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[출력 형식]
절대로, 무슨 일이 있어도 다른 설명 없이 오직 아래 규칙을 따르는 JSON 형식으로만 응답해야 합니다.

{
  "score": "학생에게 보여줄 10.0 만점의 채점 점수 (숫자 형식)",
  "analysis": {
    "original_korean_question": "채점의 기준이 된 한국어 원문",
    "student_answer_original": "학생이 제출한 이탈리아어 답안 원문",
    "student_answer_korean_translation": "학생의 답안을 자연스러운 한국어로 번역한 문장",
    "score": "채점 점수 (위의 score와 동일한 값)",
    "key_phrases_italian": "학생 답안에서 핵심이 되는 이탈리아어 어휘나 관용구 2~3개를 담은 리스트(배열)",
    "key_phrases_korean_translation": "위에서 추출한 이탈리아어 어휘/관용구의 한국어 뜻풀이를 담은 리스트(배열)"
  }
}
"""

# --- 웹 페이지 라우트 ---
@app.route('/')
def home():
    # 나중에 이 부분에서 퀴즈 문제를 DB에서 불러와서 HTML에 전달하게 됩니다.
    return render_template('index.html')

# (★★★ 이 부분이 우리 프로젝트의 심장입니다 ★★★)
# --- API: 학생 답안 제출 및 채점 처리 ---
@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    # 1. 웹페이지로부터 학생 정보와 답안을 받습니다.
    data = request.get_json()
    student_id = data.get('student_id')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id') # 어떤 문제에 대한 답인지 ID를 받습니다.

    # 필수 정보가 없는 경우 오류를 반환합니다.
    if not all([student_id, student_answer, exercise_id]):
        return jsonify({"error": "필수 정보(학생 ID, 답안, 문제 ID)가 누락되었습니다."}), 400

    conn = None
    try:
        # 2. 데이터베이스에서 채점의 기준이 될 '원본 한국어 문장'을 가져옵니다.
        conn = get_db_connection()
        korean_question = ""
        with conn.cursor() as cur:
            cur.execute("SELECT korean_sentence FROM exercises WHERE id = %s;", (exercise_id,))
            result = cur.fetchone()
            if result:
                korean_question = result[0]
            else:
                return jsonify({"error": "해당 ID의 문제를 찾을 수 없습니다."}), 404

        # 3. AI에게 보낼 프롬프트를 완성합니다.
        prompt_text = EVALUATION_PROMPT.format(
            Korean_Question=korean_question,
            Student_Answer=student_answer
        )

        # 4. AI를 호출하여 채점 및 분석을 요청합니다.
        if not model:
            return jsonify({"error": "AI 모델이 설정되지 않았습니다."}), 500
        
        response = model.generate_content(prompt_text)
        
        # AI 응답(텍스트)에서 순수 JSON 부분만 추출하고 파싱합니다.
        # AI가 가끔 ```json ... ``` 같은 마크다운을 포함할 때가 있어 안전장치를 추가합니다.
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        ai_result = json.loads(cleaned_text)
        
        score = ai_result.get('score')
        analysis = ai_result.get('analysis')

        # 5. 채점 결과를 'submissions' 테이블에 저장합니다.
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submissions (exercise_id, student_id, student_answer, score, ai_analysis_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (exercise_id, student_id, student_answer, score, json.dumps(analysis))
            )
            conn.commit()
        
        # 6. 학생의 웹페이지에는 '점수'만 간단히 보내줍니다.
        return jsonify({"success": True, "score": score})

    except Exception as e:
        # 어떤 단계에서든 오류가 발생하면 서버 로그에 기록하고 에러 메시지를 반환합니다.
        print(f"🚨 /api/submit-answer 처리 중 오류 발생: {e}")
        return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500
    finally:
        # 모든 작업이 끝나면 데이터베이스 연결을 반드시 닫습니다.
        if conn:
            conn.close()