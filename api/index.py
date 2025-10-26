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
import requests
import hashlib
from datetime import datetime

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
                
                # 1. 말하기 문제 테이블 (Speaking Exercises)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS speaking_exercises (
                        id SERIAL PRIMARY KEY,
                        class_name VARCHAR(50) NOT NULL,
                        situation_description TEXT NOT NULL,
                        required_expression TEXT NOT NULL,
                        expected_korean_answer TEXT NOT NULL,
                        target_vocabulary JSONB NOT NULL,
                        teacher_criterion TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 2. 말하기 제출 테이블 (Speaking Submissions)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS speaking_submissions (
                        id SERIAL PRIMARY KEY,
                        exercise_id INTEGER REFERENCES speaking_exercises(id) ON DELETE SET NULL,
                        class_name VARCHAR(50) NOT NULL,
                        student_id VARCHAR(100) NOT NULL,
                        audio_file_url TEXT NOT NULL,
                        recognized_korean_text TEXT,
                        ai_analysis_json JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(student_id, exercise_id)
                    );
                """)

                print("✅ 말하기 퀴즈 테이블(speaking_exercises, speaking_submissions)이 생성되었습니다.")

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

EVALUATION_PROMPT = """
너는 한국어와 이탈리아어에 모두 능통한 언어 평가 전문가이다. 너의 유일한 임무는 '한국어 원문'을 들은 학생이 작성한 '이탈리아어 답안'이 원문의 의미를 얼마나 정확하게 이해하고 반영했는지를 평가하는 것이다.

[입력 정보]
- 한국어 원문: "{Korean_Question}"
- 학생의 이탈리아어 답안: "{Student_Answer}"

[핵심 원칙]
이것은 이탈리아어 작문 시험이 아니다. 학생의 이탈리아어 문법이 다소 어색하거나 사소한 오류가 있더라도, 원문의 의미를 이해했다고 판단되면 절대 감점하지 마라. 평가는 오직 '의미의 정확성' 하나만을 기준으로 한다.

[채점 기준: 의미의 정확성 (Semantic Accuracy) - 100%]

1. **시작 점수: 10.0점**

2. **점수는 반드시 소수점 첫째 자리까지 평가해야 한다 (예: 9.6, 8.1, 7.3).**
   정수(7, 8, 9)로만 점수를 매기는 것은 허용되지 않는다.

3. **AI 자율성:**
   각 감점 범위 내에서 ±0.5점 조정이 가능하다.
   오류의 심각도, 문장 복잡도, 맥락을 고려하여 판단한다.

4. **번역의 핵심 원칙:**
   - 이것은 번역 수업이다. 학생은 원문에 있는 내용만 번역해야 한다.
   - 직역과 의역 모두 허용되나, 원문의 의미를 정확히 전달해야 한다.
   - 의역이 한국어 표현 구조가 다르다는 이유만으로 감점하지 않는다.

5. 아래 기준에 따라 오류를 발견할 때마다 점수를 차감한다:

---

[Level 1] 완전한 오역 또는 의미 왜곡 (Critical)
감점: -6.5 ~ -7.5점

• 원문의 핵심 의미를 완전히 잘못 이해하여 정반대의 의미나 
  전혀 다른 의미로 번역한 경우.

• 예시:
  - 방향/상태 정반대: "학교에 갔다" → "È tornato da scuola"
  - 부정/긍정 혼동: "좋아한다" → "Non mi piace"
  - 주체 완전 오인: "동생이 간다" → "Vado io"
  - 시제 완전 오류: "갔다" → "Andrò" (과거 → 미래)

**중요 - 시제 완전 오류 기준:**
- **Level 1 (완전한 오류):** 과거 ↔ 미래 (사건 발생 여부가 정반대)
  예: "갔다" (완료) → "Andrà" (미완료)
  
- **Level 4 (사소한 불일치):** 현재 ↔ 미래, 현재 ↔ 진행형
  예: "갈 거다" → "Va" (둘 다 "가는 행위", 시점만 다름)
  예: "모일 거예요" → "Si riuniscono" (둘 다 "모이는 행위", 시점만 다름)

**절대 금지:**
- "현재 → 미래" 또는 "미래 → 현재"를 Level 1으로 감점하지 마라!
- 이는 반드시 Level 4 (-0.3 ~ -0.7점)이다!
---

[Level 2] 핵심 정보 누락/오류 (Major)
감점: -3.0 ~ -3.5점

• 문장의 주어, 목적어, 동사, 장소, 시간 등 핵심적인 구성 요소나 
  정보를 빠뜨리거나 틀리게 번역한 경우.

• 예시:
  - 주어 누락: "동생이 간다" → "Va" (누가?)
  - 목적어 누락: "영화를 봤다" → "Ho visto" (뭘?)
  - 장소 누락: "서울에 갔다" → "Sono andato" (어디로?)
  - 시간 누락: "어제 갔다" → "Sono andato" (언제?)
  - 핵심 동사 오역: "공부한다" → "Lavoro"

---

[Level 3] 원문에 없는 정보 추가 (과잉 추론)
감점: -0.5 ~ -3.5점 (정도에 따라)

• 번역이 아닌 학생의 추론이나 창작으로 원문에 없는 정보를 추가한 경우.

3-1. 사소한 추론 (-0.5 ~ -1.0점)
     - 맥락상 자연스럽지만 원문에는 없는 사소한 추가
     - 예: "공부한다" → "Studia con attenzione"

3-2. 중간 추론 (-1.5 ~ -2.5점)
     - 목적이나 이유를 추가하여 의미를 확장한 경우
     - 예: "도서관에 간다" → "Vado in biblioteca per studiare"

3-3. 심각한 추론 (-3.0 ~ -3.5점)
     - 원문과 무관한 구체적 정보를 창작한 경우
     - 예: "집에 있다" → "È a casa perché è malato e ha la febbre alta"

---

[Level 4] 사소한 의미 불일치 (Minor)
감점: -0.2 ~ -1.5점

• 전체적인 의미는 맞지만, 특정 단어나 표현의 뉘앙스를 
  잘못 이해하여 약간의 의미 차이가 발생한 경우.

4-1. 시제 뉘앙스 차이 (-0.3 ~ -0.7점)
     **중요:** 핵심 행위/상태는 같고 시점만 다른 경우
     
     예시:
     - 현재 ↔ 미래: "갈 거다" → "Va" 또는 "간다" → "Andrà"
       → 둘 다 "가는 행위"를 설명, 시점만 다름 (-0.5점)
     
     - 진행형 누락: "먹고 있다" → "Mangio"
       → 지속성 표현 누락 (-0.3점)
     
     **판단 기준:**
     ✓ 행위/상태의 본질이 동일한가? → YES면 Level 4
     ✓ 사건의 발생 여부가 반대인가? → YES면 Level 1

4-2. 강도/정도 표현 누락 (-0.2 ~ -0.7점)
     - "많이", "조금", "매우" 등의 부사 누락
     - 예: "비가 많이 온다" → "Piove"
     - 예: "아주 예쁘다" → "È bella"

4-3. 관형어/수식어 누락 (-0.5 ~ -1.5점)
     - 예: "예쁜 꽃" → "Fiore"
     - 예: "큰 집" → "Casa"

4-4. 복수/단수 혼동 (-0.3 ~ -0.8점)
     - 예: "친구들" → "amico"
     - 예: "책" → "libri"

---

[Level 5] 허용 가능한 추가 정보 및 표현 차이
감점: 없음 (10.0점 유지)

• 다음 경우는 번역 과정에서 자연스럽게 발생할 수 있으므로 
  절대 감점하지 않는다:

5-1. 문법상 자연스러운 추가
     - 부사/형용사 추가 (강도 표현): "Piove molto"
     - 관사 추가: "il libro", "la casa"
     - 대명사 강조: "Lui è a casa"
     - 시제 자연스러운 변형: "Sta piovendo" (진행형)

5-2. 자연스러운 의역
     - 예: "표를 끊다" → "comprare i biglietti"
     - 예: "날씨가 좋다" → "Che bella giornata!"
     - 조건: 원문의 모든 핵심 정보 포함 + 추가/삭제 없음
     - 판정: 10.0점 유지

5-3. 동사 선택의 뉘앙스 차이 (의미는 정확)
     - 예: "집에 있다" → "Si trova a casa" (È a casa가 더 정확)
     - 예: "공부한다" → "Fa lo studio" (Studia가 더 정확)
     - 판정: 10.0점 유지
     - [교사용 참고]로 더 자연스러운 표현 제시

---

[의역(Paraphrase)에 대한 특별 지침]

의역은 번역의 자연스러운 과정이지만, 다음 원칙을 지켜야 한다:

• 허용되는 의역 (감점 없음):
  - 관용구의 자연스러운 번역
  - 문화적 표현의 적절한 전환
  - 동사 선택의 자연스러운 변형
  - 조건: 원문의 모든 핵심 정보 포함 + 추가/삭제 없음

• 감점되는 의역:
  - 원문에 없는 강도/정도 추가 → Level 3 (과잉 추론)
  - 의미 축소/확대 → Level 4 (사소한 불일치)
  - 핵심 정보 누락 → Level 2 (핵심 누락)

• 판단 체크리스트 (의역 평가 시 반드시 확인):
  ✓ 원문의 모든 핵심 정보가 포함되었는가?
  ✓ 원문에 없는 정보를 추가하지 않았는가?
  ✓ 의미의 강도/정도가 유지되는가?
  
  → 모두 YES → 의역 허용 (10.0점)
  → 하나라도 NO → 해당 레벨로 감점

• 핵심: 직역과 의역 모두 원문의 의미를 정확히 전달하면 동등하게 평가한다.

---

[뉘앙스 및 격식 (Nuance & Formality)]

• 이것은 절대 감점 요인이 아니다.

• 다음 차이는 '오류'로 간주하지 않으며, 절대로 감점의 근거가 될 수 없다:
  - 존댓말/반말 처리
  - 어조 차이
  - 단어 선택의 미묘한 차이
  - 격식체/비격식체

• 다만, 이러한 차이점이 교육적으로 의미가 있다고 판단될 경우,
  반드시 'evaluation_feedback'에 [교사용 참고] 태그를 사용하여
  그 차이점만 객관적으로 서술한다.

---

[누적 감점 및 최종 점수]
• 여러 오류가 발견될 경우 감점을 누적한다.
• 누적 감점이 10.0점을 초과하면 최종 점수는 0.0점으로 처리한다.
• 최종 점수는 반드시 0.0 ~ 10.0 사이여야 한다.
• 점수는 반드시 소수점 첫째 자리까지 표기한다 (예: 7.5, 8.3, 9.1).

---

[학생용 힌트 생성 규칙]

• "student_hint" 필드는 반드시 다음 규칙을 따라 생성해야 한다:

1. **Level 4, 5 (사소한 오류 또는 오류 없음):**
   - student_hint: "" (빈 문자열)
   - 학생에게 피드백을 보여주지 않는다.

2. **Level 1, 2, 3 (심각한 오류):**
   - student_hint: "한 문장으로 핵심 오류만 지적"
   - **반드시 이탈리아어로만 작성**
   - 최대 1문장, 20단어 이내
   - 친절한 설명 없이, 오류의 종류만 간단히 힌트
   
3. **힌트 작성 예시 (모두 이탈리아어):**
   - **완전한 오역**: "Hai tradotto il contrario del significato originale."
   - **시제가 정반대**: "Il tempo verbale è opposto: passato ≠ futuro."
   - **주어 누락**: "Manca il soggetto della frase."
   - **목적어 누락**: "Manca l'oggetto principale."
   - **장소 누락**: "Manca l'informazione del luogo."
   - **시간 누락**: "Manca l'informazione temporale."
   - **핵심 동사 오역**: "Il verbo principale è stato tradotto in modo errato."
   - **원문에 없는 정보 추가 (사소한)**: "Hai aggiunto dettagli non presenti nel testo."
   - **원문에 없는 정보 추가 (심각한)**: "Hai inventato informazioni che non esistono nell'originale."

4. **절대 금지 사항:**
   - 정답을 직접 제시하지 마라
   - 격려나 칭찬 문구를 포함하지 마라
   - 설명을 길게 늘리지 마라
   - 단순히 "오류가 있습니다"라고만 하지 마라 (구체적이어야 함)
   - **한국어나 영어를 절대 사용하지 마라 (100% 이탈리아어)**

5. **student_hint는 반드시 이탈리아어로 작성한다.**

---

[출력 형식]
JSON ONLY. 다른 설명 없이 JSON 객체만 반환해야 합니다.
{{
  "score": "9.5, 8.0, 7.5 등과 같은 10.0 형식의 숫자 문자열",
  "student_hint": "학생용 힌트 (Level 1, 2, 3일 때만, 이탈리아어)",
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

SPEAKING_EVALUATION_PROMPT = """
너는 한국어 말하기 교육 전문 AI이다. 이탈리아 학생이 특정 상황에서 한국어로 말한 음성을 평가한다.

[입력 정보]
- **상황 설명 (이탈리아어):** "{situation_description}"
- **학생이 해야 할 말 (이탈리아어):** "{required_expression}"
- **예상 정답 (한국어):** "{expected_korean_answer}"
- **목표 어휘:** {target_vocabulary_json}
- **교수님 추가 기준:** "{teacher_criterion}"

---

[절대 규칙: 음성 인식]

**문맥 보정 금지!**
- 학생이 발음한 소리를 **있는 그대로** 텍스트로 변환하라.
- 문맥상 이상하더라도 절대 자동 수정하지 마라.
- 예시:
  * 학생 발음: "그 남자 맛있다" → 인식: "그 남자 맛있다" (✅)
  * 학생 발음: "그 남자 맛있다" → 인식: "그 남자 멋있다" (❌ 절대 금지!)

- 단, `evaluation` 필드에서 오류를 명확히 지적하라:
  "학생이 '맛있다'라고 발음했으나, 문맥상 '멋있다'가 정확한 표현임. 발음 혼동으로 -1.5점 감점."

---

[채점 기준 - 총 10.0점]

**[1순위] 어휘 적합성 (50% = 5.0점)**

**평가 항목:**
1. **목표 어휘 사용 (3.0점)**
   - 계산: (사용한 목표 어휘 수 / 전체 목표 어휘 수) × 3.0
   - 유의어 허용 기준:
     * 교수님 기준(`teacher_criterion`)이 있으면 우선 적용
     * 없으면: 맥락에 자연스러운 유의어만 인정
     * 예: "구입하다" → "사다" (일반 상황: OK)
     * 예: "터지다" → "폭파되다" (부적절: 감점)
     * 예: "쓰여 있다" → "쓰인" (OK, 단 교수님 기준 참고)

2. **맥락 적합성 (2.0점)**
   - 상황 설명에 부합하는 어휘 선택인가?
   - 높임법/격식이 상황에 맞는가?
   - 감점 기준:
     * 상황과 완전 불일치: -1.5 ~ -2.0점
     * 높임법 오류 (필수 상황): -1.0 ~ -1.5점
     * 약간 어색한 선택: -0.3 ~ -0.8점

**[2순위] 문법 정확성 (30% = 3.0점)**

**한국인의 이해를 방해하는 문법 오류 집중 평가:**

**감점 기준:**
1. **피동/사동 오류 (심각):** -1.0 ~ -1.5점
   - 예: "문이 닫**았**어요" (X) → "문이 닫**혔**어요" (O)
   - 예: "아기를 자**요**" (X) → "아기를 재**워**요" (O)

2. **조사 오류 (심각):** -0.8 ~ -1.2점
   - 예: "그 사람**이** 갈게요" (X) → "제**가**/저**가** 갈게요" (O)
   - 예: "학교**를** 가요" (X) → "학교**에** 가요" (O)

3. **불규칙 활용 오류:** -0.5 ~ -1.0점
   - 예: "덥**어**요" (X) → "더**워**요" (O)
   - 예: "쉽**어**요" (X) → "쉬**워**요" (O)

4. **시제/연결 오류:** -0.3 ~ -0.8점
   - 예: "어제 가**요**" (X) → "어제 갔**어요**" (O)

**[3순위] 발음 명료도 (20% = 2.0점)**

**원칙: 사소한 발음 차이는 감점 최소화. 심각한 오류만 지적.**

**감점 기준:**
1. **의미 혼동 발음 (심각):** -1.0 ~ -1.5점
   - 예: "멋있다" → "맛있다" (완전히 다른 의미)
   - 예: "사과" → "사고" (의미 왜곡)

2. **중간 수준 오류:** -0.3 ~ -0.7점
   - 예: 경음화 오류: "사랑해요" → "싸랑해요"
   - 예: 자음 혼동: "자다" → "차다"

3. **사소한 발음 (피드백만, 감점 없음):**
   - 예: "ㅈ/ㅊ" 미세 차이
   - 예: 억양의 부자연스러움
   - → `feedback`에만 언급 ("Fai attenzione alla differenza tra ㅈ e ㅊ")

4. **극심한 발음 오류 (희귀):** -1.5 ~ -2.0점
   - 예: "농협은행" → "너며쁘네" (완전 불일치)

---

[출력 형식 - JSON Only]

{{
  "recognized_text": "학생이 실제 발음한 한국어 텍스트 (문맥 보정 없이 그대로!)",
  "score": 8.5,
  "vocabulary_usage": {{
    "쓰여 있다": {{
      "used": true,
      "actual_form": "쓰인",
      "is_synonym": true,
      "note": "교수님 기준에 따라 허용. '쓰여 있다' 권장 피드백 제공."
    }},
    "방향": {{
      "used": true,
      "note": "정확한 사용"
    }},
    "-는지": {{
      "used": false,
      "note": "문법 항목 누락"
    }}
  }},
  "grammar_errors": [
    {{
      "type": "불규칙 활용",
      "student_said": "덥어요",
      "correct_form": "더워요",
      "deduction": -0.8
    }}
  ],
  "pronunciation_issues": [
    {{
      "severity": "심각",
      "student_said": "맛있다",
      "intended": "멋있다",
      "note": "의미 혼동 발생",
      "deduction": -1.5
    }},
    {{
      "severity": "사소함",
      "issue": "ㅈ/ㅊ 구분 미흡",
      "note": "이해에 지장 없음, 피드백만 제공",
      "deduction": 0
    }}
  ],
  "evaluation": "(한국어) 상세 채점 근거.
  - 어휘: 목표 어휘 2/3 사용 (2.0/3.0점). '쓰인' 사용은 허용되나 '쓰여 있다' 권장.
  - 문법: 불규칙 활용 오류 1건 (-0.8점). 2.2/3.0점.
  - 발음: '맛있다'/'멋있다' 혼동 (-1.5점). 0.5/2.0점.
  - 총점: 4.7/10.0점.",
  
  "feedback": "(이탈리아어) Hai usato bene alcuni vocaboli, ma c'è un errore di pronuncia importante: hai detto '맛있다' (delizioso) invece di '멋있다' (bello). Fai attenzione! Inoltre, ricorda la coniugazione irregolare di '덥다' → '더워요'."
}}

**중요:**
- `recognized_text`는 문맥 보정 없이 학생의 실제 발음 그대로!
- `grammar_errors`와 `pronunciation_issues`는 구체적 오류 목록
- `evaluation`은 교수님용 한국어 상세 분석
- `feedback`은 학생용 이탈리아어 피드백 (건설적이고 격려적으로)
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
                
                cur.execute(
                    """INSERT INTO comprehension_submissions 
                       (comprehension_exercise_id, student_id, student_answer, ai_analysis_json, class_name) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    (exercise_id, student_id, student_answer, 
                     psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False)), 
                     class_name)
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
            student_hint = ai_result.get('student_hint', '')  # ★★★ 추가 ★★★
            student_feedback = analysis.get('evaluation_feedback', 'Nessun feedback disponibile.')
        elif quiz_type == 'comprehension':
            student_hint = ''  # ★★★ 추가 필수 ★★★
            student_feedback = ai_result.get('feedback', 'Nessun feedback disponibile.')
        else:
            student_hint = ''  # ★★★ 추가 필수 ★★★
            student_feedback = 'Feedback non disponibile.'

        return jsonify({
            "success": True, 
            "score": score,
            "rating_category": rating_info["category"],
            "rating_color": rating_info["color"],
            "student_hint": student_hint,  # ★★★ 추가 ★★★
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

@app.route('/api/submit-speaking-answer', methods=['POST'])
def submit_speaking_answer():
    """말하기 퀴즈 전용 제출 엔드포인트"""
    
    # 1. 폼 데이터 수신
    student_id = request.form.get('student_id')
    exercise_id = request.form.get('exercise_id')
    class_name = request.form.get('class_name')
    quiz_type = request.form.get('quiz_type')
    audio_file = request.files.get('audio_file')
    
    if not all([student_id, exercise_id, class_name, quiz_type, audio_file]):
        return jsonify({"error": "필수 정보 누락"}), 400
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "DB 연결 실패"}), 500
        
        with conn.cursor() as cur:
            # 2. 1회 제출 제한 체크
            cur.execute(
                "SELECT id FROM speaking_submissions WHERE student_id = %s AND exercise_id = %s",
                (student_id, exercise_id)
            )
            if cur.fetchone():
                return jsonify({"error": "이미 제출하셨습니다.", "already_submitted": True}), 400
            
            # 3. 문제 정보 조회
            cur.execute("""
                SELECT situation_description, required_expression, expected_korean_answer, 
                       target_vocabulary, teacher_criterion 
                FROM speaking_exercises 
                WHERE id = %s
            """, (exercise_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "문제 ID 없음"}), 404
            
            situation_desc, required_expr, expected_ans, target_vocab, teacher_crit = row
            
            # 음성 파일을 Gemini에 업로드
            audio_bytes = audio_file.read()

            # 4. Vercel Blob에 음성 파일 업로드
            BLOB_TOKEN = os.environ.get('BLOB_READ_WRITE_TOKEN')
            if not BLOB_TOKEN:
                print("🚨 BLOB_READ_WRITE_TOKEN 환경변수 미설정")
                return jsonify({"error": "Blob storage 미설정"}), 500

            # 파일명 생성 (중복 방지)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(f"{student_id}_{exercise_id}_{timestamp}".encode()).hexdigest()[:8]
            filename = f"speaking/{class_name}/{student_id}_{exercise_id}_{file_hash}.webm"

            # Vercel Blob API 호출
            try:
                print(f"📤 Blob 업로드 시작: {filename}")
                
                upload_response = requests.put(
                    f"https://blob.vercel-storage.com/{filename}",
                    headers={
                        "Authorization": f"Bearer {BLOB_TOKEN}",
                        "Content-Type": "audio/webm",
                        "x-vercel-blob-add-random-suffix": "1"
                    },
                    data=audio_bytes
                )
                
                if upload_response.status_code not in [200, 201]:
                    print(f"🚨 Blob 업로드 실패: {upload_response.status_code}")
                    print(f"응답: {upload_response.text}")
                    return jsonify({"error": "음성 파일 업로드 실패"}), 500
                
                blob_response = upload_response.json()
                audio_url = blob_response.get('url')
                
                if not audio_url:
                    print(f"🚨 URL 없음: {blob_response}")
                    return jsonify({"error": "파일 URL 생성 실패"}), 500
                
                print(f"✅ Blob 업로드 성공: {audio_url}")
                
            except Exception as e:
                print(f"🚨 Blob 업로드 오류: {e}")
                traceback.print_exc()
                return jsonify({"error": f"파일 저장 실패: {str(e)}"}), 500            

            # 5. Gemini API 호출 (음성 → 텍스트 → 평가)
            if not pro_model:
                return jsonify({"error": "AI 모델 미설정"}), 500
                        
            # Gemini 파일 업로드 (임시 파일로 저장 후 업로드)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            uploaded_audio = genai.upload_file(tmp_file_path, mime_type='audio/webm')
            
            # 프롬프트 생성
            prompt_text = SPEAKING_EVALUATION_PROMPT.format(
                situation_description=situation_desc,
                required_expression=required_expr,
                expected_korean_answer=expected_ans,
                target_vocabulary_json=json.dumps(target_vocab, ensure_ascii=False),
                teacher_criterion=teacher_crit or "자율 판단"
            )
            
            # Gemini 호출
            response = pro_model.generate_content(
                [prompt_text, uploaded_audio],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1  # 문맥 보정 최소화
                }
            )
            
            print(f"🤖 [말하기 퀴즈] gemini-2.5-pro 사용 - 학생: {student_id}")
            
            # 임시 파일 삭제
            import os
            os.unlink(tmp_file_path)
            
            # 응답 파싱
            raw_text = getattr(response, 'text', '').strip()
            json_str = extract_first_json_block(raw_text) or raw_text
            ai_result = json.loads(json_str)
            
            score_raw = ai_result.get('score')
            score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw else None
            recognized_text = ai_result.get('recognized_text', '')
            
            # 6. DB에 저장
            cur.execute("""
                INSERT INTO speaking_submissions 
                (exercise_id, class_name, student_id, audio_file_url, recognized_korean_text, ai_analysis_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                exercise_id, class_name, student_id, audio_url, recognized_text,
                psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False))
            ))
            
            conn.commit()
            
            # 7. 점수 등급 계산
            def get_rating_details(score):
                score = float(score) if score else 0
                if score >= 8.6: return {"category": "Eccellente", "color": "teal"}
                if score >= 7.1: return {"category": "Buono", "color": "lightgreen"}
                if score >= 5.6: return {"category": "Sufficiente", "color": "gold"}
                if score >= 4.1: return {"category": "Da migliorare", "color": "orange"}
                return {"category": "Riprova", "color": "red"}
            
            rating_info = get_rating_details(score)
            
            return jsonify({
                "success": True,
                "score": score,
                "rating_category": rating_info["category"],
                "rating_color": rating_info["color"],
                "feedback": ai_result.get('feedback', 'Nessun feedback disponibile.'),
                "recognized_text": recognized_text
            })
    
    except Exception as e:
        print(f"🚨 /api/submit-speaking-answer 오류: {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return jsonify({"error": "서버 내부 오류"}), 500
    finally:
        if conn:
            conn.close()
        
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
            elif quiz_type == 'speaking':
                cur.execute("""
                    SELECT id, situation_description, required_expression, expected_korean_answer 
                    FROM speaking_exercises 
                    WHERE class_name = %s 
                    ORDER BY id
                """, (class_name,))
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

@app.route('/api/get-submissions')
@teacher_required
def api_get_submissions():
    """페이지네이션 지원 - 특정 페이지의 10개 제출물 반환"""
    if not session.get('is_teacher'): 
        return jsonify({"error": "unauthorized"}), 401
    
    page = int(request.args.get('page', 1))
    quiz_type = request.args.get('quiz_type', 'translation')
    class_name = request.args.get('class_name', 'all')
    
    per_page = 10
    offset = (page - 1) * per_page
    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if quiz_type == 'translation':
                # 전체 개수 조회
                if class_name == 'all':
                    cur.execute("SELECT COUNT(*) as total FROM translation_submissions")
                else:
                    cur.execute("SELECT COUNT(*) as total FROM translation_submissions WHERE class_name = %s", (class_name,))
                total = cur.fetchone()['total']
                
                # 페이지네이션 데이터 조회
                if class_name == 'all':
                    cur.execute("""
                        SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, 
                               s.created_at, e.korean_sentence, s.class_name 
                        FROM translation_submissions s 
                        JOIN translation_exercises e ON e.id = s.exercise_id 
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (per_page, offset))
                else:
                    cur.execute("""
                        SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, 
                               s.created_at, e.korean_sentence, s.class_name 
                        FROM translation_submissions s 
                        JOIN translation_exercises e ON e.id = s.exercise_id 
                        WHERE s.class_name = %s
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (class_name, per_page, offset))
            
            elif quiz_type == 'comprehension':
                # 전체 개수 조회
                if class_name == 'all':
                    cur.execute("SELECT COUNT(*) as total FROM comprehension_submissions")
                else:
                    cur.execute("SELECT COUNT(*) as total FROM comprehension_submissions WHERE class_name = %s", (class_name,))
                total = cur.fetchone()['total']
                
                # 페이지네이션 데이터 조회
                if class_name == 'all':
                    cur.execute("""
                        SELECT s.id, s.student_id, s.student_answer, s.ai_analysis_json, 
                               s.created_at, e.korean_dialogue, e.key_points, s.class_name 
                        FROM comprehension_submissions s 
                        JOIN comprehension_exercises e ON e.id = s.comprehension_exercise_id 
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (per_page, offset))
                else:
                    cur.execute("""
                        SELECT s.id, s.student_id, s.student_answer, s.ai_analysis_json, 
                               s.created_at, e.korean_dialogue, e.key_points, s.class_name 
                        FROM comprehension_submissions s 
                        JOIN comprehension_exercises e ON e.id = s.comprehension_exercise_id 
                        WHERE s.class_name = %s
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (class_name, per_page, offset))

            elif quiz_type == 'speaking':
                # 전체 개수 조회
                if class_name == 'all':
                    cur.execute("SELECT COUNT(*) as total FROM speaking_submissions")
                else:
                    cur.execute("SELECT COUNT(*) as total FROM speaking_submissions WHERE class_name = %s", (class_name,))
                total = cur.fetchone()['total']
                
                # 페이지네이션 데이터 조회
                if class_name == 'all':
                    cur.execute("""
                        SELECT s.id, s.student_id, s.audio_file_url, s.recognized_korean_text, 
                            s.ai_analysis_json, s.created_at, 
                            e.situation_description, e.expected_korean_answer, e.target_vocabulary, s.class_name 
                        FROM speaking_submissions s 
                        JOIN speaking_exercises e ON e.id = s.exercise_id 
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (per_page, offset))
                else:
                    cur.execute("""
                        SELECT s.id, s.student_id, s.audio_file_url, s.recognized_korean_text, 
                            s.ai_analysis_json, s.created_at, 
                            e.situation_description, e.expected_korean_answer, e.target_vocabulary, s.class_name 
                        FROM speaking_submissions s 
                        JOIN speaking_exercises e ON e.id = s.exercise_id 
                        WHERE s.class_name = %s
                        ORDER BY s.id DESC 
                        LIMIT %s OFFSET %s
                    """, (class_name, per_page, offset))

            rows = cur.fetchall()
            
        items = []
        for r in rows:
            r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None
            items.append(r)
        
        total_pages = (total + per_page - 1) // per_page
        
        return jsonify({
            "items": items, 
            "quiz_type": quiz_type,
            "total": total,
            "total_pages": total_pages,
            "current_page": page
        })
    finally:
        if conn: conn.close()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)