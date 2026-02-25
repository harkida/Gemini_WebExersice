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

from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from google.genai import types
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

if api_key:
    try:
        gemini_client = genai.Client(api_key=api_key)
        print("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
        print("   📌 번역 : gemini-3-flash (빠르고 경제적)")
        print("   📌 이해력 : gemini-3.0-pro (정밀한 평가)")
    except Exception as e:
        gemini_client = None
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

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

def get_rating_details(score):
    """프로젝트 전체에서 사용하는 표준화된 점수 평가 함수"""
    try:
        score = float(score)
    except (ValueError, TypeError):
        score = 0.0
    
    if score >= 8.5: return {"category": "Eccellente", "color": "#00cc9f"}
    if score >= 7.0: return {"category": "Buono", "color": "#00cc29"}
    if score >= 5.5: return {"category": "Sufficiente", "color": "#cccc00"}
    if score >= 4.0: return {"category": "Da migliorare", "color": "#cc6400"}
    return {"category": "Riprova", "color": "#cc0000"}

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
[Input Information]
- **Korean Original Sentence:** "{Korean_Question}"
- **Student's Italian Answer:** "{Student_Answer}"
{Dialogue_Context_Section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🎯 CORE PRINCIPLE: Hierarchical Semantic Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is NOT an Italian grammar test. Even if the student's Italian has minor grammatical errors or awkward phrasing, DO NOT deduct points if the meaning of the original Korean sentence is understood.
{Dialogue_Context_Instruction}
**Your task:** Evaluate how accurately the student's Italian answer reflects the meaning of the original Korean sentence using a **hierarchical, stop-at-first-match system**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📊 SCORING STRUCTURE (Total: 10.0 points)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Semantic Accuracy (의미 정확성) - 60% (6.0 points)
2. Vocabulary Coverage (어휘 커버리지) - 30% (3.0 points)
3. Information Coverage (정보 커버리지) - 10% (1.0 points)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🔍 COMPONENT 1: Semantic Accuracy (60% = 6.0 points)
## ⚠️ HIERARCHICAL EVALUATION - STOP AT FIRST MATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**CRITICAL RULE:** Once you find an error at Level A, B, or C, STOP evaluation immediately. Do NOT check lower levels.

---

### **STEP 1: Check Level A (완전한 오역 또는 의미 왜곡)**
Score Range: 0.0 ~ 1.5 points (0% ~ 25%)

**Criteria:**
The student has completely misunderstood the Korean original, resulting in:
 - Direction/state reversal: "갔다" (went) → "È tornato" (came back)
 - Affirmation/negation error: "좋아한다" (like) → "Non mi piace" (don't like)
 - Subject complete error: "동생이 간다" (younger sibling goes) → "Vado io" (I go)
 - Tense complete error (action ↔ non-action): "갔다" (went, completed) → "Andrà" (will go, not yet done)

**Action:**
 - IF Level A error found → Judge severity within 0.0 ~ 1.5 range
 - Assign semantic_accuracy_score between 0.0 ~ 1.5
 - Set evaluation_stopped = "A"
 - STOP evaluation (do NOT check B, C, D)

**Severity judgment within Level A:**
 - 극도로 심각 (Extreme): 0.0 ~ 0.5점 (complete opposite meaning)
 - 심각 (Severe): 0.5 ~ 1.0점 (major misunderstanding)
 - 중간 (Moderate): 1.0 ~ 1.5점 (significant error but some understanding)

**⚠️ EXCEPTION: Logical Equivalence in Conditional Statements**
Check if the student's answer is logically equivalent to the original:
 - "~하지 않으면 X" ≡ "~하면 not X"
 - "~하면 not X" ≡ "~하지 않으면 X"
 - Example: "Se non ricorda → problemi" = "Se ricorda → non ci saranno problemi"

Action:
 - IF logically equivalent → DO NOT count as Level A error
 - Proceed to Level D evaluation (4.5 ~ 6.0 points)
 - Note in evaluation_feedback: "[교사용 참고] 학생이 논리적으로 동치인 조건문 구조를 사용했습니다."

---

### **STEP 2: Check Level B (핵심 정보 누락 또는 오류)**
Score Range: 1.5 ~ 3.0 points (25% ~ 50%)

**Only check this if Level A was NOT found.**

**Criteria:**
The student has omitted or incorrectly translated core information elements:
 - Subject missing: "동생이 간다" → "Va" (subject missing)
 - Object missing: "영화를 봤다" → "Ho visto" (object missing)
 - Place missing: "서울에 갔다" → "Sono andato" (place missing)
 - Time missing: "어제 갔다" → "Sono andato" (time missing)
 - Main verb error: "공부한다" (study) → "Lavoro" (work)

**Action:**
 - IF Level B error found → Judge severity within 1.5 ~ 3.0 range
 - Assign semantic_accuracy_score between 1.5 ~ 3.0
 - Set evaluation_stopped = "B"
 - STOP evaluation (do NOT check C, D)

**Severity judgment within Level B:**
 - 복수 핵심 누락 (Multiple core missing): 1.5 ~ 2.0점
 - 단일 핵심 누락 (Single core missing): 2.0 ~ 2.5점
 - 부가 정보 누락 (Secondary info missing): 2.5 ~ 3.0점

---

### **STEP 3: Check Level C (원문에 없는 정보 추가 - 과잉 추론)**
Score Range: 3.0 ~ 4.5 points (50% ~ 75%)

**Only check this if Level A and B were NOT found.**

**Criteria:**
The student has added information NOT present in the Korean original:
 - Minor inference: "공부한다" → "Studia con attenzione" (added "with attention")
 - Moderate inference: "도서관에 간다" → "Vado in biblioteca per studiare" (added purpose)
 - Major inference: "집에 있다" → "È a casa perché è malato" (invented reason)

{Dialogue_Context_LevelC_Exception}

**Action:**
 - IF Level C error found → Judge severity within 3.0 ~ 4.5 range
 - Assign semantic_accuracy_score between 3.0 ~ 4.5
 - Set evaluation_stopped = "C"
 - STOP evaluation (do NOT check D)

**Severity judgment within Level C:**
 - 심각한 추론 (Serious invention): 3.0 ~ 3.5점
 - 중간 추론 (Moderate addition): 3.5 ~ 4.0점
 - 사소한 추론 (Minor addition): 4.0 ~ 4.5점

---

### **STEP 4: Level D (사소한 의미 불일치 또는 완벽한 번역)**
Score Range: 4.5 ~ 6.0 points (75% ~ 100%)

**Only reach this if Level A, B, C were NOT found.**

**Criteria:**
D-1. Minor semantic inaccuracies (4.5 ~ 6.0점 미만):
 - Tense nuance difference (NOT opposite): "갈 거다" (will go) → "Va" (goes) - same action, different time expression
 - Intensity/degree missing: "비가 많이 온다" → "Piove" (missing "molto")
 - Modifier missing: "예쁜 꽃" → "Fiore" (missing "bello")
 - Singular/plural mix-up: "친구들" → "amico"

D-2. Perfect translation (6.0점):
All core information included, semantically accurate, natural expression.

**ALLOWED without penalty:**
• Adding adverbs (intensity expressions): "Piove molto"
• Article addition: "il libro", "la casa"
• Pronoun emphasis: "Lui è a casa"
• Natural paraphrase: 
 - Example : "comprare i biglietti" for "표를 끊다"
 - Example : "Che bella giornata!" for "날씨가 좋다" 
• Natural tense variation: "Sta piovendo" for "비가 온다"
• Nuance Differences in Verb Choice:
 - Example : "Si trova a casa" for "집에 있다"
 - Example : "Fa lo studio" for "공부한다"


**Action:**
 - Judge quality within 4.5 ~ 6.0 range
 - Assign semantic_accuracy_score between 4.5 ~ 6.0
 - Set evaluation_stopped = null (evaluation completed)
 - Perfect translation = 6.0점

[NUANCE AND FORMALITY]

• This is ABSOLUTELY NOT a deduction factor.

• The following differences are NOT considered 'errors' and can NEVER be grounds for deduction:
 - Formal/informal speech handling
 - Tone differences
 - Subtle differences in word choice
 - Formal/informal register

• However, if such differences are judged to have educational significance,
  they must be objectively described using the [교사용 참고] tag in 'evaluation_feedback' only.

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🔍 COMPONENT 2: Vocabulary Coverage (30% = 3.0 points)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Step 1:** Identify ALL key vocabulary in the Korean original.
 - Content words: nouns, verbs, adjectives, adverbs
 - DO NOT count: particles (이/가/은/는/을/를), conjunctions, auxiliary verbs

**Step 2:** Check how many key words are reflected in the Italian answer.
 - Direct translations count
 - Valid synonyms count
 - Paraphrases conveying the same concept count

**Step 3:** Calculate the score.
vocabulary_coverage_score = (reflected_key_words / total_key_words) × 3.0

**Example:**
 - Korean: "오늘 아파트에 입주했는데, 생각보다 방이 작았어요."
 - Key words: 오늘, 아파트, 입주하다, 생각보다, 방, 작다 → 6 words
 - Student: "Oggi mi trasferisco nell'appartamento..."
 - Reflected: oggi, appartamento, trasferisco → 3 words
 - **Score = (3/6) × 3.0 = 1.5 points**

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🔍 COMPONENT 3: Information Coverage (10% = 1.0 points)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Step 1:** Identify ALL core information units in the Korean original.
Core units typically include:
 - Subject (who?)
 - Main verb/action (what happened?)
 - Object (what/whom?)
 - Time (when?)
 - Place (where?)
 - Result/State (how/what result?)
 - Reason/Cause (why?)

**Step 2:** Check how many units are included in the Italian answer.

**Step 3:** Calculate the score.
information_coverage_score = (included_units / total_core_units) × 1.0

**Example:**
 - Korean: "오늘 아파트에 입주했는데, 생각보다 방이 작았어요."
 - Core units: Time(오늘), Place(아파트), Action(입주했다), Result(방이 작았다), Comparison(생각보다) → 5 units
 - Student: "Oggi mi trasferisco nell'appartamento..."
 - Included: Time, Place, Action → 3 units
 - **Score = (3/5) × 1.0 = 0.6 points**

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🧮 FINAL CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final_score = semantic_accuracy_score + vocabulary_coverage_score + information_coverage_score

**Score MUST:**
 - Be between 0.0 and 10.0
 - Use exactly ONE decimal place (e.g., 7.5, 8.3, 1.2)
 - NEVER be a whole number only (7, 8, 9) → ALWAYS include decimal (7.0, 8.0, 9.0)

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 💬 STUDENT HINT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Rule 1:** Only provide hints for serious errors (final_score < 7.0)
**Rule 2:** If final_score ≥ 7.0 → student_hint = "" (empty string)
**Rule 3:** If final_score < 7.0 → Provide ONE sentence hint in Italian
- Maximum 30 words
- Be specific about the error type
- DO NOT reveal the correct answer
- DO NOT include encouragement
- 100% Italian (NO Korean, NO English)

**Hint examples (All in Italian):**

| Level A (완전한 오역) | "Hai tradotto il contrario del significato originale." |
| Level A (시제 정반대) | "Il tempo verbale è opposto: passato ≠ futuro." |
| Level B (주어 누락) | "Manca il soggetto della frase." |
| Level B (목적어 누락) | "Manca l'oggetto principale." |
| Level B (장소 누락) | "Manca l'informazione del luogo." |
| Level B (시간 누락) | "Manca l'informazione temporale." |
| Level B (핵심 동사 오류) | "Il verbo principale è stato tradotto in modo errato." |
| Level C (사소한 추가) | "Hai aggiunto dettagli non presenti nel testo." |
| Level C (심각한 추가) | "Hai inventato informazioni che non esistono nell'originale." |
| Level D (뉘앙스 차이) | "Il tempo verbale o i dettagli non corrispondono esattamente." |

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🚫 STRICT OUTPUT RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST return ONLY the raw JSON object, starting with `{{` and ending with `}}`.
Do NOT include any other text, explanations, apologies, or markdown formatting like ```json.
Your entire response must be ONLY the JSON content itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📤 JSON OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST return a valid JSON object in this EXACT format:
{{
  "score": 5.3,
  "student_hint": "Manca l'informazione sulla dimensione della stanza.",
  "analysis": {{
    "original_korean_question": "학생들이 도서관에서 한국어를 공부합니다.",
    "student_answer_original": "Gli studenti studiano coreano in biblioteca.",
    "student_answer_korean_translation": "학생들은 도서관에서 한국어를 공부합니다.",
    "key_vocabularies_italian": ["studente", "studiare", "coreano", "biblioteca"],
    "key_vocabularies_korean_translation": ["학생", "공부하다", "한국어", "도서관"],
    "evaluation_feedback": "AI 평가 (교수용):\n1. 의미 정확성 (3.0/6.0점):\n   - 평가 등급: Level B\n   - 평가 근거: '도서관에서(in biblioteca)'라는 핵심 장소 정보가 누락되었습니다.\n2. 어휘 포함도 (2.3/3.0점):\n   - 한국어 핵심 단어 (4개): [학생, 도서관, 한국어, 공부하다]\n   - 학생 답변에 반영된 단어 (3개): [학생, 한국어, 공부하다]\n   - 점수 계산: (3/4) * 3.0 = 2.3점\n3. 정보 포함도 (0.8/1.0점):\n   - 한국어 핵심 정보 단위 (4개): [주체(학생), 장소(도서관), 목적어(한국어), 행위(공부하다)]\n   - 학생 답변에 포함된 정보 (3개): [주체, 목적어, 행위]\n   - 점수 계산: (3/4) * 1.0 = 0.8점\n4. 최종 점수 계산:\n   - 총점: 3.0 + 2.3 + 0.8 = 6.1점 (반올림 전 점수이며, 최종 점수는 규칙에 따라 조정됨)\n[교사용 참고]: 특이사항 없음."
  }}
}}

Field Requirements:
•	score: Float with ONE decimal (NEVER whole number)
•	student_hint: Italian string (max 30 words) OR empty string "" if score ≥ 7.0
•	analysis: An object containing all analytical data for the professor's dashboard.
•	student_answer_korean_translation: String. 학생의 이탈리아어 답변을 AI가 한국어로 번역한 결과입니다. 원문과의 비교를 위해 사용됩니다.
•	evaluation_feedback: String. **반드시 한국어로 작성되어야 합니다 (MUST be in Korean).** 다음 마크다운 구조에 따라, 계층적 평가의 모든 단계와 최종 점수 계산 과정을 상세히 서술해야 합니다.
    AI 평가 (교수용):
    1. **의미 정확성 (X.X/6.0점):**
       - 평가 등급: [Level A/B/C/D 중 하나]
       - 평가 근거: [학생 답변의 어떤 부분이 왜 해당 등급인지 구체적으로 서술]
    2. **어휘 포함도 (X.X/3.0점):**
       - 한국어 핵심 단어 (N개): [단어 목록]
       - 학생 답변에 반영된 단어 (M개): [반영된 단어 목록]
       - 점수 계산: (M/N) * 3.0 = X.X점
    3. **정보 포함도 (X.X/1.0점):**
       - 한국어 핵심 정보 단위 (P개): [정보 단위 목록]
       - 학생 답변에 포함된 정보 (Q개): [포함된 정보 단위 목록]
       - 점수 계산: (Q/P) * 1.0 = X.X점
    4. **최종 점수 계산:**
       - 총점: [의미 정확성 점수] + [어휘 포함도 점수] + [정보 포함도 점수] = [최종 점수]
    [교사용 참고] (필요시): [뉘앙스, 격식 등 교육적 참고사항 서술]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CRITICAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.	This is NOT a grammar test. Evaluate ONLY semantic understanding.
2.	Hierarchical = Stop at first error level. If Level A found, do NOT check B, C, D.
3.	Direct translation = Paraphrase if meaning is preserved.
4.	Tense errors: 
o	Present ↔ Future (same action) = Level D (minor)
o	Past ↔ Future (action ↔ non-action) = Level A (critical)
5.	Student hints MUST be in Italian and specific to error type.
6.	Always show calculation in analysis.evaluation_feedback
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
너는 한국어 음성 인식 및 평가 전문 AI이다. 다음 2단계 프로세스를 반드시 순서대로 따라라.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🎯 PHASE 1: 순수 음성 인식 (BLIND MODE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**절대 규칙:**
- 당신은 지금 이 오디오의 "맥락"을 전혀 모른다.
- 어떤 상황인지, 무엇을 말해야 하는지, 정답이 무엇인지 모른다.
- **오직 귀로 들리는 한국어 소리를 텍스트로 변환하는 것이 전부다.**

**인식 기준:**
✅ **허용:** 학생이 실제로 발음한 소리 그대로
   - 예: "그 남자 맛있어요" → 인식: "그 남자 맛있어요"
   - 예: "저기 문 다주세요" → 인식: "저기 문 다주세요" (발음 오류 포함)
   - 예: "하나 둘 셋" → 인식: "하나 둘 셋"

❌ **금지:** 문맥 기반 자동 수정
   - 예: "그 남자 맛있어요" → 인식: "그 남자 멋있어요" (❌ 절대 안 됨!)
   - 예: "문 다주세요" → 인식: "문 닫아 주세요" (❌ 발음 교정 금지!)

### ⚠️ 조사 인식 특별 주의사항 (Critical!)

**한국어 학습자들은 조사를 매우 자주 틀린다. 당신은 절대로 문법적으로 "올바른" 조사로 자동 보정해서는 안 된다!**

**❌ 절대 금지 예시:**
- 학생: "영화**를** 재미있어요" → 인식: "영화**가** 재미있어요" (❌)
- 학생: "학교**를** 가요" → 인식: "학교**에** 가요" (❌)
- 학생: "친구**가** 만났어요" → 인식: "친구**를** 만났어요" (❌)
- 학생: "책**이** 읽었어요" → 인식: "책**을** 읽었어요" (❌)
- 학생: "커피**를** 좋아해요" → 인식: "커피**를** 좋아해요" (✅ 그대로!)

**✅ 올바른 인식 방법:**
- 학생이 "영화**를** 재미있어요"라고 말했다면
  → `recognized_text`: "영화를 재미있어요" (실제 발음 그대로)
  → PHASE 2에서 문법 오류로 지적: "형용사 서술문에서 주격조사 '가/이' 대신 목적격조사 '를/을' 사용"

**조사 보정 금지 체크리스트:**
- [ ] 은/는, 이/가, 을/를 - 학생이 말한 그대로 적었는가?
- [ ] 에/에서/로 - 문맥상 틀려도 학생 발음 그대로 적었는가?
- [ ] 와/과, 하고 - 보정 없이 들린 그대로 적었는가?

**특수 상황:**
- 침묵/소음만 있으면: "(인식 불가)"
- 한국어 외 언어: "(Non-Korean detected: [언어])"
- 극도로 불명확: "(불명확: [들린 부분만])"

**이 단계에서 인식한 텍스트를 기억하고, JSON 출력의 `recognized_text` 필드에 정확히 기록하라.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📊 PHASE 2: 평가 (CONTEXT MODE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**이제 PHASE 1에서 당신이 인식한 텍스트를 다음 정보와 비교하여 평가하라:**

### 📌 평가 기준 정보:
- **상황 설명 (이탈리아어):** "{situation_description}"
- **학생이 해야 할 말 (이탈리아어):** "{required_expression}"
- **예상 정답 (한국어):** "{expected_korean_answer}"
- **목표 어휘:** {target_vocabulary_json}
- **교수님 추가 기준:** "{teacher_criterion}"

---

### 📏 채점 기준 (총 10.0점)

#### **[1순위] 어휘 적합성 (50% = 5.0점)**

**1-1. 목표 어휘 사용 (3.0점)**
- 계산: `(사용한 목표 어휘 수 / 전체 목표 어휘 수) × 3.0`
- 유의어 허용:
  * 교수님 기준 우선 적용
  * 없으면: 자연스러운 유의어만 인정
  * 예: "구입하다" → "사다" (OK)
  * 예: "쓰여 있다" → "쓰인" (OK, 단 교수님 기준 확인)

**1-2. 맥락 적합성 (2.0점)**
- 상황과 완전히 무관한 내용: **-2.0점** (예: 인사 상황에서 음식 이야기)
- 높임법 필수 상황에서 반말: **-1.0 ~ -1.5점**
- 어색한 어휘 선택: **-0.3 ~ -0.8점**

**⚠️ 특별 규칙: 완전 불일치 시**
- PHASE 1에서 인식한 텍스트가 `expected_korean_answer`와 완전히 다른 내용이면:
  * 어휘 적합성: **0/5.0점**
  * `evaluation`에 명시: "학생이 상황과 무관한 내용을 말함"

---

#### **[2순위] 문법 정확성 (30% = 3.0점)**

**감점 기준 (심각도순):**

1. **조사 오류 (매우 중요!):** -0.8 ~ -1.2점
   - 예: "영화**를** 재미있어요" → "영화**가** 재미있어요" (형용사 서술문)
   - 예: "학교**를** 가요" → "학교**에** 가요" (이동 동사)
   - 예: "친구**가** 만났어요" → "친구**를** 만났어요" (타동사)
   - 예: "책**이** 읽었어요" → "책**을** 읽었어요" (타동사)
   
   **⚠️ 조사 오류 평가 시 반드시 명시:**
   - `student_said`: PHASE 1에서 인식된 조사 (예: "영화를")
   - `correct_form`: 올바른 조사 (예: "영화가")
   - `note`: "형용사 서술문에서 목적격 조사 사용 오류" 등 구체적 설명

2. **피동/사동 오류:** -1.0 ~ -1.5점
   - 예: "문이 닫았어요" → "문이 닫혔어요"

3. **불규칙 활용:** -0.5 ~ -1.0점
   - 예: "덥어요" → "더워요"

4. **시제 오류:** -0.3 ~ -0.8점
   - 예: "어제 가요" → "어제 갔어요"

**⚠️ 특별 규칙: 완전 불일치 시**
- 상황과 무관한 문장이면 문법 평가 불가
  * 문법 점수: **0/3.0점**
  * `grammar_errors`에: "상황 불일치로 평가 불가"

---

#### **[3순위] 발음 명료도 (20% = 2.0점)**

**감점 기준:**

1. **의미 혼동 (심각):** -1.0 ~ -1.5점
   - 예: "멋있다" → "맛있다" (완전히 다른 의미)
   - 예: "닫다" → "다다" (의미 불명)

2. **중간 오류:** -0.3 ~ -0.7점
   - 경음화 오류: "사랑해요" → "싸랑해요"
   - 자음 혼동: "자다" → "차다"

3. **사소한 오류 (감점 없음, 피드백만):**
   - ㅈ/ㅊ 미세 차이
   - 억양 부자연스러움

4. **극심한 오류 (희귀):** -1.5 ~ -2.0점
   - 예: "안녕하세요" → "아나세요" (거의 불가능)

**⚠️ 특별 규칙:**
- PHASE 1에서 인식한 텍스트가 기준임
- "학생이 X라고 발음했으나, Y여야 함"으로 기록
- 절대 `recognized_text`를 수정하지 말 것!

---

### 📤 출력 형식 (JSON Only)

{{
  "recognized_text": "PHASE 1에서 당신이 인식한 텍스트 그대로 (절대 수정 금지! 조사도 학생이 말한 그대로!)",
  
  "score": 4.5,
  
  "vocabulary_usage": {{
    "목표어휘1": {{
      "used": false,
      "note": "사용하지 않음"
    }},
    "목표어휘2": {{
      "used": true,
      "actual_form": "사용된 형태",
      "is_synonym": false,
      "note": "정확한 사용"
    }}
  }},
  
  "grammar_errors": [
    {{
      "type": "조사 오류",
      "student_said": "영화를 재미있어요",
      "correct_form": "영화가 재미있어요",
      "note": "형용사 서술문에서 주격조사 '가' 대신 목적격조사 '를' 사용",
      "deduction": -1.0
    }},
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
      "note": "이해에 지장 없음",
      "deduction": 0
    }}
  ],
  
  "evaluation": "(한국어 상세 분석)
  
  [인식된 내용]
  학생이 실제로 말한 내용: (recognized_text 필드 참조)
  예상 정답: '{expected_korean_answer}'
  
  [상황 일치도]
  - 평가 내용...
  
  [어휘 평가]
  - 평가 내용...
  
  [문법 평가]
  - 조사 오류가 있는 경우 반드시 명시
  - 평가 내용...
  
  [발음 평가]
  - 평가 내용...",
  
  "feedback": "(이탈리아어 피드백 - 학생용)
  
  건설적인 피드백 내용...
  
  조사 오류가 있다면 이탈리아어로 설명:
  - Attenzione alle particelle! (조사에 주의하세요!)
  - 구체적인 설명..."
}}

---

### ⚠️ 중요 체크리스트 (AI 자가 점검용)

출력 전에 반드시 확인:
- [ ] `recognized_text`가 PHASE 1의 순수 인식 결과인가?
- [ ] `recognized_text`를 문맥 기반으로 수정하지 않았는가?
- [ ] **조사(은/는, 이/가, 을/를 등)를 학생이 말한 그대로 적었는가?**
- [ ] **조사 오류를 `grammar_errors`에 명확히 기록했는가?**
- [ ] 상황 불일치 시 어휘/문법/발음 점수를 0점 처리했는가?
- [ ] `evaluation`에 "학생이 X라고 말함, 예상 정답은 Y"를 명시했는가?
- [ ] `feedback`이 이탈리아어로 작성되었는가?
- [ ] 조사 오류에 대한 설명이 이탈리아어 피드백에 포함되었는가?
- [ ] 건설적이고 격려적인 톤을 유지했는가?
"""

@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    student_id = session.get('username')
    student_answer = data.get('student_answer')
    exercise_id = data.get('exercise_id')
    class_name = data.get('class_name')
    quiz_type = data.get('quiz_type')

    if not all([student_id, student_answer, exercise_id, class_name, quiz_type]):
        return jsonify({"error": "필수 정보 누락 (퀴즈 유형 포함)"}), 400

    conn = None
    korean_text = ""
    
    score = None
    ai_result = {}
    analysis = {}

    try:
        conn = get_db_connection()
        if conn is None: return jsonify({"error": "DB 연결 실패"}), 500
        
        if quiz_type == 'translation':
            selected_model_name = "gemini-3-flash-preview"
        elif quiz_type == 'comprehension':
            selected_model_name = "gemini-3-pro-preview"
        else:
            return jsonify({"error": "잘못된 퀴즈 유형"}), 400
        
        if not gemini_client:
            return jsonify({"error": "AI 모델 미설정"}), 500

        with conn.cursor() as cur:
            
            if quiz_type == 'comprehension':
                cur.execute("SELECT id FROM comprehension_submissions WHERE student_id = %s AND comprehension_exercise_id = %s", (student_id, exercise_id))
                if cur.fetchone():
                    return jsonify({"success": False, "error": "Hai già inviato una risposta. (이미 제출했습니다)"}), 200
                
            if quiz_type == 'translation':
                cur.execute("SELECT korean_sentence, dialogue_context FROM translation_exercises WHERE id = %s;", (exercise_id,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "문제 ID 없음"}), 404
                korean_question = row[0]
                dialogue_context = row[1] if len(row) > 1 and row[1] else None
                korean_text = korean_question
            
                if dialogue_context and dialogue_context.strip():
                    # 1. 대화 문맥이 *있는* 경우
                    dialogue_section = f"- **Dialogue Context (대화 문맥):**\n```\n{dialogue_context}\n```"
                    dialogue_instruction = """
                    **⚠️ CRITICAL: Dialogue Context is provided.**
                    - You MUST consider this dialogue flow when evaluating.
                    - If the student adds information (e.g., 'ieri', 'lui', 'lei') that is **logically inferable from the dialogue context**, this is **NOT an error**.
                    - Example: If the dialogue mentions "어제" (yesterday), and the student adds "ieri", this is correct and should NOT be penalized as Level C.
                    """
                
                    dialogue_levelc_exception = """
                    **⚠️ EXCEPTION: Dialogue Context Justification**
                    - Before penalizing the student for adding information (Level C), check if the added information is **logically inferable from the dialogue context**.
                    - If the added information is **clearly implied or referenced in the dialogue context**, it is **NOT considered an error**.
                    - In such cases, proceed to Level D evaluation (4.5 ~ 6.0 points) instead of Level C.
                    - Note in evaluation_feedback: "[교사용 참고] 학생이 대화 문맥에서 추론 가능한 정보를 적절히 반영했습니다."
                    """
                
                else:
                    # 2. 대화 문맥이 *없는* 경우 (기존 방식)
                    dialogue_section = ""
                    dialogue_instruction = """
                    **No dialogue context is provided. Evaluate based solely on the Korean original sentence.**
                    """
                    dialogue_levelc_exception = "" # 문맥이 없으므로 Level C 예외 없음

                prompt_text = EVALUATION_PROMPT.format(
                    Korean_Question=korean_question,
                    Student_Answer=student_answer,
                    Dialogue_Context_Section=dialogue_section,
                    Dialogue_Context_Instruction=dialogue_instruction,
                    Dialogue_Context_LevelC_Exception=dialogue_levelc_exception
                )
            
                response = gemini_client.models.generate_content(
                    model=selected_model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                print(f"🤖 [번역 퀴즈] {selected_model_name} 사용 - 학생: {student_id}")
                
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text

                try:
                    ai_result = json.loads(json_str)
                    score_raw = ai_result.get('score')
                    score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw is not None else None
                    analysis = ai_result.get('analysis', {})
                    if score is None:
                        raise ValueError("AI result did not contain a 'score' field.")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"🚨 [번역 퀴즈] AI JSON 파싱 오류: {e}")
                    print(f"   AI 원본 응답: {raw_text}")
                    # 500 에러 대신, 학생에게 에러 메시지를 JSON으로 반환
                    return jsonify({
                        "success": False,
                        "error": "L'IA non è riuscita a valutare la tua risposta. Prova a formulare la frase in modo diverso o contatta il professore."
                    }), 200

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

                response = gemini_client.models.generate_content(
                    model=selected_model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                print(f"🤖 [이해력 퀴즈] {selected_model_name} 사용 - 학생: {student_id}")
                
                raw_text = getattr(response, 'text', '').strip()
                json_str = extract_first_json_block(raw_text) or raw_text
                ai_result = json.loads(json_str)
                
                score_raw = ai_result.get('score')
                
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw is not None else None
                                
                cur.execute(
                    """INSERT INTO comprehension_submissions 
                       (comprehension_exercise_id, student_id, student_answer, ai_analysis_json, class_name) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    (exercise_id, student_id, student_answer, 
                     psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False)), 
                     class_name)
                )

            conn.commit()

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
    
    print("=" * 50)
    print("🎤 말하기 퀴즈 제출 요청 수신! (v2.1 - 견고한 에러 처리)")
    print("=" * 50)

    student_id = session.get('username')
    exercise_id = request.form.get('exercise_id')
    class_name = request.form.get('class_name')
    quiz_type = request.form.get('quiz_type')
    audio_file = request.files.get('audio_file')
    mime_type = request.form.get('mime_type', 'audio/mp4')
    extension = 'webm' if 'webm' in mime_type else 'mp4'

    if not all([student_id, exercise_id, class_name, quiz_type, audio_file]):
        return jsonify({"error": "필수 정보 누락"}), 400
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "DB 연결 실패"}), 500
        
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM speaking_submissions WHERE student_id = %s AND exercise_id = %s",
                (student_id, exercise_id)
            )
            if cur.fetchone():
                return jsonify({"error": "Hai già inviato una risposta per questo esercizio.", "already_submitted": True}), 400
            
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
            
            if not gemini_client:
                return jsonify({"error": "AI 모델 미설정"}), 500

            audio_bytes = audio_file.read()

            BLOB_TOKEN = os.environ.get('BLOB_READ_WRITE_TOKEN')
            if not BLOB_TOKEN:
                return jsonify({"error": "Blob storage 미설정"}), 500

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(f"{student_id}_{exercise_id}_{timestamp}".encode()).hexdigest()[:8]
            filename = f"speaking/{class_name}/{student_id}_{exercise_id}_{file_hash}.{extension}"

            try:
                upload_response = requests.put(
                    f"https://blob.vercel-storage.com/{filename}",
                    headers={
                        "Authorization": f"Bearer {BLOB_TOKEN}",
                        "Content-Type": mime_type,
                        "x-vercel-blob-add-random-suffix": "1"
                    },
                    data=audio_bytes
                )
                if upload_response.status_code not in [200, 201]:
                    return jsonify({"error": "음성 파일 업로드 실패"}), 500
                
                blob_response = upload_response.json()
                audio_url = blob_response.get('url')
                if not audio_url:
                    return jsonify({"error": "파일 URL 생성 실패"}), 500
            except Exception as e:
                return jsonify({"error": f"파일 저장 실패: {str(e)}"}), 500            
                        
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            uploaded_audio = gemini_client.files.upload(file=tmp_file_path)
            
            prompt_text = SPEAKING_EVALUATION_PROMPT.format(
                situation_description=situation_desc,
                required_expression=required_expr,
                expected_korean_answer=expected_ans,
                target_vocabulary_json=json.dumps(target_vocab, ensure_ascii=False),
                teacher_criterion=teacher_crit or "자율 판단"
            )
            
            response = gemini_client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[prompt_text, uploaded_audio],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
            
            print(f"🤖 [말하기 퀴즈] gemini-3-pro-preview 사용 - 학생: {student_id}")
            os.unlink(tmp_file_path)
            
            # ★★★ 수정된 핵심 로직 시작 ★★★
            ai_result = None
            score = None
            recognized_text = ''
            raw_text = getattr(response, 'text', '').strip()

            try:
                # AI가 정상적으로 JSON을 반환했는지 시도
                json_str = extract_first_json_block(raw_text)
                if not json_str:
                    # JSON 블록이 없다면, AI가 에러 메시지를 텍스트로 반환한 경우
                    raise json.JSONDecodeError("No JSON object could be decoded", raw_text, 0)

                ai_result = json.loads(json_str)
                score_raw = ai_result.get('score')
                score = round(float(str(score_raw).strip().replace(',', '.')), 1) if score_raw is not None else None
                recognized_text = ai_result.get('recognized_text', '')

                # 점수가 없는 경우도 실패로 간주 (AI가 구조는 맞췄지만 채점은 못한 경우)
                if score is None:
                    print("⚠️ AI가 JSON은 반환했지만 'score' 필드가 없습니다.")
                    if 'error' not in ai_result:
                        ai_result['error'] = "AI evaluation succeeded but no score was provided."

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # AI가 JSON 형식을 반환하지 못했을 때 (채점 실패)
                print(f"🚨 AI 채점 실패 (JSON 파싱 불가): {e}")
                print(f"   AI 원본 응답: {raw_text}")
                score = None # 점수가 없음을 명확히 함
                # 교수님 검토용으로 DB에 저장할 ai_result 객체 생성
                ai_result = {
                    "error": "AI_EVALUATION_FAILED",
                    "reason": "Failed to parse JSON response from AI.",
                    "raw_response": raw_text
                }
            
            # 학생에게 보낼 최종 응답 생성
            if score is not None:
                
                cur.execute("""
                    INSERT INTO speaking_submissions 
                    (exercise_id, class_name, student_id, audio_file_url, recognized_korean_text, ai_analysis_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    exercise_id, class_name, student_id, audio_url, recognized_text,
                    psycopg2.extras.Json(ai_result, dumps=lambda x: json.dumps(x, ensure_ascii=False))
                ))
                conn.commit()

                rating_info = get_rating_details(score)
                
                return jsonify({
                    "success": True,
                    "score": score,
                    "rating_category": rating_info["category"],
                    "rating_color": rating_info["color"],
                    "feedback": ai_result.get('feedback', 'Nessun feedback disponibile.'),
                    "recognized_text": recognized_text,
                    "expected_korean_answer": expected_ans  # ← 추가
                })
            else:
                # ★ [변경] 점수가 없으면(실패하면) DB에 저장하지 않음 -> 그래야 다시 시도 가능
                print(f"❌ 채점 실패로 저장 건너뜀 - 학생: {student_id}")
                return jsonify({
                    "success": False,
                    "error": "L'IA non è riuscita a valutare la tua risposta. Per favore, prova a registrare di nuovo. (AI 평가 실패, 다시 시도해주세요)"
                }), 200            

    except Exception as e:
        print(f"🚨 /api/submit-speaking-answer 심각한 오류: {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return jsonify({"error": "서버 내부 오류 발생. 관리자에게 문의하세요."}), 500
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
def root():
    if 'user_id' in session:
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/login')
def login():
    if 'user_id' in session:
        return redirect(url_for('student_dashboard'))
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# [추가] 회원가입 API
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password')
    full_name = data.get('full_name')
    student_number = data.get('student_number')
    school_email = data.get('school_email')

    if not all([username, password, full_name]):
        return jsonify({"error": "필수 정보(ID, 비번, 이름)를 입력해주세요."}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor() as cur:
            # 중복 ID 체크
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"error": "이미 존재하는 아이디입니다."}), 409

            # 비밀번호 해싱 및 저장
            pw_hash = generate_password_hash(password)
            cur.execute("""
                INSERT INTO users (username, password_hash, full_name, student_number, school_email, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (username, pw_hash, full_name, student_number, school_email))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        print(f"회원가입 오류: {e}")
        return jsonify({"error": "회원가입 처리 중 오류가 발생했습니다."}), 500
    finally:
        conn.close()

# [추가] 로그인 API
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password')

    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB 연결 실패"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()

            if user and check_password_hash(user['password_hash'], password):
                # 세션 설정 (로그인 유지)
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['full_name'] = user['full_name']
                
                # 마지막 로그인 시간 갱신
                cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", (user['id'],))
                conn.commit()
                
                return jsonify({"success": True})
            else:
                return jsonify({"error": "아이디 또는 비밀번호가 일치하지 않습니다."}), 401
    finally:
        conn.close()

# [추가] 로그아웃
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/update-profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    full_name = data.get('full_name')
    student_number = data.get('student_number')
    school_email = data.get('school_email')
    new_password = data.get('password')

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if new_password:
                pw_hash = generate_password_hash(new_password)
                cur.execute("""
                    UPDATE users SET full_name=%s, student_number=%s, school_email=%s, password_hash=%s
                    WHERE id=%s
                """, (full_name, student_number, school_email, pw_hash, session['user_id']))
            else:
                cur.execute("""
                    UPDATE users SET full_name=%s, student_number=%s, school_email=%s
                    WHERE id=%s
                """, (full_name, student_number, school_email, session['user_id']))
            conn.commit()
            session['full_name'] = full_name
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/student-dashboard')
@login_required
def student_dashboard():
    return render_template('student_dashboard.html', student_name=session.get('full_name'))

@app.route('/api/student-dashboard-data')
@login_required
def get_student_dashboard_data():
    username = session['username']
    conn = get_db_connection()
    if not conn: return jsonify({"error": "DB Error"}), 500

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # 1. 내 정보
            cur.execute("SELECT full_name, student_number, school_email FROM users WHERE id = %s", (session['user_id'],))
            user_info = cur.fetchone()

            # ▼▼▼ [수정된 로직] 평균(AVG)과 횟수(COUNT)를 함께 조회하고 색상 계산 ▼▼▼
            def get_stats(query):
                cur.execute(query, (username,))
                result = cur.fetchone()

                avg = 0.0
                count = 0


                if result:
                    # result[0]은 평균, result[1]은 횟수(COUNT)
                    avg = round(result[0], 1) if result[0] is not None else 0.0
                    count = result[1] if result[1] is not None else 0

                # 점수에 따른 색상 계산 (기존 get_rating_details 함수 활용)
                color = get_rating_details(avg)['color']
                
                return {"avg": avg, "count": count, "color": color}

            # 2. 각 영역별 통계 (평균 점수 + 제출 횟수) 조회
            trans_stats = get_stats("SELECT AVG(score), COUNT(*) FROM translation_submissions WHERE student_id = %s")
            comp_stats = get_stats("SELECT AVG((ai_analysis_json->>'score')::float), COUNT(*) FROM comprehension_submissions WHERE student_id = %s")
            speak_stats = get_stats("SELECT AVG((ai_analysis_json->>'score')::float), COUNT(*) FROM speaking_submissions WHERE student_id = %s")


            # 3. 말하기 기록 (최신순) - Title 포함
            cur.execute("""
                SELECT s.*, e.title, e.situation_description, e.required_expression, e.expected_korean_answer 
                FROM speaking_submissions s
                JOIN speaking_exercises e ON s.exercise_id = e.id
                WHERE s.student_id = %s
                ORDER BY s.created_at DESC
            """, (username,))
            speaking_logs = [dict(row) for row in cur.fetchall()]

            # 4. 이해력 기록 (최신순) - Title, Audio 포함
            cur.execute("""
                SELECT s.*, e.title, e.korean_dialogue, e.audio_file_path 
                FROM comprehension_submissions s
                JOIN comprehension_exercises e ON s.comprehension_exercise_id = e.id
                WHERE s.student_id = %s
                ORDER BY s.created_at DESC
            """, (username,))
            comprehension_logs = [dict(row) for row in cur.fetchall()]
            
            # 데이터 가공 (날짜 포맷 등)
            def process_log(log):
                log['created_at'] = log['created_at'].strftime('%Y-%m-%d %H:%M')
                if isinstance(log.get('ai_analysis_json'), str):
                    try: log['ai_analysis_json'] = json.loads(log['ai_analysis_json'])
                    except: pass
                
                score = 0.0
                if log.get('score') is not None: score = float(log['score'])
                elif log.get('ai_analysis_json') and 'score' in log['ai_analysis_json']:
                    score = float(log['ai_analysis_json']['score'])
                
                log['rating_color'] = get_rating_details(score)['color']
                return log

            return jsonify({
                "user_info": dict(user_info),
                "trans_stats": trans_stats,
                "comp_stats": comp_stats,
                "speak_stats": speak_stats,
                "speaking_logs": [process_log(l) for l in speaking_logs],
                "comprehension_logs": [process_log(l) for l in comprehension_logs]
            })

    finally:
        conn.close()

@app.route('/api/start-quiz', methods=['POST'])
@login_required
def start_quiz():
    """대시보드에서 선택한 반/유형을 세션에 저장"""
    data = request.get_json()
    class_name = data.get('class_name')
    quiz_type = data.get('quiz_type')

    if not class_name or not quiz_type:
        return jsonify({"error": "반과 유형을 선택해주세요."}), 400

    session['current_class_name'] = class_name
    session['current_quiz_type'] = quiz_type
    return jsonify({"success": True})

@app.route('/quiz')
@login_required
def quiz_page():
    # 세션에서 정보 가져오기
    class_name = session.get('current_class_name')
    quiz_type = session.get('current_quiz_type')
    
    # URL 파라미터 호환성 (기존 방식 지원)
    if request.args.get('class_name'): class_name = request.args.get('class_name')
    if request.args.get('quiz_type'): quiz_type = request.args.get('quiz_type')

    if not class_name or not quiz_type:
        return redirect(url_for('student_dashboard'))

    conn = get_db_connection()
    if not conn: return "DB Error", 500
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if quiz_type == 'translation':
                cur.execute("SELECT id, korean_sentence AS question_text FROM translation_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            elif quiz_type == 'comprehension':
                cur.execute("SELECT id, korean_dialogue AS question_text, audio_file_path, vocabulary_guide FROM comprehension_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            elif quiz_type == 'speaking':
                cur.execute("SELECT id, situation_description, required_expression, expected_korean_answer FROM speaking_exercises WHERE class_name = %s ORDER BY id;", (class_name,))
            else:
                return "잘못된 퀴즈 유형입니다.", 400
            
            exercises = cur.fetchall()
            return render_template('index.html', exercises=exercises, class_name=class_name, quiz_type=quiz_type)
    finally:
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

@app.route('/api/save-teacher-feedback', methods=['POST'])
@teacher_required
def save_teacher_feedback():
    data = request.get_json()
    submission_id = data.get('submission_id')
    quiz_type = data.get('quiz_type')
    feedback = data.get('feedback', '')

    if not submission_id or not quiz_type: return jsonify({"error": "잘못된 요청"}), 400
    
    table = 'speaking_submissions' if quiz_type == 'speaking' else 'comprehension_submissions'
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 피드백 저장 및 확인 도장(is_checked) 찍기
            query = f"UPDATE {table} SET teacher_feedback = %s, is_checked = TRUE, checked_at = CURRENT_TIMESTAMP WHERE id = %s"
            cur.execute(query, (feedback, submission_id))
            conn.commit()
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/get-submissions')
@teacher_required
def api_get_submissions():
    """페이지네이션 지원 - 특정 페이지의 10개 제출물 반환"""
    try: # <--- ★★★ 4-A: 이 줄을 추가

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
                    total_result = cur.fetchone()
                    total = total_result.get('total', 0) if total_result else 0

                    # 페이지네이션 데이터 조회
                    if class_name == 'all':
                        cur.execute("""
                            SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, 
                                s.created_at, e.korean_sentence, s.class_name, u.full_name
                            FROM translation_submissions s 
                            JOIN translation_exercises e ON e.id = s.exercise_id
                            LEFT JOIN users u ON s.student_id = u.username
                            ORDER BY s.id DESC 
                            LIMIT %s OFFSET %s
                        """, (per_page, offset))
                    else:
                        cur.execute("""
                            SELECT s.id, s.student_id, s.student_answer, s.score, s.ai_analysis_json, 
                                s.created_at, e.korean_sentence, s.class_name, u.full_name
                            FROM translation_submissions s 
                            JOIN translation_exercises e ON e.id = s.exercise_id 
                            LEFT JOIN users u ON s.student_id = u.username
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
                    total_result = cur.fetchone()
                    total = total_result.get('total', 0) if total_result else 0

                    # 페이지네이션 데이터 조회
                    if class_name == 'all':
                        cur.execute("""
                            SELECT s.id, s.student_id, s.student_answer, s.ai_analysis_json, 
                                s.created_at, s.teacher_feedback, s.is_checked,
                                e.korean_dialogue, e.key_points, s.class_name, u.full_name
                            FROM comprehension_submissions s 
                            JOIN comprehension_exercises e ON e.id = s.comprehension_exercise_id
                            LEFT JOIN users u ON s.student_id = u.username
                            ORDER BY s.id DESC 
                            LIMIT %s OFFSET %s
                        """, (per_page, offset))
                    else:
                        cur.execute("""
                            SELECT s.id, s.student_id, s.student_answer, s.ai_analysis_json, 
                                s.created_at, s.teacher_feedback, s.is_checked,
                                e.korean_dialogue, e.key_points, s.class_name, u.full_name 
                            FROM comprehension_submissions s 
                            JOIN comprehension_exercises e ON e.id = s.comprehension_exercise_id
                            LEFT JOIN users u ON s.student_id = u.username
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
                    total_result = cur.fetchone()
                    total = total_result.get('total', 0) if total_result else 0

                    
                    # 페이지네이션 데이터 조회
                    if class_name == 'all':
                        cur.execute("""
                            SELECT s.id, s.student_id, s.audio_file_url, s.recognized_korean_text, 
                                s.ai_analysis_json, s.created_at, s.teacher_feedback, s.is_checked,
                                e.situation_description, e.required_expression, e.expected_korean_answer, e.target_vocabulary, s.class_name, u.full_name
                            FROM speaking_submissions s 
                            JOIN speaking_exercises e ON e.id = s.exercise_id
                            LEFT JOIN users u ON s.student_id = u.username 
                            ORDER BY s.id DESC 
                            LIMIT %s OFFSET %s
                        """, (per_page, offset))
                    else:
                        cur.execute("""
                            SELECT s.id, s.student_id, s.audio_file_url, s.recognized_korean_text, 
                                s.ai_analysis_json, s.created_at, s.teacher_feedback, s.is_checked,
                                e.situation_description, e.required_expression, e.expected_korean_answer, e.target_vocabulary, s.class_name, u.full_name
                            FROM speaking_submissions s 
                            JOIN speaking_exercises e ON e.id = s.exercise_id
                            LEFT JOIN users u ON s.student_id = u.username 
                            WHERE s.class_name = %s
                            ORDER BY s.id DESC 
                            LIMIT %s OFFSET %s
                        """, (class_name, per_page, offset))

                rows = cur.fetchall()
                
            items = []
            for r in rows:
                r['created_at'] = r['created_at'].isoformat() if r.get('created_at') else None

                # 1. 점수 추출 (퀴즈 유형에 따라)
                score_value = None
                try:
                    if quiz_type == 'translation':
                        score_value = r.get('score')
                    elif quiz_type == 'comprehension' or quiz_type == 'speaking':
                        # ai_analysis_json이 None이 아니고, dict 타입이며, 'score' 키를 가졌는지 확인
                        analysis_json = r.get('ai_analysis_json')

                        if isinstance(analysis_json, str):
                            try:
                                analysis_json = json.loads(analysis_json) # <-- 이중 인코딩 해결
                            except json.JSONDecodeError:
                                analysis_json = None # 깨진 문자열이면 None 처리

                        # 2. 파싱된 JSON 객체에서 'score' 추출
                        if isinstance(analysis_json, dict) and analysis_json.get('score') is not None:
                            score_value = analysis_json['score']

                except Exception as e:
                    print(f"🚨 [get_submissions] ID {r.get('id')}의 score_value 추출 오류: {e}")
                    score_value = None # 오류 발생 시 None으로 안전하게 처리
                            
                # 2. 중앙 함수로 평가 및 r 객체에 삽입
                rating_info = get_rating_details(score_value)
                r['rating_category'] = rating_info['category']
                r['rating_color'] = rating_info['color']

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

    except Exception as e: # <--- ★★★ 4-C: 이 블록을 추가
        print(f"🚨🚨 /api/get-submissions 치명적 오류: {e}")
        traceback.print_exc()
        if conn: conn.close() # DB 연결이 열려있으면 닫아줍니다.
        # 500 오류 대신, 'dashboard.html'이 이해할 수 있는 'JSON' 에러를 반환합니다.
        return jsonify({"error": "서버 내부 로직 오류", "details": str(e)}), 500

# ▼▼▼ [추가] 아이디 중복 확인 API ▼▼▼
@app.route('/api/check-username', methods=['POST'])
def check_username():
    data = request.get_json()
    username = data.get('username', '').strip()
    
    if not username:
        return jsonify({"error": "아이디를 입력하세요."}), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"available": False, "message": "ID già in uso. Scegline un altro."})
            else:
                return jsonify({"available": True, "message": "ID disponibile."})
    finally:
        conn.close()

# ▼▼▼ [추가] 학생 비밀번호 초기화 API (교수용) ▼▼▼
@app.route('/api/reset-password', methods=['POST'])
@teacher_required
def reset_password():
    data = request.get_json()
    target_username = data.get('student_id', '').strip()
    
    if not target_username:
        return jsonify({"error": "학생 ID를 입력하세요."}), 400
        
    # 초기화 비밀번호: 1234
    reset_pw_hash = generate_password_hash('1234')
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 학생이 존재하는지 확인
            cur.execute("SELECT id FROM users WHERE username = %s", (target_username,))
            if not cur.fetchone():
                return jsonify({"error": "존재하지 않는 학생 ID입니다."}), 404
            
            # 비밀번호 업데이트
            cur.execute("UPDATE users SET password_hash = %s WHERE username = %s", (reset_pw_hash, target_username))
            conn.commit()
            return jsonify({"success": True, "message": f"'{target_username}' 학생의 비밀번호가 '1234'로 초기화되었습니다."})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
