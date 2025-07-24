import os
import google.generativeai as genai
from flask import Flask, render_template, jsonify

# --- 기본 설정 ---
app = Flask(__name__)

# --- AI 모델 설정 (가장 중요한 부분) ---
# Vercel에 저장된 환경 변수(API 키)를 가져옵니다.
# Vercel 프로젝트 설정에서 환경 변수의 이름을 'GEMINI_API_KEY'로 저장했다고 가정합니다.
# 만약 다른 이름으로 저장하셨다면, 아래 코드의 'GEMINI_API_KEY' 부분을 교수님께서 지정한 이름으로 바꿔주세요.
try:
    api_key = os.environ.get('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    # 사용할 AI 모델을 지정합니다. 'gemini-1.5-flash'는 빠르고 효율적입니다.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
except Exception as e:
    # API 키 설정에 실패하면, model 변수를 None으로 설정하고 오류를 출력합니다.
    model = None
    print(f"🚨 Gemini AI 모델 설정 오류: {e}")


# --- 웹 페이지 라우트 ---
# '/' 주소로 접속하면 학생용 퀴즈 페이지(index.html)를 보여줍니다.
@app.route('/')
def home():
    return render_template('index.html')


# --- API 엔드포인트 라우트 ---
# '/api/generate-quiz' 주소로 접속하면 AI가 퀴즈를 생성해주는 '창구' 역할을 합니다.
@app.route('/api/generate-quiz')
def generate_quiz():
    # 모델 설정이 실패했다면 에러 메시지를 반환합니다.
    if not model:
        return jsonify({"error": "AI 모델이 설정되지 않았습니다. API 키를 확인하세요."}), 500

    # AI에게 전달할 명령문(프롬프트)입니다.
    # 우리의 최종 설계도에 맞게, JSON 형식으로 답변을 달라고 구체적으로 요청합니다.
    prompt = """
    당신은 이탈리아 학생들에게 한국어를 가르치는 언어 교사입니다.
    초급 수준의 한국어 듣기 평가 퀴즈를 딱 1개만 생성해주세요.

    아래 규칙을 반드시 따르는 JSON 형식으로만 답변해야 합니다:
    {
      "question": "학생에게 들려줄 한국어 문장",
      "options": [
        "이탈리아어 해석 선택지 1",
        "이탈리아어 해석 선택지 2",
        "이탈리아어 해석 선택지 3",
        "이탈리아어 해석 선택지 4"
      ],
      "answer": "정답인 이탈리아어 해석"
    }
    """

    try:
        # AI에게 프롬프트를 보내고 응답을 받습니다.
        response = model.generate_content(prompt)
        # AI가 생성한 텍스트 응답을 JSON 형식으로 웹페이지에 보여줍니다.
        return jsonify({"ai_response": response.text})
    except Exception as e:
        # AI 호출 중 에러가 발생하면 에러 메시지를 반환합니다.
        return jsonify({"error": f"AI 호출 중 오류 발생: {e}"}), 500