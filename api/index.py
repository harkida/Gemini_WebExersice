from flask import Flask, render_template

# Flask 애플리케이션을 생성합니다.
app = Flask(__name__, template_folder='../templates')

# 웹사이트의 루트('/') 주소로 요청이 오면 이 함수를 실행합니다.
@app.route('/')
def home():
    # 'templates' 폴더 안에 있는 'index.html' 파일을 찾아서 사용자에게 보여줍니다.
    return render_template('index.html')

# 이 파일이 직접 실행될 때 (로컬 테스트용) 서버를 가동합니다.
if __name__ == '__main__':
    app.run(debug=True)