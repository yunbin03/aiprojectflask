import pickle
from flask import Flask,render_template, request

app = Flask(__name__)  # Flask 애플리케이션 객체 생성

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    try:
        # 모델 파일 로드
        with open('lr_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # 테스트 데이터를 로드 (데이터 준비 필요)
        # 테스트 데이터를 준비하거나 직접 하드코딩
        X_test = [[1, 25, 0, 300, 12, 1, 0]]  # 예시 입력값
        y_test = [0]  # 예시 실제값

        # 예측값 계산
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 평가 지표 계산 (여기서는 간단히 예측값과 확률 출력)
        evaluation_result = {
            "Predicted": y_pred.tolist(),
            "Predicted_Probability": y_pred_proba.tolist(),
            "Actual": y_test,
        }

        return render_template('evaluation.html', evaluation_result=evaluation_result)

    except Exception as e:
        error_message = f"모델 평가 중 오류 발생: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)  # Flask 서버 실행