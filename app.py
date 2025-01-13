import pickle
import numpy as np
from flask import Flask,render_template, request
import joblib

app = Flask(__name__)

#모델 로드
#with open('lr_model.pkl','rb') as f:
#    model = pickle.load(f)

# 모델파일이 잘 불러오는지 확인하기 위한 코드 터미널에 출력됨
try:
    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("모델 로딩 성공")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")

try:
    scaler = joblib.load('scaler.joblib')
    print("스케일러 로딩 성공")
except Exception as e:
    print(f"스케일러 로딩 중 오류 발생: {e}")

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/result',methods=['POST'])
def result():
    # 사용자 입력 받기
    gender = request.form['gender']
    age = int(request.form['age'])
    cause = request.form['cause']
    fee_charged = int(request.form['fee_charged'])
    membership_period = int(request.form['membership_period'])
    number_of_claims = int(request.form['number_of_claims'])
    number_of_dependants = int(request.form['number_of_dependants'])

    try:
        # 범주형 변수 처리: 원-핫 인코딩
        gender_encoded = 1 if gender.lower() == 'female' else 0  # female=1, male=0


        cause_mapping = {
            "Accident At Home": 0,
            "Accident At Work": 1,
            "Road Traffic Accident": 2,
            "Other": 3
        }
        cause_encoded = cause_mapping.get(cause, 3)

        # 특징 결합
        features = [
            float(gender_encoded),
            float(age),
            float(cause_encoded),
            float(fee_charged),
            float(membership_period),
            float(number_of_claims),
            float(number_of_dependants)
        ]

        #입력 데이터 배열로 변환
        final_features = np.array(features).reshape(1, -1)


        # 예측 수행
        prediction = model.predict_proba(final_features)
        fraud_probability = prediction[0][1] * 100  # 백분율로 변환

        return render_template('result.html', fraud_probability=fraud_probability)

    except ValueError as e:
        error_message = f"데이터 변환 중 오류 발생: {str(e)}"
        print(f"Error details: {error_message}")  # 콘솔에 오류 정보 출력
        print(f"Input data: {request.form}")  # 입력 데이터 출력
        return render_template('error.html', error_message=error_message)
    except Exception as e:
        error_message = f"예상치 못한 오류 발생: {str(e)}"
        print(f"Unexpected error: {error_message}")
        print(f"Input data: {request.form}")
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)