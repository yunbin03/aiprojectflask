import pickle

# 모델 파일 로드
try:
    with open('lr_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 모델 내용을 문자열로 변환
    model_details = str(model)

    # HTML 파일 생성
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Details</title>
    </head>
    <body>
        <h1>Model Details</h1>
        <pre>{model_details}</pre>
    </body>
    </html>
    """

    # HTML 파일 저장
    with open('static/model_details.html', 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)

    print("모델 세부정보가 model_details.html 파일에 저장되었습니다. 브라우저에서 열어 확인하세요.")

except Exception as e:
    print(f"모델 파일 로드 중 오류 발생: {e}")