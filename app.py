from flask import Flask, request, render_template
import pandas as pd
from io import StringIO

# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# # import gluonnlp as nlp
# import numpy as np
# from tqdm import tqdm, tqdm_notebook

# #KoBERT
# from kobert.utils import get_tokenizer
# from kobert.pytorch_kobert import get_pytorch_kobert_model
# #transformer
# from transformers import AdamW
# from transformers.optimization import get_cosine_schedule_with_warmup

# #GPU 설정
# device = torch.device("cuda:0")
# #bertmodel의 vocabulary
# bertmodel, vocab = get_pytorch_kobert_model()

# chatbot_data = pd.read_excel('data.xlsx')

# len(chatbot_data) #79473 
# chatbot_data.sample(n=10)



# model_name = "kykim/bert-kor-base"
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# tokenizer = BertTokenizer.from_pretrained(model_name)

# def classify_emotion(text):
#     # 텍스트 토큰화 및 패딩
#     tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

#     # 예측 수행
#     with torch.no_grad():
#         prediction = model(**tokens)

# 		# 예측 결과를 바탕으로 감정 출력
#     prediction = F.softmax(prediction.logits, dim=1)
#     output = prediction.argmax(dim=1).item()
#     labels = ["부정적", "긍정적"]
#     print(f'[{output}]\\n')

# def predict_sentence():
#     input_sentence = input('문장을 입력해 주세요: ')
#     classify_emotion(input_sentence)


from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        # 파일이 업로드되었는지 확인
        if 'file' not in request.files:
            return "파일이 업로드되지 않았습니다."
        if 'name' not in request.values:
            return "이름이 입력되지 않았습니다."
        
        file = request.files['file']
        name = request.values['name']
        
        # 파일이 비어있는지 확인
        if file.filename == '':
            return "선택된 파일이 없습니다."
        
        # 파일을 읽고 처리
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        data = []

        # 최대 필드 개수를 찾기 위한 초기값
        max_fields = 0

        # 각 행을 처리
        for line in lines:
            fields = line.split()  # 공백을 기준으로 분할
            max_fields = max(max_fields, len(fields))
            data.append(fields)

        # 모든 행의 필드 수를 최대 필드 개수에 맞추기
        for fields in data:
            fields.extend([''] * (max_fields - len(fields)))

        # DataFrame으로 변환
        df = pd.DataFrame(data, columns=[f'col{i+1}' for i in range(max_fields)])

        # 결과를 저장할 배열 초기화
        chat_data = []

        # 각 행을 한 줄씩 처리하여 필요한 데이터를 chat_data 리스트에 추가
        for _, row in df.iterrows():
            user = row.iloc[0]       # 첫 번째 열은 사용자 이름
            am_pm = row.iloc[1]      # 두 번째 열은 오전/오후
            time = row.iloc[2]       # 세 번째 열은 시간
            message = ' '.join(map(str, row[3:])).strip()  # 나머지는 대화 내용

            if not (user.startswith('[') and user.endswith(']')):
                continue

            # chat_data 리스트에 정보 추가
            chat_data.append([user, am_pm, time, message])

        # 처리된 데이터를 result.html 템플릿으로 전달
        return render_template("result.html", chat_data=chat_data)
    
if __name__ == '__main__':
    app.run(debug=True)
