from flask import Flask, request, render_template
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pykospacing import Spacing
from hanspell import spell_checker
from konlpy.tag import Okt  
from ckonlpy.tag import Twitter
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.normalizer import *
from soynlp.tokenizer import MaxScoreTokenizer
from concurrent.futures import ThreadPoolExecutor  
import re
import os
import pickle # 객체를 직렬화하여 저장
twt = Okt()


# 파일 처리 함수
def process_file(file):
    chunk_size = 1000  # 한 번에 읽을 줄 수
    chunk_list = []  # 모든 청크를 저장할 리스트

    for chunk in pd.read_csv(file, chunksize=chunk_size, encoding='utf-8', delimiter='\t'):
        # 각 청크 처리 (여기서 chunk는 pandas DataFrame)
        # 예를 들어, 각 청크를 전처리하는 로직을 여기에 추가
        processed_chunk = process_chunk(chunk)
        
        # 처리된 청크를 리스트에 추가
        chunk_list.append(processed_chunk)

    # 모든 청크가 처리된 후 합치기
    df = pd.concat(chunk_list, ignore_index=True)
    return df

# 청크 처리 함수
def process_chunk(chunk):
    # 각 청크에서 필요한 전처리 작업을 수행
    # 예를 들어, 공백을 기준으로 분할하는 등의 작업
    data = []
    for _, row in chunk.iterrows():
        fields = row.iloc[0].split()  # 공백 기준 분할
        data.append(fields)

    # DataFrame으로 변환 후 반환
    max_fields = max(len(fields) for fields in data)
    for fields in data:
        fields.extend([''] * (max_fields - len(fields)))
    
    return pd.DataFrame(data)

def process_message(message, word_score_table, stop_words):
    
    scores = {word: score.cohesion_forward for word, score in word_score_table.items()}
    maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
    
    # 메시지 토큰화 및 명사 추출
    tokenized_message = maxscore_tokenizer.tokenize(message)
    words = []
    for tokens in tokenized_message:    
        tokens = twt.pos(tokens)
        for word, j in tokens:
            if j=='Noun' and len(word) > 1 and word not in stop_words:  # 1글자 제외, 불용어 제외
                words.append(word)
    print("Processed words:", words)
    return words

# 사용자별 채팅 데이터 분리 함수    
def extract_chat_data(df):
    from collections import defaultdict
    
    chat_data = {}
    log_data = []

    user_words = defaultdict(list)  # 사용자별 단어 저장
    user_sentiments = defaultdict(lambda: {'positive': 0, 'negative': 0})  # 긍정/부정 단어 빈도 저장

    all_word = []
    stop_words = set(['그리고', '이것', '그것', '저것','이거','그거','저거','이제', '혹시', '이번', '저번',
                '보통', '먼저', '오늘', '내일', '어제', '오전', '오후', '잠깐', '일찍', '정도', '이제', '다시', '바로', '대신', '거의', '어디'])
    positive_words = ['좋다', '훌륭', '기쁘다', '멋지다', '사랑', '행복']  # 추가 가능
    negative_words = ['나쁘다', '싫다', '짜증', '슬프다', '화난다', '불행']  # 추가 가능

    urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
    corpus = DoublespaceLineCorpus("2016-10-20.txt")

    # 훈련된 모델이 있는지 확인하고, 있으면 로드하고 없으면 새로 훈련
    if os.path.exists('word_score_table.pkl'):
        with open('word_score_table.pkl', 'rb') as f:
            word_score_table = pickle.load(f)
        print("훈련된 결과를 불러왔습니다.")
    else:
        print("훈련된 결과가 없어서 새로 훈련을 시작합니다.")
        word_extractor = WordExtractor()
        word_extractor.train(corpus)
        word_score_table = word_extractor.extract()
        with open('word_score_table.pkl', 'wb') as f:
            pickle.dump(word_score_table, f)
        print("훈련이 완료되었습니다.") 

    ## Process chat messages in parallel for efficiency
    with ThreadPoolExecutor() as executor:
        futures = []
        for _, row in df.iterrows():
            user = row.iloc[0]  # 첫 번째 열: 사용자 이름
            am_pm = row.iloc[1]  # 두 번째 열: 오전/오후
            time = row.iloc[2]  # 세 번째 열: 시간
            message = ' '.join(map(str, row[3:])).strip()  # 메시지 내용

            # 형식 확인 
            if not (user.startswith('[') and user.endswith(']')):
                continue
            if not (am_pm.startswith('[') and time.endswith(']')):
                continue

            user = user[1:-1]
            am_pm = am_pm[1:]
            time = time[:-1]

            # 시간 변환
            hour, minute = map(int, time.split(':'))
            if am_pm == '오후':
                hour = (hour % 12) + 12
            elif hour == 12:
                hour = 0
            formatted_time = f"{hour:02}:{minute:02}"

            ## 채팅 메시지 전처리
    
            # 반복되는 문자 정제
            message = re.sub(r'(.)\1+', r'\1\1', message) 

            # 맞춤법 검사
            message = spell_checker.check(message)
            message = message.checked

            # 띄어쓰기 검사
            spacing = Spacing()
            message = spacing(message)

            futures.append(executor.submit(process_message, message, word_score_table, stop_words))
            
            # 사용자별 채팅 데이터 저장
            if user not in chat_data:
                chat_data[user] = []
            chat_data[user].append([formatted_time, message])
            # 채팅 로그 순서대로 데이터 저장
            log_data.append([user, formatted_time, message])

        # Wait for all futures to complete and collect results
        print(futures)
        for future in futures:
            words = future.result()
            print(words)
            all_word.extend(words)

    # 사용자별 단어 리스트 생성
    for word in all_word:
        # 사용자별로 단어 저장
        user_words[user].append(word)

        # ## 메시지에서 명사 추출 및 분석

        # # 학습에 기반한 단어 토큰화
        # tokenized_message = maxscore_tokenizer.tokenize(message)
        # for words in tokenized_message:
        #     print(words)
        #     word = twt.pos(words)    
        #     for i, j in word:
        #         if j == 'Noun' and len(i) > 1: # and i not in stop_words:  # 1글자 제외, 불용어 제외
        #             all_word.append(i)
        #             user_words[user].append(i)

    # 모든 단어의 빈도 데이터프레임 생성
    all_word_df = pd.DataFrame({'words': all_word, 'count': len(all_word) * [1]})
    all_word_df = all_word_df.groupby('words').count()
    print(all_word_df.sort_values('count', ascending=False))

    # # 사용자별 긍정/부정 점수 출력
    # print("\nUser Sentiment Scores:")
    # for user, sentiments in user_sentiments.items():
    #     print(f"{user}: {sentiments}")

    # 사용자별 단어 리스트 확인
    #print("\nUser-specific Words:")
    #for user, words in user_words.items():
    #    print(f"{user}: {words}")

    return chat_data, log_data


# 감정 분석 함수
def analyze_sentiment_korean(chats):
    results = []
    
    return results

# 시간대별 대화량 분석
def main_chat_time_by_user(chats):
    time_slot = [0] * 24
    for chat in chats:
        time = chat[0]
        hour, _ = map(int, time.split(':'))
        time_slot[hour] += 1
    return time_slot

def calculate_length_score(chat_data, name):
    for user in chat_data:
            if(user==name):
                target_len = len(chat_data[user])
            else:
                other_len = len(chat_data[user])
    print(target_len)
    print(other_len)
    ratio = target_len / other_len
    if ratio <= 0.5:
        return 0
    elif ratio >= 2:
        return 100
    elif 1 <= ratio < 2:
        return round(80 + 20 * (ratio - 1))  # 1~2 사이 선형 증가
    else:  # 0.5 < ratio < 1
        return round(80 - 80 * (1 - ratio))  # 0.5~1 사이 선형 감소

def calculate_response_time_score(chat_data, log_data, name):
    response_times = defaultdict(list)  # 사용자별 응답 시간 저장
    avg_response_times = {}  # 사용자별 평균 응답 시간 저장
    
    for idx in range(len(log_data)):
        if(not idx):
            time_prev = datetime.strptime(log_data[0][1], "%H:%M")
            continue

        user_prev =log_data[idx-1][0]
        user = log_data[idx][0]
        time_str_prev = log_data[idx-1][1]
        time_str = log_data[idx][1]

        if(user == user_prev):
            continue
        
        time_prev = datetime.strptime(time_str_prev, "%H:%M")  # 이전 메시지 시간
        time_curr = datetime.strptime(time_str, "%H:%M")  # 현재 메시지 시간 
        time_delta = (time_curr - time_prev).total_seconds() / 60  # 시간 차이를 분으로 변환

        if time_delta < 0 or time_delta > 300:  # 비정상 시간 데이터 제거
            continue
        response_times[user].append(time_delta)

    for user in chat_data.keys():
        # 평균 응답 시간 계산
        if response_times[user]:  # 응답 시간이 존재할 경우
            avg_response_times[user] = sum(response_times[user]) / len(response_times[user])    
        else:  # 응답 시간이 없을 경우 0으로 설정
            avg_response_times[user] = 0
    
    response_time_score = round(max(0, 100 - (abs(avg_response_times[user] - 0) / 60) * 100))

    return response_time_score


app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        if 'file' not in request.files:
            return "파일이 업로드되지 않았습니다."
        if 'name' not in request.values:
            return "이름이 입력되지 않았습니다."

        file = request.files['file']
        name = request.values['name']

        if file.filename == '':
            return "선택된 파일이 없습니다."

        df = process_file(file)
        (chat_data, log_data) = extract_chat_data(df)

        user_chats = chat_data[name]
        sentiment_results = analyze_sentiment_korean(user_chats)
        time_slot = main_chat_time_by_user(user_chats)

        # 대화량 점수
        length_score = calculate_length_score(chat_data, name)
        print(f'length_score : {length_score}')
        
        # 답장 속도 점수
        response_time_score = calculate_response_time_score(chat_data, log_data, name)
        print(f'response_time_score : {response_time_score}')
        
        # 결과를 result.html 템플릿으로 전달
        return render_template("result.html", sentiment_results=sentiment_results, time_slot=time_slot, length_score=length_score, response_time_score=response_time_score)

if __name__ == '__main__':
    app.run(debug=True)