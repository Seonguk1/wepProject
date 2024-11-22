from flask import Flask, request, render_template
import pandas as pd
from konlpy.tag import Okt
from datetime import datetime
from collections import defaultdict
twt = Okt()

app = Flask(__name__, template_folder='.')

# 파일 처리 함수
def process_file(file):
    content = file.read().decode('utf-8')
    lines = content.strip().split('\n')
    data = []

    max_fields = 0  # 최대 필드 개수를 찾기 위한 초기값

    for line in lines:
        fields = line.split()  # 공백 기준 분할
        max_fields = max(max_fields, len(fields))
        data.append(fields)

    # 모든 행의 필드 수를 최대 필드 개수에 맞춤
    for fields in data:
        fields.extend([''] * (max_fields - len(fields)))

    # DataFrame 변환
    df = pd.DataFrame(data, columns=[f'col{i+1}' for i in range(max_fields)])
    return df

# 사용자별 채팅 데이터 분리 함수
def extract_chat_data(df):
    from collections import defaultdict
    
    chat_data = {}
    log_data = []

    user_words = defaultdict(list)  # 사용자별 단어 저장
    user_sentiments = defaultdict(lambda: {'positive': 0, 'negative': 0})  # 긍정/부정 단어 빈도 저장

    all_word = []
    stop_words = ['그리고', '이것', '그것', '저것','이거','그거','저거','이모티콘','이제','사진', '혹시', '이번', '저번',
                   '보통', '먼저', '오늘', '내일', '어제', '오전', '오후', '잠깐', '일찍', '정도', '이제', '다시', '바로', '대신', '거의', '어디','']
    positive_words = ['좋다', '훌륭', '기쁘다', '멋지다', '사랑', '행복']  # 추가 가능
    negative_words = ['나쁘다', '싫다', '짜증', '슬프다', '화난다', '불행']  # 추가 가능

    for _, row in df.iterrows():
        user = row.iloc[0]  # 첫 번째 열: 사용자 이름
        am_pm = row.iloc[1]  # 두 번째 열: 오전/오후
        time = row.iloc[2]  # 세 번째 열: 시간
        message = ' '.join(map(str, row[3:])).strip()  # 나머지 열: 메시지 내용

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

        
        # 사용자별 채팅 데이터 저장
        if user not in chat_data:
            chat_data[user] = []
        chat_data[user].append([formatted_time, message])

        # 채팅 로그 순서대로 데이터 저장
        log_data.append([user, formatted_time, message])

        # 메시지에서 명사 추출 및 분석
        words = twt.pos(message)
        for i, j in words:
            if j == 'Noun' and len(i) > 1 and i not in stop_words:  # 1글자 제외, 불용어 제외
                all_word.append(i)
                user_words[user].append(i)
            
            # 긍정/부정 단어 빈도 체크
            if i in positive_words:
                user_sentiments[user]['positive'] += 1
            elif i in negative_words:
                user_sentiments[user]['negative'] += 1

    # 모든 단어의 빈도 데이터프레임 생성
    all_word_df = pd.DataFrame({'words': all_word, 'count': len(all_word) * [1]})
    all_word_df = all_word_df.groupby('words').count()
    print(all_word_df.sort_values('count', ascending=False))

    # 사용자별 긍정/부정 점수 출력
    print("\nUser Sentiment Scores:")
    for user, sentiments in user_sentiments.items():
        print(f"{user}: {sentiments}")

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

        if name not in chat_data:
            return f"'{name}'라는 사용자의 데이터를 찾을 수 없습니다."

        user_chats = chat_data[name]
        sentiment_results = analyze_sentiment_korean(user_chats)
        time_slot = main_chat_time_by_user(user_chats)

        # 대화량 점수
        length_score = calculate_length_score(chat_data, name)

        # 답장 속도 점수
        response_time_score = calculate_response_time_score(chat_data, log_data, name)
        
        # 결과를 result.html 템플릿으로 전달
        return render_template("result.html", sentiment_results=sentiment_results, time_slot=time_slot, length_score=length_score, response_time_score=response_time_score)

if __name__ == '__main__':
    app.run(debug=True)

