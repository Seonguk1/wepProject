from flask import Flask, request, render_template
import pandas as pd
import re, os, pickle
from datetime import datetime, timedelta
import emoji
from pykospacing import Spacing
from hanspell import spell_checker
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from konlpy.tag import Okt
import matplotlib.pyplot as plt

# 대화 로그에서 날짜를 추출하는 함수
def extract_date_from_log(log_line):
    match = re.match(r"[-]+ (\d{4}년 \d{1,2}월 \d{1,2}일) .+", log_line)
    if match:
        return match.group(1)
    return None

# 데이터를 정리하여 DataFrame으로 변환
def parse_message(logs):
    messages = []
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)
    current_date = None
    
    for log in logs:
        # 날짜 정보가 포함된 로그가 있을 때
        date = extract_date_from_log(log)
        if date:
            current_date = datetime.strptime(date, "%Y년 %m월 %d일")
        
        # 날짜가 최근 30일 내에 있을 때만 처리
        if current_date and current_date >= thirty_days_ago:
            # 대화 로그에서 시간과 메시지를 추출
            match = re.match(r"\[(.+?)\] \[(.+?)\] (.+)", log)
            if match:
                user, time, text = match.groups()
                messages.append({"user": user, "time": time, "text": text, "date": current_date})
    
    return pd.DataFrame(messages)

# 시간대별 대화량을 계산
def main_chat_time_by_user(df):
    time_slots = {user: [0] * 24 for user in df['user'].unique()}

    for _, row in df.iterrows():
        time = row['time']
        user = row['user']

        # 오전/오후 구분을 위해 시간 문자열 처리
        time_parts = time.split(' ')
        period = time_parts[0]  # '오전' 또는 '오후'
        hour = int(time_parts[1].split(':')[0])  # 시각만 추출 (오전/오후 구분이 된 시간)
        
        # 오전/오후에 따른 시간 계산
        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0  # 오전 12시는 0시로 처리

        # 각 유저별로 해당 시간대의 대화량 증가
        time_slots[user][hour] += 1

    return time_slots

# 시간 차이 계산 함수
def calculate_time_differences(df):
    df["time"] = df["time"].str.replace("오전", "AM").str.replace("오후", "PM")
    df["time_diff"] = (
        pd.to_datetime(df["time"], format="%p %I:%M", errors="coerce")
        .diff()
        .dt.total_seconds() / 60
    )
    df["time_diff"] = df["time_diff"].apply(lambda x: x if 0 <= x < 180 else None)
    df["user_diff"] = df["user"].shift() != df["user"]
    df["time_diff"] = df["time_diff"].where(df["user_diff"])
    
    # 유저별 평균 답장 시간 계산
    avg_time_per_user = df.groupby('user')['time_diff'].mean().reset_index(name='avg_time_diff')

    return df, avg_time_per_user 

# 점수 계산 함수 정의
def calculate_scores_by_user(df):
    # 1. 답장 속도 평균 (5점 만점)
    avg_time_per_user = df.groupby('user')['time_diff'].mean()
    avg_time_scores = avg_time_per_user.apply(
        lambda x: 5 if x <= 30 else 4 if x <= 60 else 3 if x <= 120 else 2 if x <= 300 else 1
    )

    # 2. 메시지 길이 평균 × 채팅 개수 (5점 만점)
    message_volume_per_user = df.groupby('user')['text'].apply(lambda texts: texts.str.len().mean() * len(texts))
    message_volume_scores = message_volume_per_user.apply(
        lambda x: 5 if x > 5000 else 4 if x > 3000 else 3 if x > 1500 else 2 if x > 500 else 1
    )
    total_volume = message_volume_per_user.sum()
    volume_ratios = (message_volume_per_user / total_volume).tolist()       


    # 3. 감정 점수: "ㅋㅋ" 또는 "ㅎㅎ"가 포함된 메시지 개수 / 전체 메시지 개수
    df['has_laughter'] = df['text'].apply(lambda text: "ㅋㅋ" in text or "ㅎㅎ" in text)
    laughter_ratio = df.groupby('user')['has_laughter'].mean()
    laughter_scores = laughter_ratio.apply(
        lambda x: 5 if x > 0.7 else 4 if x > 0.5 else 3 if x > 0.3 else 2 if x > 0.1 else 1
    )

    # 4. 이모지 사용 점수: 이모지가 포함된 메시지 개수 / 전체 메시지 개수
    df['has_emoji'] = df['text'].apply(lambda text: any(emoji.is_emoji(char) for char in text) or text.strip() == "이모티콘")
    emoji_ratio = df.groupby('user')['has_emoji'].mean()
    emoji_scores = emoji_ratio.apply(
        lambda x: 5 if x > 0.7 else 4 if x > 0.5 else 3 if x > 0.3 else 2 if x > 0.1 else 1
    )

    print(f"avg_time_per_user : {avg_time_per_user}")
    print(f"avg_length_per_user : {message_volume_per_user}")
    print(f"laughter_ratio : {laughter_ratio}")
    print(f"emoji_ratio : {emoji_ratio}")

    # 유저별 최종 평균 점수 DataFrame 생성
    scores_per_user = pd.DataFrame({
        'speed_score': avg_time_scores,
        'length_score': message_volume_scores,
        'sentiment_score': laughter_scores,
        'emoji_score': emoji_scores
    })

    return scores_per_user, volume_ratios
    

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        if 'file' not in request.files:
            return "파일이 업로드되지 않았습니다."

        file = request.files['file']
        file_lines = file.read().decode('utf-8').splitlines()

        df = parse_message(file_lines)
        
        # 날짜별 대화량 계산
        daily_chat_volume = df.groupby('date').size().reset_index(name='chat_count')

        # 시간 차이 추가
        df, avg_time_per_user = calculate_time_differences(df)

        # 유저별 점수 계산
        scores_per_user, volume_ratios = calculate_scores_by_user(df)

        # 시간대별 대화량 분석
        chat_time_by_user = main_chat_time_by_user(df)
        

        user1_name = df["user"].unique()[0]
        user2_name = df["user"].unique()[1]
        user1_scores = scores_per_user.loc[user1_name].to_dict()
        user2_scores = scores_per_user.loc[user2_name].to_dict()
        
        daily_chat_volume_date = daily_chat_volume['date'].dt.strftime('%Y-%m-%d').tolist()
        daily_chat_volume_count = daily_chat_volume['chat_count'].tolist()
            

        # 각 유저의 시간대별 대화량을 배열로 변환
        user1_time_slots = chat_time_by_user.get(user1_name, [0]*24)
        user2_time_slots = chat_time_by_user.get(user2_name, [0]*24)
        return render_template('result.html',
                               user1_name=user1_name,
                               user2_name=user2_name, 
                               user1_time_slots=user1_time_slots,
                               user2_time_slots=user2_time_slots,
                               daily_chat_volume_date=daily_chat_volume_date,
                               daily_chat_volume_count=daily_chat_volume_count, 
                                user1_scores=user1_scores,
                                user2_scores=user2_scores,
                                user1_ratio=volume_ratios[0],
                                user2_ratio=volume_ratios[1],
                               )

if __name__ == '__main__':
    app.run(debug=True)