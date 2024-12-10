from flask import Flask, request, render_template
import pandas as pd
import re
from datetime import datetime, timedelta
import emoji

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
    
    return df 

def calculate_message_volume_by_user(df):
    # 메시지 길이 평균 × 채팅 개수
    message_volume_per_user = df.groupby('user')['text'].apply(lambda texts: texts.str.len().mean() * len(texts))
    total_volume = message_volume_per_user.sum()
    volume_ratios = (message_volume_per_user / total_volume).tolist() 

    return volume_ratios

# 점수 계산 함수 정의
def calculate_scores_by_user(df):
    # 1. 답장 속도 점수: 답장 평균 시간이 20분 이상이면 0점으로 기준
    avg_time_per_user = df.groupby('user')['time_diff'].mean()
    avg_time_scores = avg_time_per_user.apply(
        lambda x: min(max(100 - x*5,0),100)
    )      

    # 2. 콘텐츠 공유 점수: 콘텐츠(사진, 동영상, 파일)를 하루에 2번 이상 보내는 것을 100점으로 기준
    df['has_content_share'] = df['text'].isin(['사진', '동영상']) | df['text'].str.startswith('파일: ')
    content_share_count = df.groupby('user')['has_content_share'].sum()
    content_share_scores = content_share_count.apply(
        lambda x: min(x*5/3,100)
    )


    # 3. 감정 점수: "ㅋㅋ" 또는 "ㅎㅎ"가 포함된 메시지를 하루에 10번 이상 보내는 것을 100점으로 기준
    df['has_laughter'] = df['text'].apply(lambda text: "ㅋㅋ" in text or "ㅎㅎ" in text)
    laughter_ratio = df.groupby('user')['has_laughter'].sum()
    laughter_scores = laughter_ratio.apply(
        lambda x: min(x / 3, 100)
    )

    # 4. 이모지 사용 점수: 이모티콘이 포함된 메시지를 하루에 5번 이상 보내는 것을 100점으로 기준
    df['has_emoji'] = df['text'].apply(lambda text: any(emoji.is_emoji(char) for char in text) or text.strip() == "이모티콘")
    emoji_count = df.groupby('user')['has_emoji'].sum()
    emoji_scores = emoji_count.apply(
        lambda x: min(x*2/3, 100)
    )

    # 5. 애정도 점수: '사랑'이 포함된 메시지를 하루에 1번 이상 보내는 것을 100점으로 기준
    affection_keywords = ['사랑']
    def calculate_affection_score(text):
        return sum(keyword in text for keyword in affection_keywords)

    df['has_affection'] = df['text'].apply(calculate_affection_score)
    affection_count = df.groupby('user')['has_affection'].sum()
    affection_scores = affection_count.apply(
        lambda x: min(x*10/3,100)
    )

    print(f"avg_time_per_user : {avg_time_scores}")
    print(f"content_share : {content_share_scores}")
    print(f"laughter_ratio : {laughter_scores}")
    print(f"emoji_ratio : {emoji_scores}")
    print(f"affection_ratio : {affection_scores}")

    analyze_per_user = pd.DataFrame({
        'avg_time_per_user': avg_time_per_user,
        'content_share_count': content_share_count,
        'laughter_ratio': laughter_ratio,
        'emoji_count': emoji_count,
        'affection_count': affection_count
    })

    scores_per_user = pd.DataFrame({
        'speed_score': avg_time_scores,
        'content_share_score' : content_share_scores,
        'sentiment_score': laughter_scores,
        'emoji_score': emoji_scores,
        'affection_score': affection_scores,
    })

    # 최종 점수 계산 (각 점수의 평균)
    scores_per_user['final_score'] = scores_per_user.mean(axis=1)

    return analyze_per_user, scores_per_user
    

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        if 'file' not in request.files:
            # 파일이 업로드되지 않았을 경우 alert 메시지 반환
            return jsonify({"error": "파일을 업로드해주세요."}), 400

        file = request.files['file']
        if not file.filename:
            # 파일이 선택되지 않았을 경우 alert 메시지 반환
            return jsonify({"error": "파일을 업로드해주세요."}), 400

        file = request.files['file']
        file_lines = file.read().decode('utf-8').splitlines()

        df = parse_message(file_lines)
        unique_users = df["user"].unique()
        if len(unique_users) > 2:
            allowed_users = unique_users[:2]  # 첫 2명의 유저만 유지
            df = df[df["user"].isin(allowed_users)]
        
        # 날짜별 대화량 계산
        daily_chat_volume = df.groupby('date').size().reset_index(name='chat_count')

        # 시간 차이 추가
        df = calculate_time_differences(df)

        # 유저별 점수 계산
        analyze_per_user, scores_per_user = calculate_scores_by_user(df)

        volume_ratios = calculate_message_volume_by_user(df)

        # 시간대별 대화량 분석
        chat_time_by_user = main_chat_time_by_user(df)
        

        user1_name = df["user"].unique()[0]
        user2_name = df["user"].unique()[1]
        analyze_per_user_json = analyze_per_user.to_dict(orient="index")
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
                               analyze_per_user = analyze_per_user_json,
                                user1_scores=user1_scores,
                                user2_scores=user2_scores,
                                user1_ratio=volume_ratios[0],
                                user2_ratio=volume_ratios[1],
                               )
@app.route('/final_score')
def final_score():
    final_score = 85  # 최종 점수, 예시로 85점 설정
    return render_template('final_score.html', score=final_score)

if __name__ == '__main__':
    app.run(debug=True)