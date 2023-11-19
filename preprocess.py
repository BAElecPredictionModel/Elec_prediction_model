import os
import pandas as pd
import numpy as np

# 데이터 파일 리스트 생성
root_path = os.path.dirname(os.path.abspath(__file__))
data_path = root_path + '/data'
file_list = os.listdir(data_path)
xlsx_list = [file for file in file_list if file[-5:]=='.xlsx']

# 데이터 불러와 concat
df_list = [pd.read_excel(f'{data_path}/{file}') for file in xlsx_list]
df = pd.concat(df_list).iloc[:,1:]

# 시간 feature
df.index = pd.to_datetime(df.Date + df.Time, format="%Y-%m-%d%H:%M")
df.index.name = "Time"
df = df.sort_index()
df = df.drop(['Date', 'Time'], axis=1)

# 시간 벡터화

# 시간 정보 추출
timeIndex = df.index
day = timeIndex.day
days_in_month = timeIndex.days_in_month
month = timeIndex.month

# 벡터화 함수 정의
def vectorize(x): return (np.sin(2*np.pi*x), np.cos(2*np.pi*x))

# Sin, Cos 변환
df['day_month_x'], df['day_month_y'] = vectorize(np.array(day) / np.array(days_in_month))
df['day_year_x'], df['day_year_y'] = vectorize(day / 365)
df['month_x'], df['month_y'] = vectorize(month / 24)

# 요일 원-핫 인코딩
day_of_week = pd.get_dummies(timeIndex.dayofweek, dtype=int, prefix='day_')
day_of_week.index = timeIndex

# concat
df = pd.concat([df, day_of_week], axis=1)
df = df.drop('Day of the week', axis=1)

# 평일 or 휴일 원-핫 인코딩
df['weekdays'] = df['Weekdays or Weekend']=='Weekdays'
df['weekend'] = df['Weekdays or Weekend']=='Weekend'
df.weekdays = df.weekdays.astype(int)
df.weekend = df.weekend.astype(int)
df = df.drop('Weekdays or Weekend', axis=1)

# 풍향 벡터화
df['wd_x'], df['wd_y'] = vectorize(df.wd / 360)
df = df.drop('wd', axis=1)

# 운형 원-핫 인코딩 (중복 허용)
df.clfmAbbrCd = np.where(df.clfmAbbrCd.isnull(), '', df.clfmAbbrCd)
cloud_type = ['Ci', 'Cc', 'Cs', 'Ac', 'As', 'Ns', 'Sc', 'St', 'Cu', 'Cb']
for ct in cloud_type:
    df[f'ct_{ct}'] = df.clfmAbbrCd.str.contains(ct).astype(int)
df = df.drop('clfmAbbrCd', axis=1)

# 가시거리(vs) feature
df.vs = 2000 - df.vs

# 플래그값 기반 결측값 처리: 최근 3시간 평균으로 대체
targets = ['ta', 'rn', 'ws', 'hm', 'ts'] # wd는 결측값이 없으므로 제외, 일조량 플래그 무시
for target in targets:
    df[target] = np.where(df[f'{target}Qcflg']==9, df[target].shift(1).rolling(3).mean(), df[target])

# 플래그 feature 제거
df = df.drop(['taQcflg', 'rnQcflg', 'wsQcflg', 'wdQcflg', 'hmQcflg', 'ssQcflg', 'tsQcflg'], axis=1)

# 24시간 간격으로 resample

# resample method - sum: 전기사용량, 강수량, 일조량, 일사량, 적설량, 운형 원-핫 인코딩(중복 허용이므로 sum 사용)
sum_list = ['AveragePower', 'rn', 'ss', 'icsr', 'dsnw', 
            'ct_Ci', 'ct_Cc', 'ct_Cs', 'ct_Ac', 'ct_As', 'ct_Ns', 'ct_Sc', 'ct_St', 'ct_Cu', 'ct_Cb']

# resample method - mean: 기온, 풍속, 습도, 전운량(10분위), 중하층운량(10분위), 가시거리, 지온, 시간 벡터, 풍향 벡터, 요일 원-핫 인코딩, 평/휴일 원-핫 인코딩
mean_list = ['ws', 'hm', 'dc10Tca', 'dc10LmcsCa', 'vs', 'ts', 'day_month_x', 'day_month_y', 'day_year_x', 'day_year_y', 'month_x', 
             'month_y', 'day__0', 'day__1', 'day__2', 'day__3', 'day__4', 'day__5', 'day__6', 'weekdays', 'weekend', 'wd_x', 'wd_y']

# resample
df_sum = df[sum_list].resample('24H').sum()
df_mean = df[mean_list].resample('24H').mean()
df_resampled = pd.concat([df_sum, df_mean], axis=1)

# 최근 사용량 및 이동평균 추가: 4일, 1주일, 2주일
df_resampled['power_yesterday'] = df_resampled.AveragePower.shift(1)
df_resampled['power_ema4'] = df_resampled.AveragePower.shift(1).ewm(4).mean()
df_resampled['power_ema7'] = df_resampled.AveragePower.shift(1).ewm(7).mean()
df_resampled['power_ema14'] = df_resampled.AveragePower.shift(1).ewm(14).mean()

# 이동평균이 계산되지 않는 시점과 AveragePower(=target value)가 수집되지 않은 시점 삭제
df_resampled = df_resampled.dropna()

# 전처리한 데이터 저장
df_resampled.to_csv(f'{data_path}/preprocessed_data.csv')