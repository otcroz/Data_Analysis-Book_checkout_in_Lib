# 폴더 위치는 /기말프로젝트 에 위치시킵니다.
import pandas as pd
import numpy as np
import os
from datetime import datetime
# In[] 도서관 정보 데이터 가공: 서울시 공공도서관 저장


lib_info = pd.read_csv('./raw_data/전국_공공_도서관_정보.csv')
lib_info_seoul = lib_info[(lib_info.ONE_AREA_CD =='서울특별시') & (lib_info.LBRRY_TY_NM == '공공')]

lib_info_seoul = lib_info_seoul.sort_values(by='LBRRY_CD')

lib_info_seoul_data = lib_info_seoul[['LBRRY_CD', 'LBRRY_NM', 'ONE_AREA_NM', 'TWO_AREA_NM']]

# check
lib_info_seoul_data.info()

# export
lib_info_seoul_data.to_csv('./clean_data/서울시_공공_도서관_정보.csv', sep=',', index=False, encoding="utf-8-sig")


# In[] 도서관 대출내역 불러오기
file_date = '202305-2'
# C:/Users/dbtnd/Downloads/NL_CO_LOAN_PUB_202302-16.csv
lib_borrow = pd.read_csv('C:/Users/dbtnd/Downloads/NL_CO_LOAN_PUB_' + file_date +'.csv', encoding="utf-8")
lib_info_seoul_data = pd.read_csv('./clean_data/서울시_공공_도서관_정보.csv', encoding="utf-8")

lib_borrow.info()

# if 서울시 도서관에서 대출한 내역에 해당하는 데이터
lib_borrow_seoul = lib_borrow[lib_borrow.LBRRY_CD.isin(lib_info_seoul_data.LBRRY_CD)]

lib_borrow_seoul.info()

# 회원번호 grouping, count한 값을 추가 
borrow_counts = lib_borrow_seoul.groupby(['MBER_SEQ_NO_VALUE', 'LON_DE']).size().reset_index(name='BORROW_COUNT')

lib_borrow_seoul = lib_borrow_seoul[['LBRRY_CD', 'MBER_SEQ_NO_VALUE', 'LON_DE']]

lib_borrow_seoul = lib_borrow_seoul.merge(borrow_counts, on=['MBER_SEQ_NO_VALUE', 'LON_DE'], how='left')

# 겹치는 회원번호, 날짜 drop 처리
lib_borrow_seoul = lib_borrow_seoul.drop_duplicates(subset=['MBER_SEQ_NO_VALUE', 'LON_DE'])

# 서울시, 구 추가
lib_borrow_seoul['ONE_AREA_NM'] = lib_info_seoul_data['ONE_AREA_NM']
lib_borrow_seoul['TWO_AREA_NM'] = lib_info_seoul_data['TWO_AREA_NM']

# 해당 연도가 아닌 것 삭제
lib_borrow_seoul['LON_DE'] = pd.to_datetime(lib_borrow_seoul['LON_DE'], errors='coerce')
lib_borrow_seoul = lib_borrow_seoul[lib_borrow_seoul['LON_DE'].dt.year == 2023]

# 연월일 분리
lib_borrow_seoul['YEAR'] = lib_borrow_seoul['LON_DE'].dt.year
lib_borrow_seoul['MONTH'] = lib_borrow_seoul['LON_DE'].dt.month
lib_borrow_seoul['DAY'] = lib_borrow_seoul['LON_DE'].dt.day

# 요일 추가하기

dic_weekday = {'Monday': '월','Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'}

def get_weekday_korean(date_obj):
    weekday_name = date_obj.strftime("%A")
    return dic_weekday[weekday_name]
lib_borrow_seoul['WEEKDAY'] = lib_borrow_seoul['LON_DE'].apply(get_weekday_korean)

# check
#print(lib_borrow_seoul.groupby(['MBER_SEQ_NO_VALUE', 'LON_DE'])['BORROW_COUNT'].count())
print(pd.DataFrame((lib_borrow_seoul.groupby(['MBER_SEQ_NO_VALUE', 'LON_DE'])['BORROW_COUNT'])).shape)

# export
if len(lib_borrow_seoul) > 0:
    lib_borrow_seoul.to_csv('./clean_data/public_library/서울시_공공_대출내역_' + file_date +'.csv', sep=',', index=False, encoding="utf-8-sig")
else: print('no data!')

# In[] 1차 가공했던 도서관 대출 내역 concat, 특정 날짜에 도서관에 대출한 사람 수 grouping

folder_path = './clean_data/public_library'

# 폴더 내의 모든 CSV 파일을 읽어서 데이터프레임으로 합치기
df_list = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)

# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(df_list, ignore_index=True)

# 중복된 행 삭제
combined_df.drop_duplicates(subset=['LBRRY_CD', 'MBER_SEQ_NO_VALUE', 'LON_DE'], inplace=True)

# 결과 출력 (또는 파일로 저장)
print(combined_df)

combined_df_count = combined_df.groupby([
    'LBRRY_CD', 'LON_DE', 'YEAR', 'MONTH', 'DAY', 'WEEKDAY'])['MBER_SEQ_NO_VALUE'].count().reset_index().rename(columns={'MBER_SEQ_NO_VALUE': 'COUNT'})

# 특정 날짜에 대출한 사람 수
lib_borrow_seoul.to_csv('./clean_data/public_library/서울시_공공_대출내역_202312-1.csv', sep=',', index=False, encoding="utf-8-sig")

# In[] 기상요소, 인구밀집 관련 데이터 합치기
precipitation_seoul_data = pd.read_csv('./raw_data/서울시_일별강수량_2021-2023.csv', encoding="cp949")
temp_seoul_data = pd.read_csv('./raw_data/서울시_일별기온_2021-2023.csv', encoding="cp949")
humidity_seoul_data = pd.read_csv('./raw_data/서울시_일별습도_2021-2023.csv', encoding="cp949")
air_poll_seoul_data = pd.read_csv('./raw_data/일별평균대기오염도_2023.csv', encoding="cp949")
person_dens_seoul_data = pd.read_csv('./raw_data/서울시_인구밀도_2021-2023.csv', encoding="utf-8")

## 데이터 가공 ##
# precipitation_seoul_data
precipitation_seoul_data.info()
precipitation_seoul_data = precipitation_seoul_data[['일시', '강수량(mm)']]
# temp_seoul_data
temp_seoul_data.info()
temp_seoul_data = temp_seoul_data[['일시', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]
# humidity_seoul_data
humidity_seoul_data.info()
humidity_seoul_data = humidity_seoul_data[['일시', '평균습도(%rh)']]
# air_poll_seoul_data
air_poll_seoul_data.info()
air_poll_seoul_data = air_poll_seoul_data[['측정일시','측정소명','미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)']]
air_poll_seoul_data.rename(columns={'측정소명': '구'}, inplace=True)
air_poll_seoul_data.측정일시 = pd.to_datetime(air_poll_seoul_data.측정일시, errors='coerce').dt.date
# person_dens_seoul_data
person_dens_seoul_data.info()
person_dens_seoul_data = person_dens_seoul_data[['구', '2023']]
person_dens_seoul_data.rename(columns={'2023': '인구밀도'}, inplace=True)

## 데이터 merge ##
weather_seoul_data = pd.merge(precipitation_seoul_data, temp_seoul_data, on='일시', how='outer')
weather_seoul_data = pd.merge(weather_seoul_data, temp_seoul_data, on='일시', how='outer')
weather_seoul_data = pd.merge(humidity_seoul_data, temp_seoul_data, on='일시', how='outer')

air_person_seoul_data = pd.merge(air_poll_seoul_data, person_dens_seoul_data, on='구', how='outer')
air_person_seoul_data = air_person_seoul_data.loc[0:18249] # 필요없는 데이터 제거

weather_seoul_data.to_csv('./clean_data/기상데이터_2023.csv', sep=',', index=False, encoding="utf-8-sig")
air_person_seoul_data.to_csv('./clean_data/인구밀집_대기_2023.csv', sep=',', index=False, encoding="utf-8-sig")

# In[]
