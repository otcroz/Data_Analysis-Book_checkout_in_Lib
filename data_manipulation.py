# 폴더 위치는 /기말프로젝트 에 위치시킵니다.
import pandas as pd
import numpy as np

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


lib_borrow = pd.read_csv('./raw_data/전국_공공_대출내역_202312-4.csv', encoding="utf-8")
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

# 해당 연도가 아닌 것 삭제
lib_borrow_seoul['LON_DE'] = pd.to_datetime(lib_borrow_seoul['LON_DE'], errors='coerce')
lib_borrow_seoul = lib_borrow_seoul[lib_borrow_seoul['LON_DE'].dt.year == 2023]

# check
print(lib_borrow_seoul.groupby(['MBER_SEQ_NO_VALUE', 'LON_DE'])['BORROW_COUNT'].count())
print(pd.DataFrame((lib_borrow_seoul.groupby(['MBER_SEQ_NO_VALUE', 'LON_DE'])['BORROW_COUNT'])).shape)

# export
lib_borrow_seoul.to_csv('./clean_data/서울시_공공_대출내역_202312-4.csv', sep=',', index=False, encoding="utf-8-sig")


# In[]