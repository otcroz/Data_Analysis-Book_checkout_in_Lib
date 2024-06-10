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






# In[]