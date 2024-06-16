# In[1] 데이터 로드 및 확인
import pandas as pd
import numpy as np

data = pd.read_csv('./clean_data/도서_일별대출횟수_기상정보_2023.csv', encoding='utf-8')
data = pd.DataFrame(data)
data.info()
#data.describe()

# In[] 데이터 인코딩: 요일

data.요일.replace('월', 0, inplace=True)
data.요일.replace('화', 1, inplace=True)
data.요일.replace('수', 2, inplace=True)
data.요일.replace('목', 3, inplace=True)
data.요일.replace('금', 4, inplace=True)
data.요일.replace('토', 5, inplace=True)
data.요일.replace('일', 6, inplace=True)

data.월 = data.월.astype('object')
data.일 = data.일.astype('object')
data.요일 = data.요일.astype('object')

# In[] 데이터 이상치 처리
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 1) 대출인원수 이상치 처리
data.boxplot(column='대출인원수', return_type='both')
Q1_borrow = data['대출인원수'].quantile(q=0.25)
Q3_borrow = data['대출인원수'].quantile(q=0.75)
IQR_borrow = Q3_borrow-Q1_borrow

# 2) 미세먼지농도 이상치 처리
data.boxplot(column='미세먼지농도', return_type='both')
Q1_dust = data['미세먼지농도'].quantile(q=0.25)
Q3_dust = data['미세먼지농도'].quantile(q=0.75)
IQR_dust = Q3_dust-Q1_dust

# 3) 초미세먼지농도 이상치 처리
data.boxplot(column='초미세먼지농도', return_type='both')
Q1_dust_small = data['초미세먼지농도'].quantile(q=0.25)
Q3_dust_small = data['초미세먼지농도'].quantile(q=0.75)
IQR_dust_small = Q3_dust_small-Q1_dust_small

# 4) 평균풍속 이상치 처리
#data.boxplot(column='평균풍속', return_type='both')
#Q1_wind = data['평균풍속'].quantile(q=0.25)
#Q3_wind = data['평균풍속'].quantile(q=0.75)
#IQR_wind = Q3_wind-Q1_wind

# 5) 평균기온 이상치 처리
data.boxplot(column='평균기온', return_type='both')
Q1_temp = data['평균기온'].quantile(q=0.25)
Q3_temp = data['평균기온'].quantile(q=0.75)
IQR_temp = Q3_temp-Q1_temp

# 6) 최고기온 이상치 처리
data.boxplot(column='최고기온', return_type='both')
Q1_temp_h = data['최고기온'].quantile(q=0.25)
Q3_temp_h = data['최고기온'].quantile(q=0.75)
IQR_temp_h = Q3_temp_h-Q1_temp_h

# 7) 최저기온 이상치 처리
data.boxplot(column='최저기온', return_type='both')
Q1_temp_l = data['최저기온'].quantile(q=0.25)
Q3_temp_l = data['최저기온'].quantile(q=0.75)
IQR_temp_l = Q3_temp_l-Q1_temp_l

# 8) 평균습도 이상치 처리
data.boxplot(column='평균습도', return_type='both')
Q1_humi = data['평균습도'].quantile(q=0.25)
Q3_humi = data['평균습도'].quantile(q=0.75)
IQR_humi = Q3_humi-Q1_humi

# 조건 처리
cond_borrow = (data['대출인원수']<Q3_borrow+IQR_borrow*1.5)& (data['대출인원수']>Q1_borrow-IQR_borrow*1.5)
cond_dust = (data['미세먼지농도']<Q3_dust+IQR_dust*1.5)& (data['미세먼지농도']>Q1_dust-IQR_dust*1.5)
cond_dust_small = (data['초미세먼지농도']<Q3_dust_small+IQR_dust_small*1.5)& (data['초미세먼지농도']>Q1_dust_small-IQR_dust_small*1.5)
#cond_wind = (data['평균풍속']<Q3_wind+IQR_wind*1.5)& (data['평균풍속']>Q1_wind-IQR_wind*1.5)
cond_temp = (data['평균기온']<Q3_temp+IQR_temp*1.5)& (data['평균기온']>Q1_temp-IQR_temp*1.5)
cond_temp_h = (data['최고기온']<Q3_temp_h+IQR_temp_h*1.5)& (data['최고기온']>Q1_temp_h-IQR_temp_h*1.5)
cond_temp_l = (data['최저기온']<Q3_temp_l+IQR_temp_l*1.5)& (data['최저기온']>Q1_temp_l-IQR_temp_l*1.5)
cond_humi = (data['평균습도']<Q3_humi+IQR_humi*1.5)& (data['평균습도']>Q1_humi-IQR_humi*1.5)

# 이상치를 뺀 데이터 출력하기
data_IQR=data[cond_borrow & cond_dust & cond_dust_small & cond_temp & cond_temp_h & cond_temp_l & cond_humi]
data_IQR.boxplot(column='대출인원수', return_type='both')

# In[] 데이터 처리: 도봉구 공공도서관으로 모델 구축, 공휴일에 대한 처리
data_dobong = data_IQR[data_IQR.구 == '도봉구']
data_dobong.도서관코드.unique() # 9곳의 도서관

holidays_korea_2023 = [
    '2023-01-01',  # 신정
    '2023-01-21',  # 설날
    '2023-01-22',  # 설날
    '2023-01-23',  # 설날
    '2023-01-04',  # 설날 대체공휴일
    '2023-03-01',  # 삼일절
    '2023-05-05',  # 어린이날
    '2023-05-27',  # 부처님 오신 날
    '2023-05-29',  # 대체공휴일
    '2023-06-06',  # 현충일
    '2023-08-15',  # 광복절
    '2023-09-28',  # 추석 연휴
    '2023-09-29',  # 추석 연휴
    '2023-09-30',  # 추석 연휴
    '2023-10-03',  # 개천절
    '2023-10-09',  # 한글날
    '2023-12-25',  # 크리스마스
]

data_dobong = data_dobong[~data_dobong.날짜.isin(holidays_korea_2023)]
data_dobong.info()

# In[] 도서관별로 데이터를 나누어서 저장하기
filter_data = data_dobong[['도서관코드','월','일','요일','미세먼지농도','초미세먼지농도','평균풍속','평균기온','최고기온','최저기온',
                               '평균습도','강수량','대출인원수']]

grouped_data = filter_data.groupby('도서관코드')

# 딕셔너리로 도서관별 데이터 저장
lib_code_arr = []
lib_data_dict = {}
for library_code, group in grouped_data:
    lib_code_arr.append(library_code)
    lib_data_dict[library_code] = group

# test
for library_code, item in lib_data_dict.items():
    print(f"Library Code: {library_code}")
    print(item.head())  # 각 그룹의 첫 5개 행을 출력

# In[] 데이터 시각화

# 1) 분포 확인
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 2) 변수별 상관관계 확인

# 특정 변수에 대한 데이터 분포와 상관관계 확인
#import statsmodels.api as sm
#3others = list(set(filter_data.columns).difference(set(["대출인원수", "요일"])))
#p, resids = sm.graphics.plot_partregress("대출인원수", "요일", others, data = filter_data[:200], ret_coords = True)
#plt.show()
#filter_data.info()

# 2) 변수별 히스토그램을 통해 도봉구 도서관 데이터 분포 확인
data_dobong.hist(bins=50, figsize=(20,15))

# In[] 3) 특정 도서관에 대해 상관관계 분석
# 상관관계 분석 및 히트맵으로 시각화
heatmap_data = lib_data_dict[lib_code_arr[1]].drop(columns=['도서관코드'])

res_corr = heatmap_data.corr(method = 'pearson')
colormap = plt.cm.RdBu
plt.figure(figsize=(10, 8), dpi=100)
sns.heatmap(res_corr, linewidths = 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = 'white', annot=True,
            annot_kws={"size": 10, "color": 'black'}, 
            fmt='.2f')
plt.title('변수의 상관관계', size=16)
plt.show()

# In[]
# 상관관계 분석 및 산점도로 시각화
sns.pairplot(heatmap_data, hue='대출인원수')

# In[2] 데이터 분할
from sklearn.model_selection import train_test_split
# '구'
X = filter_data[['요일','미세먼지농도','초미세먼지농도','평균기온','최고기온','강수량']]
Y = filter_data[['대출인원수']]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=10)

# In[4-1] 데이터 스케일링: StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

# In[] 1. 모델: linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
 
scores = cross_val_score(lr, X_train, Y_train)
print(scores.mean())

lr.fit(X_train, Y_train)
lr.score(X_train, Y_train)


# In[] 1.2 모델 검증
print("coef_: ", lr.coef_) # 기울기, 컬럼 별 기울기
print("intercept_: ", lr.intercept_) # y절편

pred = lr.predict(X_val)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MAE = mean_absolute_error(Y_val, pred)
MSE = mean_squared_error(Y_val, pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(Y_val, pred)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('R2: ', R2)
print('Test set Score: ', lr.score(X_test, Y_test))

# In[] 1.3. 모델 테스트

pred = lr.predict(X_test[:5])
print(pred.round())

# In[] 회귀 분석 결과를 산점도로 시각화
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)

data_dobong.요일 = data_dobong.요일.astype('float64')
data_dobong.요일.dtype
x_features = ['요일', '미세먼지농도','초미세먼지농도','평균기온','최고기온','강수량']
plot_color = ['r', 'b', 'y', 'g', 'r', 'b']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='대출인원수', data=data_dobong, ax=axs[row][col], color=plot_color[i])
data_dobong.요일 = data_dobong.요일.astype('object')      


# In[] 2. 모델: decision tree regression

# 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid={'max_depth': [10, 20, 30, 40, 50, 100], 'max_features': [2, 3, 4, 5, 6, 7]}


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

# 그리드 서치 + 교차검증
grid_search=GridSearchCV(dtr, param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, Y_train)

# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search.cv_results_)

results_df = pd.DataFrame(result_grid)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)

# In[]
# 그리드 서치: 하이퍼파리미터 값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['max_depth'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_depth'], results_df['mean_test_score'], label="Test")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(results_df['max_features'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_features'], results_df['mean_test_score'], label="Test")
plt.legend()

# In[] 2.2 모델 검증
pred_dtr = grid_search.predict(X_val)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
MAE = mean_absolute_error(Y_val, pred_dtr)
MSE = mean_squared_error(Y_val, pred_dtr)
RMSE = np.sqrt(MSE)
R2 = r2_score(Y_val, pred_dtr)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('R2: ', R2)

# 정확도가 가장 높은 하이퍼파라미터 및 정확도 제시
print('최적 하이퍼 파라미터:\n', grid_search.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_search.best_score_))

# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(grid_search.score(X_val, Y_val)))

# In[] 2.3 테스트 데이터로 모델 예측값 출력하기

# 그리드 서치
pred_dtr = grid_search.predict(X_test[:5])
print(pred_dtr)

# In[] 3. 모델: random forest regression
# 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators': [1, 5, 10, 20, 30], 'max_depth': [2, 3, 4, 5, 6, 7], 'max_features': [2, 3, 4, 5, 6, 7]}

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

# 그리드 서치 + 교차검증
grid_search_rfr=GridSearchCV(rfr, param_grid, cv=5, return_train_score=True)
grid_search_rfr.fit(X_train, Y_train)


# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search_rfr.cv_results_)

results_df = pd.DataFrame(result_grid)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)

# In[]
# 그리드 서치: 하이퍼파리미터(n_estimators)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['n_estimators'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['n_estimators'], results_df['mean_test_score'], label="Test")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(results_df['max_depth'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_depth'], results_df['mean_test_score'], label="Test")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(results_df['max_features'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_features'], results_df['mean_test_score'], label="Test")
plt.legend()
plt.show()

# In[] 3.2 모델 검증
pred_rfr = grid_search_rfr.predict(X_val)

from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(Y_val, pred_rfr)
MSE = mean_squared_error(Y_val, pred_rfr)
RMSE = np.sqrt(MSE)

print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('R2: ', R2)

# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(grid_search_rfr.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search_rfr.best_score_))

# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(grid_search_rfr.score(X_val, Y_val)))

# In[] 3.3 모델 테스트

# 그리드 서치
pred_rfr = grid_search_rfr.predict(X_test[:5])
print(pred_rfr.round())

# In[] 3개의 모델 중 가장 성능이 좋은 모델로 테스트 데이터로 예측하기
# 랜덤 포레스트 회귀분석 모델
pred_rfr = grid_search_rfr.predict(X_test[:5])
print(pred_rfr.round())

# In[] 의사 결정트리 회귀모델 이미지 저장
#import graphviz
import os
from sklearn.tree import export_graphviz

dt_dot_data = tree.export_graphviz(grid_search, out_file = None,
                                  feature_names = ['요일', '미세먼지농도','초미세먼지농도','평균기온','최고기온','강수량'],
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  node_ids=True,
                                  special_characters = True,
                                  fontname="Malgun Gothic")
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)

# 생성된 .dot 파일을 .png로 변환
from subprocess import call
call(['dot', '-Tpng', 'None', '-o', 'decision-tree-regression.png', '-Gdpi=50'])

# In[]
from IPython.display import Image

Image(filename = 'decision-tree-regression.png')

# In[] 랜덤 포레스트 회귀모델 이미지 저장
#import graphviz
import os
estimator = grid_search_rfr.best_estimator_[5]
from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['요일', '미세먼지농도','초미세먼지농도','평균기온','최고기온','강수량'],
                rounded = True, proportion = False, 
                precision = 2, filled = True,
                node_ids=True,
                special_characters=True,
                fontname="Malgun Gothic")

# 생성된 .dot 파일을 .png로 변환
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'random-forest-regression.png', '-Gdpi=50'])

# In[]
from IPython.display import Image

Image(filename = 'random-forest-regression.png')