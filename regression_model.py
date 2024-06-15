# In[1] 데이터 로드 및 확인
import pandas as pd
import numpy as np

data = pd.read_csv('./clean_data/도서_일별대출횟수_기상정보_2023.csv', encoding='utf-8')
data = pd.DataFrame(data)
data.info()
#data.describe()

# In[] 데이터 처리: 도봉구 공공도서관으로 모델 구축
data = data[data.구 == '도봉구']
data.도서관코드.unique() # 9곳의 도서관

# In[] 데이터 인코딩: 요일

data.요일.replace('월', 0, inplace=True)
data.요일.replace('화', 1, inplace=True)
data.요일.replace('수', 2, inplace=True)
data.요일.replace('목', 3, inplace=True)
data.요일.replace('금', 4, inplace=True)
data.요일.replace('토', 5, inplace=True)
data.요일.replace('일', 6, inplace=True)

data.요일.astype('object')

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
cond_wind = (data['평균풍속']<Q3_wind+IQR_wind*1.5)& (data['평균풍속']>Q1_wind-IQR_wind*1.5)
cond_temp = (data['평균기온']<Q3_temp+IQR_temp*1.5)& (data['평균기온']>Q1_temp-IQR_temp*1.5)
cond_temp_h = (data['최고기온']<Q3_temp_h+IQR_temp_h*1.5)& (data['최고기온']>Q1_temp_h-IQR_temp_h*1.5)
cond_temp_l = (data['최저기온']<Q3_temp_l+IQR_temp_l*1.5)& (data['최저기온']>Q1_temp_l-IQR_temp_l*1.5)
cond_humi = (data['평균습도']<Q3_humi+IQR_humi*1.5)& (data['평균습도']>Q1_humi-IQR_humi*1.5)

# 이상치를 뺀 데이터 출력하기
data_IQR=data[cond_borrow & cond_dust & cond_dust_small & cond_wind & cond_temp & cond_temp_h & cond_temp_l & cond_humi]
data_IQR.boxplot(column='대출인원수', return_type='both')

# In[] 도서관별로 데이터를 나누어서 저장하기
filter_data = data_IQR[['도서관코드','월','일','요일','미세먼지농도','초미세먼지농도','평균풍속','평균기온','최고기온','최저기온',
                               '평균습도','강수량','대출인원수']]

grouped_data = filter_data.groupby('도서관코드')

# 딕셔너리로 도서관별 데이터셋
lib_data_dict = {}
for library_code, group in grouped_data:
    lib_data_dict[library_code] = group

for library_code, data in lib_data_dict.items():
    print(f"Library Code: {library_code}")
    print(data.head())  # 각 그룹의 첫 5개 행을 출력

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

# 2) 변수별 히스토그램을 통해 데이터 분포 확인
filter_data.hist(bins=50, figsize=(20,15))


# 3) 상관관계 분석
filter_data = filter_data.corr(method = 'pearson', numeric_only=1)
filter_data.info()
print(filter_data)
#data_corr.to_csv('./data_corr.csv', index = False, encoding='utf-8-sig')

# 히트맵 확인
heatmap_data = filter_data
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax
        = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,
        annot_kws = {"size": 10})
plt.show()

sns.set_style('dark')
sns.distplot(red_wine_quality, kde = True, color = "red", label = 'red wine')
sns.distplot(white_wine_quality, kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()
plt.show()

# In[테스트] # 구로 그룹화


heatmap_data = group_a.corr(method = 'pearson', numeric_only=1)
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax
        = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,
        annot_kws = {"size": 10})
plt.show()


# In[] 1. 모델: linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
 
scores = cross_val_score(lr, X_train, Y_train)
print(scores.mean())

lr.fit(X_train, Y_train)


# In[] 1.2 모델 검증

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
MAE = mean_absolute_error(Y_val, pred)
MSE = mean_squared_error(Y_val, pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(Y_val, pred)

print(MAE, MSE, RMSE, R2)

# In[] 1.3. 모델 테스트

pred = lr.predict(X_test)
print(pred_dtr)

# In[] 2. 모델: decision tree regression

# 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid={'max_depth': [10, 20, 30, 40, 50, 100]}

# 랜덤서치
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
param_distribs={'max_depth': [10, 20, 30, 40, 50, 100]}


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

# 그리드 서치 + 교차검증
grid_search=GridSearchCV(dtr, param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, Y_train)

# 랜덤 서치 + 교차검증
random_search=RandomizedSearchCV(dtr, 
                                 param_distributions=param_distribs, cv=5,
                                 n_iter=100, # 랜덤횟수 디폴트=10
                                return_train_score=True)
random_search.fit(X_train, Y_train)

# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search.cv_results_)

results_df = pd.DataFrame(result_grid)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)

# 그리드 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['max_depth'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_depth'], results_df['mean_test_score'], label="Test")
plt.legend()

# 랜덤서치 하이퍼파라미터별 상세 결과값
result_random = random_search.cv_results_

results_df = pd.DataFrame(result_random)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)


# 랜덤 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['max_depth'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['max_depth'], results_df['mean_test_score'], label="Test")
plt.legend()

# In[] 2.2 모델 검증
pred_dtr = grid_search.predict(X_val)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
MAE = mean_absolute_error(Y_val, pred_dtr)
MSE = mean_squared_error(Y_val, pred_dtr)
RMSE = np.sqrt(MSE)
R2 = r2_score(Y_val, pred_dtr)

print(MAE, MSE, RMSE, R2)

# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search.best_score_))

# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(grid_search.score(X_val, Y_val)))

# In[] 2.3 모델 테스트

# 그리드 서치
pred_dtr = grid_search.predict(X_test)
print(pred_dtr)

# 랜덤 서치
pred_dtr = random_search.predict(X_test)
print(pred_dtr)

# In[] 3. 모델: random forest regression
# 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators': [1, 5, 10, 20, 30]}

# 랜덤서치
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
param_distribs={'n_estimators': [1, 5, 10, 20, 30]}


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

# 그리드 서치 + 교차검증
grid_search=GridSearchCV(rfr, param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, Y_train)

# 랜덤 서치 + 교차검증
random_search=RandomizedSearchCV(rfr, 
                                 param_distributions=param_distribs, cv=5,
                                 n_iter=100, # 랜덤횟수 디폴트=10
                                return_train_score=True)
random_search.fit(X_train, Y_train)

# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search.cv_results_)

results_df = pd.DataFrame(result_grid)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)

# 그리드 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['n_estimators'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['n_estimators'], results_df['mean_test_score'], label="Test")
plt.legend()

# 랜덤서치 하이퍼파라미터별 상세 결과값
result_random = random_search.cv_results_

results_df = pd.DataFrame(result_random)
params_df = results_df['params'].apply(pd.Series)
results_df = pd.concat([results_df, params_df], axis=1)


# 랜덤 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(results_df['n_estimators'], results_df['mean_train_score'], label="Train")
plt.plot(results_df['n_estimators'], results_df['mean_test_score'], label="Test")
plt.legend()

# In[] 3.2 모델 검증
from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(Y_val, pred)
MSE = mean_squared_error(Y_val, pred)
RMSE = np.sqrt(MSE)

# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))

# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(random_search.score(X_val, Y_val)))

# In[] 3.3 모델 테스트

# 그리드 서치
pred_rfr = grid_search.predict(X_test)
print(pred_dtr)

# 랜덤 서치
pred_rfr = random_search.predict(X_test)
print(pred_dtr)
# In[] 회귀 분석 결과를 산점도로 시각화
coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)

x_features = ['월', '요일', '평균기온', '평균풍속', '최고기온']
plot_color = ['r', 'b', 'y', 'g', 'r']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='대출인원수', data=data, ax=axs[row][col], color=plot_color[i])

pred_lr = lr.predict(X_test)

pred_rfr = rfr.predict(X_test)
