# In[1] 데이터 로드 및 확인
import pandas as pd
import numpy as np

data = pd.read_csv('./clean_data/도서_일별대출횟수_기상정보_2023.csv', encoding='utf-8')
data.info()
#data.describe()

# In[] 통계 분석
from scipy import stats
from statsmodels.formula.api import ols, glm
red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']
stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False)
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + \
      residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + \
      density + pH + sulphates + alcohol'
regression_result = ols(Rformula, data = data).fit()
print(regression_result.summary())


# In[] 데이터 시각화

# 1) 분포 확인
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.distplot(red_wine_quality, kde = True, color = "red", label = 'red wine')
sns.distplot(white_wine_quality, kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()
plt.show()

# 2) 변수별 상관관계 확인
sm.graphics.plot_partregress_grid(regression_result)
plt.show()

# 특정 변수에 대한 데이터 분포 확인
others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))
p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)
plt.show()

# 2) 변수별 히스토그램 확인
data.hist(bins=50, figsize=(20,15))


# In[2] 데이터 분할


# In[3] 데이터 인코딩



# In[4-1] 데이터 스케일링: MinmaxScaler


# In[4-2] 데이터 스케일링: StandardScaler


# In[5-1] 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# In[5-2] 랜덤서치
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
param_distribs={'C': uniform(loc=0.001, scale=100)}

# In[] 1. 모델: linear regression model
from sklearn.linear_model import LinearRegression

# 그리드 서치 + 교차검증
grid_search=GridSearchCV(LinearRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

# 랜덤 서치 + 교차검증
random_search=RandomizedSearchCV(LinearRegression(), 
                                 param_distributions=param_distribs, cv=5,
                                 n_iter=100, # 랜덤횟수 디폴트=10
                                return_train_score=True)
random_search.fit(X_train, y_train)

# In[]
# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search.cv_results_)
result_grid

# 그리드 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(result_grid['param_C'], result_grid['mean_train_score'], label="Train")
plt.plot(result_grid['param_C'], result_grid['mean_test_score'], label="Test")
plt.legend()

# 랜덤서치 하이퍼파라미터별 상세 결과값
result_random = random_search.cv_results_
pd.DataFrame(result_random)

# 랜덤 서치: 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(result_random['param_C'], result_random['mean_train_score'], label="Train")
plt.plot(result_random['param_C'], result_random['mean_test_score'], label="Test")
plt.legend()


# In[] 2. 모델: decision tree regression

# 그리드 서치 + 교차검증

# 랜덤 서치 + 교차검증

# In[] 3. 모델: random forest regression

# 그리드 서치 + 교차검증

# 랜덤 서치 + 교차검증

# In[] 모델 테스트
from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y_test, pred_test)
MSE = mean_squared_error(y_test, pred_test)
RMSE = np.sqrt(MSE)

# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))

# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(random_search.score(X_test, y_test)))

# In[] 회귀 분석 결과를 산점도로 시각화
coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)

x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']
plot_color = ['r', 'b', 'y', 'g', 'r']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='mpg', data=data_df, ax=axs[row][col], color=plot_color[i])

# In[] 예측하기
