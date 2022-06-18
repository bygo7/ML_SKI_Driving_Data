import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

pd.options.display.max_rows = 80
pd.options.display.max_columns = 80
# data pre-processing
df = pd.read_csv('./Process_data.csv')
df = df.drop('Unnamed: 66', axis = 1)
df["x62"] = df['x62'].str.strip("%")
df["x62"] = df["x62"].astype('float')

df_date = df['Date']
df = df.set_index("Date")

# splitting train_data and test_data
train_data = df.iloc[0:691,:] #17년 12월 31일
test_data = df.iloc[691:,:] #18년 4월 22일

train_data.isnull().sum()
test_data.isnull().sum()



train_del = train_data.copy()
test_del = test_data.copy()

train_del = train_del.dropna(axis=1)
test_del = test_del.dropna(axis=1)
full_columns=['Y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
       'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
       'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31',
       'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41',
       'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51',
       'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61',
       'x62', 'x63', 'x64']

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean') #'median'을 쓰면 중앙값 사용
imp_mean.fit(train_data) # imp_mean 학습
mean_train = imp_mean.transform(train_data) # 학습된 결측치 대치 적용
mean_test = imp_mean.transform(test_data)  # 학습된 결측치 대치 적용 

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
mean_train = pd.DataFrame(mean_train, columns=full_columns, index = train_data.index) 
mean_test = pd.DataFrame(mean_test, columns=full_columns, index = test_data.index)

from sklearn.impute import KNNImputer

KNN = KNNImputer(n_neighbors =3)
KNN.fit(train_data) # KNN 학습
df_knn_train = KNN.transform(train_data)
df_knn_test = KNN.transform(test_data)

test_data=df_knn_test.copy()
train_data=df_knn_train.copy()

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
#df_knn_train = pd.DataFrame(df_knn_train, columns=full_columns, index = train_data.index)
#df_knn_test = pd.DataFrame(df_knn_test, columns=full_columns, index = test_data.index)
#from impyute.imputation.cs import mice
#df_mice_train=mice(train_data.values)
#df_mice_test=mice(test_data.values)

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
#df_mice_train = pd.DataFrame(df_mice_train, columns=full_columns, index = train_data.index)
#df_mice_test = pd.DataFrame(df_mice_test, columns=full_columns, index = test_data.index)

#from missingpy import MissForest

#miss_imputer = MissForest()
#miss_imputer.fit(train_data) # miss_imputer 학습
#df_miss_train = miss_imputer.transform(train_data)
#df_miss_test = miss_imputer.transform(test_data)

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
#df_miss_train = pd.DataFrame(df_miss_train, columns=full_columns, index = train_data.index)
#df_miss_test = pd.DataFrame(df_miss_test, columns=full_columns, index = test_data.index)

#test_data=df_knn_test.copy()
#train_data=df_knn_train.copy()
#df_knn_train.describe()

train_y = train_data['Y']     # 종속변수 (target variable) Y 분리
train_x = train_data.copy()
del train_x['Y']              # Y를 제거한 나머지 데이터는 독립변수 (input variable) 에 해당
# test data도 마찬가지 작업 수행
test_y = test_data['Y']
test_x = test_data.copy()
del test_x['Y']

# 데이터의 feature에 대한 사전정보가 있다면, 비슷한 카테고리의 feature끼리 나누어서 상관분석을 진행하는 것이 좋음
train_x29=train_x.iloc[:,:29]
train_x46=train_x.iloc[:,29:46]
train_x60=train_x.iloc[:,46:61]
train_x64=train_x.iloc[:,61:64]

# 사전지식이 없다면 corr = train_data.corr(), 즉 모든 feature들을 한번에 상관분석 수행
corr29 = train_x29.corr()
corr46 = train_x46.corr()
corr60 = train_x60.corr()
corr64 = train_x64.corr()
#상관계수의 절댓값이 0.9 초과인 변수만을 확인하는 방법
condition = pd.DataFrame(columns=corr29.columns, index=corr29.columns) 

for i in range(0,29):
    condition.iloc[:,[i]] = corr29[abs(corr29.iloc[:,[i]]) > 0.9].iloc[:,[i]]

# kdeplot(Kernel Density Estimator plot)은 히스토그램보다 더 부드러운 형태로 분포 곡선을 보여줌
# sns => seaborn 패키지
condition
sns.kdeplot(x=train_x['x1'])
sns.kdeplot(x=train_x['x4'],color='r')   #디폴트 색상은 파란색, color='r' 은 붉은색 적용

# distplot => 히스토그램과 kdeplot을 같이 그려줌
sns.distplot(x=train_x['x13'])
sns.distplot(x=train_x['x14'])
# violinplot => x축이 feature값, y축이 밀도. feature값의 분포를 보여줌.
sns.violinplot(x=train_x['x23'], figsize=(20,20))
sns.violinplot(x=train_x['x25'], figsize=(20,20),color='r')

train_x['x25'].plot()
train_x['x23'].plot()

# x축이 겹치지 않도록 회전시키기  
plt.xticks(rotation=50)
condition = pd.DataFrame(columns=corr46.columns, index=corr46.columns)
# corr46에서 상관계수의 절댓값이 0.9 초과인 피처들만 확인하는 코드
for i in range(0,17):
    condition.iloc[:,[i]] = corr46[abs(corr46.iloc[:,[i]]) > 0.9].iloc[:,[i]]

sns.lmplot(x='x39', y='x40', data= train_x)
sns.violinplot(x=train_x['x39'], figsize=(20,20))
sns.violinplot(x=train_x['x40'], figsize=(20,20),color='r')
sns.violinplot(x=train_x['x44'], figsize=(20,20))
sns.violinplot(x=train_x['x45'], figsize=(20,20),color='r')
sns.distplot(x=train_x['x39'])
sns.distplot(x=train_x['x40'],color='r')

condition = pd.DataFrame(columns=corr60.columns, index=corr60.columns)
#`corr60`에서 상관계수의 절댓값이 0.9 초과인 피처들만 확인하는 코드
for i in range(0,corr60.shape[1]):
    condition.iloc[:,[i]] = corr60[abs(corr60.iloc[:,[i]]) > 0.9].iloc[:,[i]]

sns.kdeplot(x=train_x['x53'])
sns.kdeplot(x=train_x['x60'],color='r')

sns.kdeplot(x=train_x['x54'])
sns.kdeplot(x=train_x['x55'],color='r')
sns.kdeplot(x=train_x['x57'], color='g')

condition = pd.DataFrame(columns=corr64.columns, index=corr64.columns)
# corr64에서 상관계수의 절댓값이 0.7 초과인 피처들만 확인하는 코드
for i in range(0,corr64.shape[1]):
    condition.iloc[:,[i]] = corr64[abs(corr64.iloc[:,[i]]) > 0.7].iloc[:,[i]]

sns.kdeplot(x=train_x['x62'])
sns.kdeplot(x=train_x['x63'],color='r')

### 3. 물성 예측을 위한 회귀 모델 적용하기
#Standard Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
import sklearn
from sklearn.preprocessing import *

ss=StandardScaler()
ss.fit(train_x)
ss_train = ss.transform(train_x)
ss_test = ss.transform(test_x)

ss_train = pd.DataFrame(ss_train, columns=train_x.columns, index=train_x.index)
ss_test = pd.DataFrame(ss_test, columns=test_x.columns, index=test_x.index)

#Min-Max Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
ms=MinMaxScaler()
ms.fit(train_x)
ms_train = ms.transform(train_x)
ms_test = ms.transform(test_x)


ms_train = pd.DataFrame(ms_train, columns=train_x.columns, index=train_x.index)
ms_test = pd.DataFrame(ms_test, columns=test_x.columns, index=test_x.index)

#Robust Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
robust=RobustScaler()
robust.fit(train_x)
robust_train = robust.transform(train_x)
robust_test = robust.transform(test_x)

robust_train = pd.DataFrame(robust_train, columns=train_x.columns, index=train_x.index)
robust_test = pd.DataFrame(robust_test, columns=test_x.columns, index=test_x.index)

#Train/Test Data set 분리
# MinMaxScaler 로 정규화된 데이터를 사용합니다.
X_train = ms_train.copy()
X_test = ms_test.copy()
y_train = train_y.copy()
y_test = test_y.copy()
#### 3.3 평가지표 구현하기
def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test-y_pred))
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
def MSE(y_test, y_pred):
    return np.mean(np.square(y_test-y_pred))
def RMSE(y_test, y_pred):
    return np.sqrt(np.mean(np.square(y_test-y_pred))) 
def MPE(y_test, y_pred):
    return np.mean((y_test-y_pred)/y_test)*100

#피쳐를 제거하지 않은 데이터에 회귀 모델 적용하기
#선형 회귀를 수행하는 코드
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
linear_r2 = r2_score(y_test,lr_predict)
plt.scatter(y_test, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Linear REGRESSION")
plt.show()
print("linear훈련 세트의 정확도 : {:.2f}".format(lr.score(X_train,y_train)))
print("linear테스트 세트의 정확도 : {:.2f}".format(lr.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

#다항 회귀를 수행하는 코드
from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_train)
poly_ftr = poly.transform(X_train)
poly_ftr_test = poly.transform(X_test)

plr = LinearRegression()
plr.fit(poly_ftr, y_train)
plr_predict = plr.predict(poly_ftr_test)

plt.scatter(y_test, plr_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(y_test,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


#Ridge 회귀를 수행하는 코드
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_predict = ridge.predict(X_test)
plt.scatter(y_test, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train,y_train)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

#Lasso 회귀를 수행하는 코드
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_predict = lasso.predict(X_test)
plt.scatter(y_test, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train,y_train)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

#ElasticNet 회귀를 수행하는 코드
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
Elastic_pred = elasticnet.predict(X_test)
plt.scatter(y_test, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train,y_train)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

#Support Vector Machine으로 회귀를 수행하는 코드
from sklearn.svm import SVR
svm_regressor = SVR()
svm_regressor.fit(X_train, y_train)
svm_pred = svm_regressor.predict(X_test)
plt.scatter(y_test, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train,y_train)))
print("svm테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

#[프로젝트5] 하이퍼파라미터 튜닝을 통한 모델 성능 향상시키기
#4. 하이퍼파라미터 튜닝
#Polynomial, Ridge, Lasso, ElasticNet Regression, SVR은 하이퍼파라미터 튜닝을 통해 성능 향상을 기대할 수 있습니다.

#직접 하이퍼파라미터를 찾아 설정하는 것은 Manual Search,

#하이퍼파라미터들을 여러 개 정하고 그 중에서 가장 좋은 것을 찾는 알고리즘은 GridSearch라고 합니다.

#Ridge 회귀 모델에 grid search를 수행하는 코드
#- `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#    - `estimator`
#    - `param_grid`
#    - `cv`
#    - `n_jobs=-1`
#    - `verbose=2`
#- 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
#- 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

## Ridge GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.1, 0.5, 1.5 , 2, 3, 4]
}

estimator = Ridge()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )
grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인

#각 모델들을 GridSearch로 찾은 하이퍼파라미터를 validation data set으로 확인하여 유효한지 검증합니다.
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train_val,y_train_val)
ridge_predict = ridge.predict(X_val)
plt.scatter(y_val, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_val,y_train_val)))
print("ridge검증 세트의 정확도 : {:.2f}".format(ridge.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(ridge_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,y_val)))
print("MSE : {:.2f}".format(MSE(ridge_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,y_val)))
print("MPE : {:.2f}".format(MPE(ridge_predict,y_val)))

#다른 하이퍼파라미터 값도 넣어 train에서 Overfitting이 생겼는지 확인합니다.
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X_train_val,y_train_val)
ridge_predict = ridge.predict(X_val)
plt.scatter(y_val, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_val,y_train_val)))
print("ridge검증 세트의 정확도 : {:.2f}".format(ridge.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(ridge_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,y_val)))
print("MSE : {:.2f}".format(MSE(ridge_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,y_val)))
print("MPE : {:.2f}".format(MPE(ridge_predict,y_val)))

##Lasso 회귀 모델에 grid search를 수행하는 코드
#- `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#    - `estimator`
#    - `param_grid`
#    - `cv`
#    - `n_jobs=-1`
#    - `verbose=2`
#- 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
#- 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

## Lasso GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.0001, 0.001,0.005, 0.02, 0.1, 0.5, 1, 2]
}

estimator = Lasso()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )
grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)
lasso.fit(X_train_val,y_train_val)
lasso_predict = lasso.predict(X_val)
plt.scatter(y_val, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_val,y_train_val)))
print("lasso검증 세트의 정확도 : {:.2f}".format(lasso.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(lasso_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,y_val)))
print("MSE : {:.2f}".format(MSE(lasso_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,y_val)))
print("MPE : {:.2f}".format(MPE(lasso_predict,y_val)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_train_val,y_train_val)
lasso_predict = lasso.predict(X_val)
plt.scatter(y_val, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_val,y_train_val)))
print("lasso검증 세트의 정확도 : {:.2f}".format(lasso.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(lasso_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,y_val)))
print("MSE : {:.2f}".format(MSE(lasso_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,y_val)))
print("MPE : {:.2f}".format(MPE(lasso_predict,y_val)))

#ElasticNet 회귀 모델에 grid search를 수행하는 코드

#- `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#    - `estimator`
#    - `param_grid`
#    - `cv`
#    - `n_jobs=-1`
#    - `verbose=2`
#- 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
#- 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

## ElasticNet GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
    'l1_ratio': [0.01, 0.1, 0.5, 0.7],
}

estimator = ElasticNet()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )

grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001, l1_ratio = 0.01)
elasticnet.fit(X_train_val, y_train_val)
Elastic_pred = elasticnet.predict(X_val)
plt.scatter(y_val, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_val,y_train_val)))
print("Elastic검증 세트의 정확도 : {:.2f}".format(elasticnet.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,y_val)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,y_val)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,y_val)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_train_val, y_train_val)
Elastic_pred = elasticnet.predict(X_val)
plt.scatter(y_val, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_val,y_train_val)))
print("Elastic검증 세트의 정확도 : {:.2f}".format(elasticnet.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,y_val)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,y_val)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,y_val)))

#Support Vector Machine 회귀 모델에 grid search를 수행하는 코드
param_grid = {
    'kernel': ['linear','poly','rbf','sigmoid']
}
estimator = SVR()

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )

grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'poly')
svm_regressor.fit(X_train_val, y_train_val)
svm_pred = svm_regressor.predict(X_val)
plt.scatter(y_val, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_val,y_train_val)))
print("svm검증 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(svm_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,y_val)))
print("MSE : {:.2f}".format(MSE(svm_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,y_val)))
print("MPE : {:.2f}".format(MPE(svm_pred,y_val)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_train_val, y_train_val)
svm_pred = svm_regressor.predict(X_val)
plt.scatter(y_val, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_val,y_train_val)))
print("svm검증 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(svm_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,y_val)))
print("MSE : {:.2f}".format(MSE(svm_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,y_val)))
print("MPE : {:.2f}".format(MPE(svm_pred,y_val)))

# 4. Univariate를 통한 피쳐 셀렉
# 4.1 Train/Validation Data 분리
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(ms_train, train_y, test_size=0.2, shuffle=False) 
##shuffle = True 무작위 추출 False는 순서대로 추출

#K선택을 통해 피쳐 셀렉을 수행하는 코드
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression,SelectPercentile, SelectFwe


r2=[]
for i in range(0,64):
    selectK = SelectKBest(score_func=f_classif, k=i+1)
    selectK.fit(ms_train, train_y) # selectK 학습
    X = selectK.transform(X_train) # 학습 데이터 변환
    X_v = selectK.transform(X_val) # 검증 데이터 변환
    lr.fit(X, y_train)
    lr.score(X_v,y_val)
    r2.append(lr.score(X_v,y_val))
plt.figure()
plt.plot(range(0, 64), r2, label='R2' )
plt.xlabel("number of features")
plt.ylabel("R2")
plt.legend()
plt.show()
#Validation 정확도 최대값 확인
np.max(r2)
#해당 피처 개수 확인
r2.index(np.max(r2))
#피처 개수별 val 정확도 확인
r2

#K선택을 통해 피쳐 셀렉을 수행하는 코드

#- `SelectKBest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#    - `score_func=f_classif`
#    - `k=9`
#- Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectKBest`를 학습시킵니다.
#- Min-Max Scaler를 통해 변환된 훈련 데이터에 학습된 `SelectKBest`를 적용합니다. (`X`)
#- 훈련, 검증, 테스트 데이터에 학습된 `SelectKBest`를 적용합니다. (`X_tr`, `X_v`, `X_t`)
##val 정확도를 확인하고 셀렉할 피처 개수를 정해줍니다. 
selectK = SelectKBest(score_func=f_classif, k=9)
selectK.fit(ms_train, train_y) # selectK 학습
X_tr = selectK.transform(X_train) # 일반 훈련 데이터 변환
X = selectK.transform(ms_train) # Min-Max Scaler를 통해 변환된 훈련 데이터를 변환
X_v = selectK.transform(X_val) # 검증 데이터 변환
X_t = selectK.transform(ms_test) # 테스트 데이터 변환
TF = selectK.get_support() # get_support통해 어떤 피처가 제거 되었는지 확인할 수 있습니다.
TF = pd.DataFrame(TF, index=ms_train.columns)

### 4.3 모델 적용 후 성능 확인하기
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X, train_y)
lr_predict = lr.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X)
poly_ftr = poly.transform(X)

poly_ftr_test = poly.transform(X_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X,train_y)
ridge_predict = ridge.predict(X_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X,train_y)
lasso_predict = lasso.predict(X_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X, train_y)
Elastic_pred = elasticnet.predict(X_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

# 5. Select From Model  피쳐 제거 방법

#Select From Model을 통해 피쳐 셀렉을 수행하는 코드
#- `SelectFromModel` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#    - `estimator=LinearRegression()`
#- Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectFromModel`를 학습시킵니다.
#- Min-Max Scaler를 통해 변환된 훈련 데이터에 학습된 `SelectFromModel`를 적용합니다.
#- Min-Max Scaler를 통해 변환된 테스트 데이터에 학습된 `SelectFromModel`를 적용합니다.

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

selector = SelectFromModel(estimator=LinearRegression())
selector.fit(train_x, train_y) # selector 학습
selector.estimator_.coef_
X_train_select=selector.transform(ms_train) # Min-Max Scaler로 변환된 훈련 데이터 적용
select=selector.get_support()
X_test_select=selector.transform(ms_test) # Min-Max Scaler로 변환된 테스트 데이터 적용

select = pd.DataFrame(select, index=ms_train.columns)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X_train_select, train_y)
lr_predict = lr.predict(X_test_select)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X_train_select,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_test_select,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_train_select)
poly_ftr = poly.transform(X_train_select)

poly_ftr_test = poly.transform(X_test_select)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X_train_select,train_y)
ridge_predict = ridge.predict(X_test_select)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_select,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_train_select,train_y)
lasso_predict = lasso.predict(X_test_select)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_select,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_train_select, train_y)
Elastic_pred = elasticnet.predict(X_test_select)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_select,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_train_select, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_test_select)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_select,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_test_select,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

## 6. MODEL SELECT + RFE
#RFE를 통해 피쳐 셀렉을 수행하는 코드
#- `RFE` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#    - `model`
#    - `6`
#- Model select를 통해 변환된 훈련 데이터(`X_train_select`)로 `RFE`를 학습시킵니다.
#- Model select를 통해 변환된 훈련 데이터(`X_train_select`)에 학습된 `RFE`를 적용합니다.
#- Model select를 통해 변환된 테스트 데이터(`X_test_select`)에 학습된 `RFE`를 적용합니다.

from sklearn.feature_selection import RFE

model = LinearRegression()
rfe = RFE(model, 6)
rfe.fit(X_train_select, train_y) # rfe 학습
X_RFE = rfe.transform(X_train_select) # X_train_select 적용
X_RFE_t = rfe.transform(X_test_select) # X_test_select 적용


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X_RFE, train_y)
lr_predict = lr.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X_RFE,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_RFE)
poly_ftr = poly.transform(X_RFE)

poly_ftr_test = poly.transform(X_RFE_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(X_RFE,train_y)
ridge_predict = ridge.predict(X_RFE_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_RFE,train_y)
lasso_predict = lasso.predict(X_RFE_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_RFE, train_y)
Elastic_pred = elasticnet.predict(X_RFE_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_RFE, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

## 7. Select K + Outlier Detection
#Univariate 피쳐 셀렉션한 최종 피처의 데이터로 이상치를 제거합니다

selectK = SelectKBest(score_func=f_classif, k=12)
selectK.fit(ms_train, train_y)
X_tr = selectK.transform(X_train)
X = selectK.transform(ms_train)
X_v = selectK.transform(X_val)
X_t = selectK.transform(ms_test)
TF = selectK.get_support()
TF = pd.DataFrame(TF, index=ms_train.columns)

train_data_K = X.copy()
train_data_K = pd.DataFrame(train_data_K, index = train_y.index)
train_data_K = pd.concat([train_y,train_data_K], axis=1)

#Isolation Forest를 통해 이상치 제거를 수행하는 코드
from sklearn.ensemble import IsolationForest

# 이상치 제거 isolation forest model 설정
clf=IsolationForest(n_estimators=100, contamination = 0.01)

#이상치 제거 피팅
clf.fit(train_data_K) # train_data_K로 clf 학습
pred = clf.predict(train_data_K)
train_data_K['anomaly']=pred
train_data_K = train_data_K.reset_index()
outliers=train_data_K.loc[train_data_K['anomaly']==-1]
outlier_index=list(outliers.index)
print(train_data_K['anomaly'].value_counts())

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
scaler = StandardScaler()
#normalize the metrics
train_data_K.set_index('Date', inplace = True)
X = scaler.fit_transform(train_data_K)
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()

# 이상치로 판단된 데이터를 제거합니다.
idx_nm_1 = train_data_K[train_data_K['anomaly'] == -1].index
train_data_outlier = train_data_K.drop(idx_nm_1)
del train_data_outlier['anomaly']

train_y_outlier = train_data_outlier['Y']
train_x_outlier = train_data_outlier.copy()
del train_x_outlier['Y']
train_data_outlier_index = train_data_outlier.index
test_data_index = test_data.index

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(train_x_outlier, train_y_outlier)
lr_predict = lr.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(train_x_outlier)
poly_ftr = poly.transform(train_x_outlier)

poly_ftr_test = poly.transform(X_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y_outlier)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(train_x_outlier,train_y_outlier)
ridge_predict = ridge.predict(X_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(train_x_outlier,train_y_outlier)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(train_x_outlier,train_y_outlier)
lasso_predict = lasso.predict(X_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(train_x_outlier,train_y_outlier)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(train_x_outlier, train_y_outlier)
Elastic_pred = elasticnet.predict(X_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(train_x_outlier,train_y_outlier)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(train_x_outlier, train_y_outlier)

# Predicting a new result
svm_pred = svm_regressor.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

## 8. Model Select + RFE + 이상치 제거(Isolation Forest)

#Model Select + RFE 통해 선택한 피쳐 데이터 셋에서 이상치 탐지 및 시각화를 하고 제거합니다.

train_data_RFE = X_RFE.copy()
train_data_RFE = pd.DataFrame(train_data_RFE, index = train_y.index)
train_data_RFE = pd.concat([train_y,train_data_RFE], axis=1)
#Isolation Forest를 통해 이상치 제거를 수행하는 코드
from sklearn.ensemble import IsolationForest

# 이상치 제거 isolation forest model 설정
clf=IsolationForest(n_estimators=100, contamination = 0.01)

	#이상치 제거 피팅
clf.fit(train_data_RFE) # train_data_RFE로 clf 훈련
pred = clf.predict(train_data_RFE)
train_data_RFE['anomaly']=pred
train_data_RFE = train_data_RFE.reset_index()
outliers=train_data_RFE.loc[train_data_RFE['anomaly']==-1]
outlier_index=list(outliers.index)
print(train_data_RFE['anomaly'].value_counts())

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
scaler = StandardScaler()
#normalize the metrics
train_data_RFE.set_index('Date', inplace = True)
X = scaler.fit_transform(train_data_RFE)
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()

idx_nm_1 = train_data[train_data_RFE['anomaly'] == -1].index
train_data_outlier = train_data_RFE.drop(idx_nm_1)
del train_data_outlier['anomaly']

train_y_outlier = train_data_outlier['Y']
train_x_outlier = train_data_outlier.copy()
del train_x_outlier['Y']
train_data_outlier_index = train_data_outlier.index
test_data_index = test_data.index

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(train_x_outlier, train_y_outlier)
lr_predict = lr.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))

from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(train_x_outlier)
poly_ftr = poly.transform(train_x_outlier)

poly_ftr_test = poly.transform(X_RFE_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y_outlier)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(train_x_outlier,train_y_outlier)
ridge_predict = ridge.predict(X_RFE_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(train_x_outlier,train_y_outlier)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(train_x_outlier,train_y_outlier)
lasso_predict = lasso.predict(X_RFE_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(train_x_outlier,train_y_outlier)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(train_x_outlier, train_y_outlier)
Elastic_pred = elasticnet.predict(X_RFE_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(train_x_outlier,train_y_outlier)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))

from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(train_x_outlier, train_y_outlier)

# Predicting a new result
svm_pred = svm_regressor.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))

