import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def scatter_plt_model(model_name, test_y, pred_y):
    plt.scatter(test_y, pred_y, alpha=0.4)
    # X1 = [7,8]
    # Y1 = [7,8]
    plt.plot(color="r", linestyle="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}")
    plt.show()
def print_accuracy(train_score, test_score, pred_y, test_y):
    print("훈련 세트의 정확도 : {:.2f}".format(test_score))
    print("테스트 세트의 정확도 : {:.2f}".format(train_score))
    print("MAE : {:.2f}".format(MAE(pred_y, test_y)))
    print("MAPE : {:.2f}".format(MAPE(pred_y, test_y)))
    print("MSE : {:.2f}".format(MSE(pred_y, test_y)))
    print("RMSE : {:.2f}".format(RMSE(pred_y, test_y)))
    print("MPE : {:.2f}".format(MPE(pred_y, test_y)))

pd.options.display.max_rows = 80
pd.options.display.max_columns = 80
# data pre-processing
df = pd.read_csv('./Process_data.csv')
def data_preprocess(df, strip_str = "%"):
    remove_cols = []
    strip_cols = []
    for elem in df.columns:
        if 'Unnamed' in elem:
            remove_cols.append(elem)
        else:
            if strip_str in df[0][elem]:
                strip_cols.append(elem)
    for elem in remove_cols:
        df.drop(elem, axis = 1)
    for elem in strip_cols:
        df[elem] = df[elem].str.strip(strip_str)
        df[elem] = df[elem].astype('float')
    return df
data_preprocess(df, "%")
#df = df.drop('Unnamed: 66', axis = 1)
#df["x62"] = df['x62'].str.strip("%")
#df["x62"] = df["x62"].astype('float')

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
def feature_corr_evaluate(train_x, corr_val):
    col_len = len(train_x.column)
    train_a = train_x.iloc[:, :col_len / 4]
    train_b = train_x.iloc[:, col_len / 4 : col_len / 2]
    train_c = train_x.iloc[:, col_len / 2 : col_len * 3 / 4]
    train_d = train_x.iloc[:, col_len * 3 / 4 ::]

    corr_a = train_a.corr()
    corr_b = train_b.corr()
    corr_c = train_c.corr()
    corr_d = train_d.corr()

    #상관계수의 절댓값이 0.9 초과인 변수만을 확인
    condition_a = pd.DataFrame(columns=corr_a.columns, index=corr_a.columns) 
    for i in range(0, col_len / 4):
        condition_a.iloc[:,[i]] = corr_a[abs(corr_a.iloc[:,[i]]) > 0.9].iloc[:,[i]]
    condition_b = pd.DataFrame(columns=corr_b.columns, index=corr_b.columns) 
    for i in range(0, col_len / 4):
        condition_b.iloc[:,[i]] = corr_b[abs(corr_b.iloc[:,[i]]) > 0.9].iloc[:,[i]]
    condition_c = pd.DataFrame(columns=corr_c.columns, index=corr_c.columns) 
    for i in range(0, col_len / 4):
        condition_c.iloc[:,[i]] = corr_c[abs(corr_c.iloc[:,[i]]) > 0.9].iloc[:,[i]]
    condition_d = pd.DataFrame(columns=corr_d.columns, index=corr_d.columns) 
    for i in range(0, col_len / 4):
        condition_d.iloc[:,[i]] = corr_d[abs(corr_d.iloc[:,[i]]) > 0.9].iloc[:,[i]]

    # # Plotting packages
    # kdeplot(Kernel Density Estimator plot)은 히스토그램보다 더 부드러운 형태로 분포 곡선을 보여줌
    # sns => seaborn 패키지
    # sns.kdeplot(x=train_x['x1'])
    # sns.kdeplot(x=train_x['x4'],color='r')   #디폴트 색상은 파란색, color='r' 은 붉은색 적용
    # # distplot => 히스토그램과 kdeplot을 같이 그려줌
    # sns.distplot(x=train_x['x13'])
    # sns.distplot(x=train_x['x14'])
    # # violinplot => x축이 feature값, y축이 밀도. feature값의 분포를 보여줌.
    # sns.violinplot(x=train_x['x23'], figsize=(20,20))
    # sns.violinplot(x=train_x['x25'], figsize=(20,20),color='r')

    # train_x['x25'].plot()
    # train_x['x23'].plot()

    # # x축이 겹치지 않도록 회전시키기  
    # plt.xticks(rotation=50)
    # sns.lmplot(x='x39', y='x40', data= train_x)


### 3. 물성 예측을 위한 회귀 모델 적용하기
def scaling_data(scaling_function, train_x, test_x):
    import sklearn
    from sklearn.preprocessing import *
    if scaling_function == 'StandardScaler':
        #Standard Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
        ss=StandardScaler()
        ss.fit(train_x)
        ss_train = ss.transform(train_x)
        ss_test = ss.transform(test_x)
        ss_train = pd.DataFrame(ss_train, columns=train_x.columns, index=train_x.index)
        ss_test = pd.DataFrame(ss_test, columns=test_x.columns, index=test_x.index)
        return ss_train, ss_test
    elif scaling_function == 'MinMaxScaler':
        #Min-Max Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
        ms=MinMaxScaler()
        ms.fit(train_x)
        ms_train = ms.transform(train_x)
        ms_test = ms.transform(test_x)
        ms_train = pd.DataFrame(ms_train, columns=train_x.columns, index=train_x.index)
        ms_test = pd.DataFrame(ms_test, columns=test_x.columns, index=test_x.index)
        return ms_train, ms_test
    elif scaling_function == 'RobustScaler':
        #Robust Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드
        robust=RobustScaler()
        robust.fit(train_x)
        robust_train = robust.transform(train_x)
        robust_test = robust.transform(test_x)
        robust_train = pd.DataFrame(robust_train, columns=train_x.columns, index=train_x.index)
        robust_test = pd.DataFrame(robust_test, columns=test_x.columns, index=test_x.index)
        return robust_train, robust_test
    else:
        print("Scaling Function Input Error, choose among StandardScaler, MinMaxScaler, RobustScaler")
#Train/Test Data set 분리
X_train, X_test = scaling_data('MinMaxScaler', train_x, test_x)
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

#다항 회귀를 수행하는 코드
from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_train)
poly_ftr = poly.transform(X_train)
poly_ftr_test = poly.transform(X_test)

plr = LinearRegression()
plr.fit(poly_ftr, y_train)
plr_predict = plr.predict(poly_ftr_test)

#Ridge 회귀를 수행하는 코드
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_predict = ridge.predict(X_test)

#Lasso 회귀를 수행하는 코드
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_predict = lasso.predict(X_test)

#ElasticNet 회귀를 수행하는 코드
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
Elastic_pred = elasticnet.predict(X_test)

#Support Vector Machine으로 회귀를 수행하는 코드
from sklearn.svm import SVR
svm_regressor = SVR()
svm_regressor.fit(X_train, y_train)
svm_pred = svm_regressor.predict(X_test)

#하이퍼파라미터 튜닝을 통한 모델 성능 향상시키기
#4. 하이퍼파라미터 튜닝
#Polynomial, Ridge, Lasso, ElasticNet Regression, SVR은 하이퍼파라미터 튜닝을 통해 성능 향상을 기대할 수 있습니다.
#직접 하이퍼파라미터를 찾아 설정하는 것은 Manual Search,
################### Manual Search ##########################
# from sklearn.preprocessing import *
# poly = PolynomialFeatures(degree=3)
# poly.fit(X_train)
# poly_ftr = poly.transform(X_train)

# poly_ftr_test = poly.transform(X_test)
# plr = LinearRegression()
# plr.fit(poly_ftr, y_train)
# plr_predict = plr.predict(poly_ftr_test)
#############################################################
#하이퍼파라미터들을 여러 개 정하고 그 중에서 가장 좋은 것을 찾는 알고리즘은 GridSearch라고 합니다.
## 회귀 모델에 grid search를 수행하는 코드
#- `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#    - `estimator`
#    - `param_grid`
#    - `cv`
#    - `n_jobs=-1`
#    - `verbose=2`
#- 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
#- 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.
### GridSearch용 Train/Validation dataset 분리
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False) ##shuffle = True 무작위 추출 False는 순서대로 추출
def model_selection(model_name):
    from sklearn.model_selection import KFold
    if model_name == 'Ridge':
        param_grid = {
            'alpha': [0.1, 0.5, 1.5 , 2, 3, 4]
        }
        estimator = Ridge()
        kf = KFold(
                n_splits=30,
                shuffle=True,
                )
    elif model_name == 'Lasso':
        param_grid = {
            'alpha': [0.0001, 0.001,0.005, 0.02, 0.1, 0.5, 1, 2]
        }
        estimator = Lasso()
        kf = KFold(
                n_splits=30,
                shuffle=True,
                )
    elif model_name == 'ElasticNet':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
            'l1_ratio': [0.01, 0.1, 0.5, 0.7],
        }
        estimator = ElasticNet()
        kf = KFold(
                n_splits=30,
                shuffle=True,
                )
    elif model_name == 'SVR':
        param_grid = {
            'kernel': ['linear','poly','rbf','sigmoid']
        }
        estimator = SVR()
        kf = KFold(
                n_splits=30,
                shuffle=True,
                )
    else:
        print("Model selection Error! Choose among the following: 'Ridge', 'Lasso', 'ElasticNet', 'SVR' ")
        raise(ValueError)
    return param_grid, estimator, kf

def grid_search_params(estimator, param_grid, kf):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )
    grid_search.fit(X_train, y_train) # grid_search 학습
    return grid_search.best_params_ # best parameter 확인
def grid_search_validation_check(model_name, best_params, X_val, y_val, val_threshold = 0.5):
    if model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha = best_params['alpha'])
        ridge.fit(X_train_val, y_train_val)
        ridge_predict = ridge.predict(X_val)
        if ridge.score(X_val,y_val) > val_threshold:
            print("Valid parameters:", best_params)
            return True
        else:
            print("Invalid parameters:", best_params)
            return False
    elif model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha = best_params['alpha'])
        lasso.fit(X_train_val, y_train_val)
        lasso_predict = lasso.predict(X_val)
        if lasso.score(X_val,y_val) > val_threshold:
            print("Valid parameters:", best_params)
            return True
        else:
            print("Invalid parameters:", best_params)
            return False
    elif model_name == 'ElasticNet':
        from sklearn.linear_model import ElasticNet
        elasticnet = ElasticNet(alpha =  best_params['alpha'], l1_ratio = best_params['l1_ratio'])
        elasticnet.fit(X_train_val, y_train_val)
        Elastic_pred = elasticnet.predict(X_val)
        if elasticnet.score(X_val,y_val) > val_threshold:
            print("Valid parameters:", best_params)
            return True
        else:
            print("Invalid parameters:", best_params)
            return False
    elif model_name == 'SVR':
        from sklearn.svm import SVR
        svm_regressor = SVR(kernel = 'poly')
        svm_regressor.fit(X_train_val, y_train_val)
        svm_pred = svm_regressor.predict(X_val)
        if svm_regressor.score(X_val,y_val) > val_threshold:
            print("Valid parameters:", best_params)
            return True
        else:
            print("Invalid parameters:", best_params)
            return False
    else:
        "Wrong model name: select among the following: 'Ridge', 'Lasso', 'ElasticNet', 'SVR'"
        raise(ValueError)
model_name = 'Ridge' # 'Ridge', 'Lasso', 'ElasticNet', 'SVR'
param_grid, estimator, kf = model_selection(model_name)
best_params_ = grid_search_params(estimator, param_grid, kf)
grid_search_validation_check(model_name, best_params_, X_val, y_val, val_threshold = 0.5) # need to rearrange the val_threshold. should be the one without model_selection

#!!! Need validation function for grid search parameters  !!!
#각 모델들을 GridSearch로 찾은 하이퍼파라미터를 validation data set으로 확인하여 유효한지 검증합니다.
#다른 하이퍼파라미터 값도 넣어 train에서 Overfitting이 생겼는지 확인합니다.
# 4. Univariate를 통한 피쳐 셀렉
# 4.1 Train/Validation Data 분리

##shuffle = True 무작위 추출 False는 순서대로 추출

#K선택을 통해 피쳐 셀렉을 수행하는 코드
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression,SelectPercentile, SelectFwe

def best_k_feature_selection(X_train_val, X_train, X_val, train_y):
    r2=[]
    for i in range(0,64):
        selectK = SelectKBest(score_func=f_classif, k=i+1)
        selectK.fit(X_train_val, train_y) # selectK 학습
        X = selectK.transform(X_train) # 학습 데이터 변환
        X_v = selectK.transform(X_val) # 검증 데이터 변환
        lr.fit(X, y_train)
        lr.score(X_v,y_val)
        r2.append(lr.score(X_v,y_val))
    return r2.index(np.max(r2))
    # plt.figure()
    # plt.plot(range(0, 64), r2, label='R2' )
    # plt.xlabel("number of features")
    # plt.ylabel("R2")
    # plt.legend()
    # plt.show()

#K선택을 통해 피쳐 셀렉을 수행하는 코드

#- `SelectKBest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#    - `score_func=f_classif`
#    - `k=9`
#- Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectKBest`를 학습시킵니다.
#- Min-Max Scaler를 통해 변환된 훈련 데이터에 학습된 `SelectKBest`를 적용합니다. (`X`)
#- 훈련, 검증, 테스트 데이터에 학습된 `SelectKBest`를 적용합니다. (`X_tr`, `X_v`, `X_t`)
##val 정확도를 확인하고 셀렉할 피처 개수를 정해줍니다.
best_k = best_k_feature_selection(X_train_val, X_train, X_val, train_y)
selectK = SelectKBest(score_func=f_classif, k=best_k)
selectK.fit(X_train, train_y) # selectK 학습
X_tr = selectK.transform(X_train_val) # 일반 훈련 데이터 변환
X = selectK.transform(X_train) # Min-Max Scaler를 통해 변환된 훈련 데이터를 변환
X_v = selectK.transform(X_val) # 검증 데이터 변환
X_t = selectK.transform(X_test) # 테스트 데이터 변환
TF = selectK.get_support() # get_support통해 어떤 피처가 제거 되었는지 확인할 수 있습니다.
TF = pd.DataFrame(TF, index=X_train.columns)

### 4.3 모델 적용 후 성능 확인하기
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
def scatter_plt_model(model_name, test_y, pred_y):
    plt.scatter(test_y, pred_y, alpha=0.4)
    # X1 = [7,8]
    # Y1 = [7,8]
    plt.plot(color="r", linestyle="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}")
    plt.show()
def print_accuracy(train_score, test_score, pred_y, test_y):
    print("훈련 세트의 정확도 : {:.2f}".format(test_score))
    print("테스트 세트의 정확도 : {:.2f}".format(train_score))
    print("MAE : {:.2f}".format(MAE(pred_y, test_y)))
    print("MAPE : {:.2f}".format(MAPE(pred_y, test_y)))
    print("MSE : {:.2f}".format(MSE(pred_y, test_y)))
    print("RMSE : {:.2f}".format(RMSE(pred_y, test_y)))
    print("MPE : {:.2f}".format(MPE(pred_y, test_y)))

def performance_check(model_name, X, X_t, train_y, test_y, plt_option = False, print_option = False):
    if model_name == 'LinearRegression':
        lr = LinearRegression()
        lr.fit(X, train_y)
        pred_y = lr.predict(X_t)
        train_score = lr.score(X, train_y)
        test_score = lr.score(X_t, test_y)
    elif model_name == 'Polynomial Regression':
        from sklearn.preprocessing import *
        poly = PolynomialFeatures()
        poly.fit(X)
        poly_ftr = poly.transform(X)
        poly_ftr_test = poly.transform(X_t)
        plr = LinearRegression()
        plr.fit(poly_ftr, train_y)
        pred_y = plr.predict(poly_ftr_test)
        pred_y_train = plr.predict(poly_ftr)
        train_score = r2_score(train_y, pred_y_train)
        test_score = r2_score(test_y, pred_y)
    elif model_name == 'Ridge':
        from sklearn.linear_model import RidgePredict
        ridge = Ridge(alpha = 3)
        ridge.fit(X,train_y)
        pred_y = ridge.predict(X_t)
        train_score = ridge.score(X, train_y)
        test_score = ridge.score(X_t, test_y)
    elif model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha = 0.0001)
        lasso.fit(X,train_y)
        pred_y = lasso.predict(X_t)
        train_score = lasso.score(X, train_y)
        test_score = lasso.score(X_t, test_y)
    elif model_name == 'ElasticNet':
        from sklearn.linear_model import ElasticNet
        elasticnet = ElasticNet(alpha =  0.001)
        elasticnet.fit(X, train_y)
        pred_y = elasticnet.predict(X_t)
        train_score = elasticnet.score(X, train_y)
        test_score = elasticnet.score(X_t, test_y)
    elif model_name == 'SVR':
        from sklearn.svm import SVR
        svm_regressor = SVR(kernel = 'linear')
        svm_regressor.fit(X, train_y)
        pred_y = svm_regressor.predict(X_t)
        train_score = svm_regressor.score(X, train_y)
        test_score = svm_regressor.score(X_t, test_y)
    else:
        "Wrong model name: select among the following: 'LinearRegression', 'Polynomial Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR'"
        raise(ValueError)
    if plt_option:
        scatter_plt_model(model_name, test_y, pred_y)
    if print_option:
        print_accuracy(train_score, test_score, pred_y, test_y)


performance_check(model_name, X, X_t, train_y, test_y, True, True)


# Predicting a new result

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
X_train_select=selector.transform(X_train) # Min-Max Scaler로 변환된 훈련 데이터 적용
select=selector.get_support()
X_test_select=selector.transform(X_test) # Min-Max Scaler로 변환된 테스트 데이터 적용

select = pd.DataFrame(select, index=X_train.columns) # Selected ones

performance_check(model_name, X_train_select, X_test_select, train_y, test_y, True, True)

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

performance_check(model_name, X_RFE, X_RFE_t, train_y, test_y, True, True)

## 7. Select K + Outlier Detection
#Univariate 피쳐 셀렉션한 최종 피처의 데이터로 이상치를 제거합니다

train_data_K = X.copy()
train_data_K = pd.DataFrame(train_data_K, index = train_y.index)
train_data_K = pd.concat([train_y,train_data_K], axis=1)

#Isolation Forest를 통해 이상치 제거를 수행하는 코드
from sklearn.ensemble import IsolationForest

def outlier_removal(train_data_K):
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
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers", c="green")
    # Plot x's for the ground truth outliers
    ax.scatter(X_reduce[outlier_index,0], X_reduce[outlier_index,1], X_reduce[outlier_index,2],
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
    return train_x_outlier, train_y_outlier

train_x_outlier, train_y_outlier = outlier_removal(train_data_K)
performance_check(model_name, train_x_outlier, X_t, train_y_outlier, test_y)

## 8. Model Select + RFE + 이상치 제거(Isolation Forest)

#Model Select + RFE 통해 선택한 피쳐 데이터 셋에서 이상치 탐지 및 시각화를 하고 제거합니다.

train_data_RFE = X_RFE.copy()
train_data_RFE = pd.DataFrame(train_data_RFE, index = train_y.index)
train_data_RFE = pd.concat([train_y,train_data_RFE], axis=1)
#Isolation Forest를 통해 이상치 제거를 수행하는 코드
train_x_outlier, train_y_outlier = outlier_removal(train_data_RFE)
performance_check(model_name, train_x_outlier, X_RFE_t, train_y_outlier, test_y)
