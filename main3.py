# %%
from sklearn.svm import SVR
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from preprocessing.correlation import Correlation2020, Correlation2021
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

# %%
df_2017 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2017.csv')
df_2018 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2018.csv')
df_2019 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2019.csv')
df_2020 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2020.csv')
df_2021 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2021.csv')
df_details = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/기업정보상세.csv')
df_main = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/main.csv')
df_RND = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/국가R&D과제.csv')
# %%
# %%
df_2017['조사년도'] = 2017
df_2018['조사년도'] = 2018
df_2019['조사년도'] = 2019
df_main_business = df_main[['사업자등록번호_마스킹']]
df_main_business
# %%
df_2019
# 2019
# ---------------------------------------------------------
# %%

drop_2017_df = df_2017.drop(['유형자산 증가율',
                             '총자산 증가율', '매출 증가율',
                             '유동비율', '수출액1', '수출액2',
                             '수출액3', '수출액4', '수출액5', '당좌부채(당좌차월)', '자본잉여금'], axis=1)

drop_2018_df = df_2018.drop(['유형자산 증가율',
                             '총자산 증가율', '매출 증가율',
                             '유동비율', '수출액1', '수출액2', '수출액3', '수출액4', '수출액5', '당좌부채(당좌차월)', '자본잉여금'], axis=1)

drop_2019_df = df_2019.drop(['유형자산 증가율',
                             '총자산 증가율', '매출 증가율',
                             '유동비율', '수출액1', '수출액2', '수출액3', '수출액4', '수출액5', '당좌부채(당좌차월)', '자본잉여금'], axis=1)

# %%
drop_2019_df
# %%
drop_2019_df.isnull().sum().sort_values(ascending=False)
# %%
correlation = Correlation2020()
# %%

df_2017_m = correlation.cc_2020(drop_2017_df)
df_2018_m = correlation.cc_2020(drop_2018_df)
df_2019_m = correlation.cc_2020(drop_2019_df)
# %%
df_2019_m


# %%
df_2017_last = df_2017[['사업자등록번호_마스킹', '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
df_2018_last = df_2018[['사업자등록번호_마스킹', '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
df_2019_last = df_2019[['사업자등록번호_마스킹',  '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본', '매출액']]
df_2019_last
# %%
df_total = df_main_business.merge(df_2017_last, on='사업자등록번호_마스킹', how='left')
df_total = df_total.merge(df_2018_last, on='사업자등록번호_마스킹', how='left')
df_total = df_total.merge(df_2019_last, on='사업자등록번호_마스킹', how='left')
df_total = df_total.merge(
    df_2020[['사업자등록번호_마스킹', '매출액']], on='사업자등록번호_마스킹', how='left')
df_total

# 2019년도 매출액 중위값 기준 하이 로우 분리
# %%
sales = int(df_total['매출액_x'].quantile(0.5))
sales
# %%
high = {}
low = {}
for i in range(len(df_total)):
    if df_total['매출액_x'].iloc[i] > sales:
        high[i] = df_2019_last.iloc[i]
    else:
        low[i] = df_2019_last.iloc[i]

# %%
low
# %%
df_total_high = pd.DataFrame(high).T
df_total_high = pd.merge(
    df_total_high, df_2020[['사업자등록번호_마스킹', '매출액']], on='사업자등록번호_마스킹', how='left')
df_total_high.ffill(inplace=True)
df_total_high_drop = df_total_high.drop(
    ['사업자등록번호_마스킹', '매출액_x'], axis=1)
# %%
df_total_low = pd.DataFrame(low).T
df_total_low = pd.merge(
    df_total_low, df_2020[['사업자등록번호_마스킹', '매출액']], on='사업자등록번호_마스킹', how='left')
df_total_low.ffill(inplace=True)
df_total_low_drop = df_total_high.drop(
    ['사업자등록번호_마스킹', '매출액_x'], axis=1)


# %%
df_total_high
# %%
# Assume a dataset
high_data = df_total_high['매출액_y']
low_data = df_total_low['매출액_y']
# Calculate skewness
print("Skewness: ", stats.skew(high_data))
print("Skewness: ", stats.skew(low_data))
# %%
sns.distplot(df_total_high['매출액_y'])
# %%
sns.distplot(df_total_low['매출액_y'])

# %%

# %%
high_total = np.log1p(df_total_high_drop)
high_total = pd.DataFrame(high_total)
high_total.isnull().sum()

# %%
high_total.ffill(inplace=True)
high_total = np.log1p(high_total)
sns.distplot(high_total)

# %%
low_total = np.log1p(df_total_low_drop)
low_total = pd.DataFrame(low_total)
low_total.isnull().sum()
# %%
low_total.ffill(inplace=True)
low_total = np.log1p(low_total)
sns.distplot(low_total)
# %%

# Calculate Q1, Q2 and IQR
Q1 = high_total['매출액_y'].quantile(0.25)
Q3 = high_total['매출액_y'].quantile(0.75)
IQR = Q3 - Q1

# # Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# %%
# Filter the data frame to remove outliers
filtered_high_total = high_total[(high_total['매출액_y'] >= lower_bound) & (
    high_total['매출액_y'] <= upper_bound)]
# %%
# # Print the shape of the dataframes for comparison
print('Shape before outlier removal:', high_total.shape)
print('Shape after outlier removal:', filtered_high_total.shape)
# %%

filtered_high_total.isnull().sum()
sns.distplot(filtered_high_total['영업외비용'])
# --------------------------------------------------------------


# %%
model = LinearRegression()
x = filtered_high_total[['총자산', '영업외비용', '매출원가',
                         '유동자산', '매출채권', '인건비', '자기자본']]  # 변수설정
y = filtered_high_total['매출액_y']  # 타겟설정
print(x.shape)  # prints: (n_samples, n_features)
print(y.shape)  # prints: (n_samples,)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
# %%
print("Linear regression coeff : {}".format(model.coef_))
print("Linear regression intercept : {}".format(model.intercept_))
# %%
print("training Data evaluation : {}".format(model.score(x_train, y_train)))
print("test Data evaluation : {}".format(model.score(x_test, y_test)))
# %%
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# %%

x_test_df = pd.DataFrame(x_test)
pd.set_option('display.float_format', '{:.2f}'.format)
x_test_df.describe()
# %%
model.fit(x_train, y_train)
model.score(x_test, y_test)
# %%
model = ElasticNet(alpha=0.005, l1_ratio=0.05)
model.fit(x_train, y_train)
# %%
model.score(x_test, y_test)
# %%
# import matplotlib as plt
# import seaborn as sns
# predicted = model.predict(x_test)
# expected = y_test
# plot(expected, predicted)
# %%
RidgeRegre = Ridge(alpha=0.2)
RidgeRegre.fit(x_train, y_train)

LassoRegre = Lasso(alpha=0.2)
LassoRegre.fit(x_train, y_train)

print('training data evaluation {}'.format(RidgeRegre.score(x_train, y_train)))
print('test data evaluation {}'.format(RidgeRegre.score(x_test, y_test)))
# %%
print('training data evaluation {}'.format(LassoRegre.score(x_train, y_train)))
print('test data evaluation {}'.format(LassoRegre.score(x_test, y_test)))
# %%

x = filtered_high_total[['총자산', '영업외비용', '매출원가',
                         '유동자산', '매출채권', '인건비', '자기자본']]  # 변수설정
y = filtered_high_total['매출액_y']  # 타겟설정
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=24)
model = DecisionTreeRegressor(
    max_depth=9, min_samples_split=6, min_samples_leaf=2)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
# %%

param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# %%
best_params = grid_search.best_params_
print("best parameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
# %%
########################
score_df = pd.DataFrame({"y_pred": y_pred[:10], "y_test": y_test.iloc[:10]})
# %%
score_df
# %%

SVM_model = SVR()
SVM_model.fit(x_train, y_train)
# %%
y_pred_svm = SVM_model.predict(x_test)
# %%
('MSE:', mean_squared_error(y_test, y_pred_svm, squared=False))

# %%
