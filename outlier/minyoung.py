# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from preprocessing.correlation import Correlation2020, Correlation2021
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing

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
df_2017['조사년도'] = 2017
df_2018['조사년도'] = 2018
df_2019['조사년도'] = 2019

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
df_2017_last = df_2017[['사업자등록번호_마스킹', '조사년도', '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
df_2018_last = df_2018[['사업자등록번호_마스킹', '조사년도', '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
df_2019_last = df_2019[['사업자등록번호_마스킹', '조사년도', '총자산',
                        '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
# %%
df_2019_last

# %%
df_details_2 = df_details[['사업자등록번호', 'CRI등급']]
df_details_2.columns = ['사업자등록번호_마스킹', 'CRI등급']
df_details_2
# %%
df_details_join_2017 = pd.merge(
    df_2017_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

df_details_join_2018 = pd.merge(
    df_2018_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

df_details_join_2019 = pd.merge(
    df_2019_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

df_details_join_2020 = pd.merge(
    df_2020, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

# %%
df_details_join_2020
# %%
df_details_join_2020.ffill(inplace=True)
# %%
CIR = df_details_join_2020['CRI등급'] == 'D'
df_details_join_2020.drop(df_details_join_2020[CIR].index, inplace=True)
# %%
CIR = df_details_join_2020['CRI등급'] == 'NG'
df_details_join_2020.drop(df_details_join_2020[CIR].index, inplace=True)

# %%
mask = df_details_join_2020['CRI등급'].str.contains('A')
df_details_join_2020.loc[mask, 'CRI등급'] = 'A'
# %%
mask = df_details_join_2020['CRI등급'].str.contains('B')
df_details_join_2020.loc[mask, 'CRI등급'] = 'B'
# %%
mask = df_details_join_2020['CRI등급'].str.contains('C')
df_details_join_2020.loc[mask, 'CRI등급'] = 'C'

# %%
df_details_join_2017
################################
# %%
df_details_join_2017.ffill(inplace=True)
# %%
CIR = df_details_join_2017['CRI등급'] == 'D'
df_details_join_2017.drop(df_details_join_2017[CIR].index, inplace=True)
# %%
CIR = df_details_join_2017['CRI등급'] == 'NG'
df_details_join_2017.drop(df_details_join_2017[CIR].index, inplace=True)

# %%
mask = df_details_join_2017['CRI등급'].str.contains('A')
df_details_join_2017.loc[mask, 'CRI등급'] = 'A'
# %%
mask = df_details_join_2017['CRI등급'].str.contains('B')
df_details_join_2017.loc[mask, 'CRI등급'] = 'B'
# %%
mask = df_details_join_2017['CRI등급'].str.contains('C')
df_details_join_2017.loc[mask, 'CRI등급'] = 'C'

# %%
df_details_join_2018.ffill(inplace=True)
# %%
CIR = df_details_join_2018['CRI등급'] == 'D'
df_details_join_2018.drop(df_details_join_2018[CIR].index, inplace=True)
# %%
CIR = df_details_join_2018['CRI등급'] == 'NG'
df_details_join_2018.drop(df_details_join_2018[CIR].index, inplace=True)

# %%
mask = df_details_join_2018['CRI등급'].str.contains('A')
df_details_join_2018.loc[mask, 'CRI등급'] = 'A'
# %%
mask = df_details_join_2018['CRI등급'].str.contains('B')
df_details_join_2018.loc[mask, 'CRI등급'] = 'B'
# %%
mask = df_details_join_2018['CRI등급'].str.contains('C')
df_details_join_2018.loc[mask, 'CRI등급'] = 'C'

# %%
df_details_join_2019.ffill(inplace=True)
# %%
CIR = df_details_join_2019['CRI등급'] == 'D'
df_details_join_2019.drop(df_details_join_2019[CIR].index, inplace=True)
# %%
CIR = df_details_join_2019['CRI등급'] == 'NG'
df_details_join_2019.drop(df_details_join_2019[CIR].index, inplace=True)

# %%
mask = df_details_join_2019['CRI등급'].str.contains('A')
df_details_join_2019.loc[mask, 'CRI등급'] = 'A'
# %%
mask = df_details_join_2019['CRI등급'].str.contains('B')
df_details_join_2019.loc[mask, 'CRI등급'] = 'B'
# %%
mask = df_details_join_2019['CRI등급'].str.contains('C')
df_details_join_2019.loc[mask, 'CRI등급'] = 'C'

# %%
df_details_join_2017 = df_details_join_2017[[
    '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
# %%
df_details_join_2018 = df_details_join_2018[[
    '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
# %%
df_details_join_2019 = df_details_join_2019[[
    '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]

# %%
df_details_join_2017
# %%
df_details_join_2018
# %%
df_details_join_2019
# %%
total = pd.concat(
    [df_details_join_2017, df_details_join_2018, df_details_join_2019], axis=0)
total_sales = pd.concat([df_details_join_2020[['매출액']], df_details_join_2020[[
                        '매출액']], df_details_join_2020[['매출액']]])
total_sales
# %%
total
# %%
total_sales
# %%
sales = int(df_2017['매출액'].quantile(0.5))
sales
# %%
high = {}
low = {}
for i in range(len(df_2017)):
    if df_2017['매출액'].iloc[i] > sales:
        high[i] = df_2017.iloc[i]
    else:
        low[i] = df_2017.iloc[i]
# %%
df_2017_high = pd.DataFrame(high).T
# %%
df_2017_high
# %%
df_2017_low = pd.DataFrame(low).T
# %%
# %%
IQR1 = total_sales.quantile(q=0.25)
IQR2 = total_sales.quantile(q=0.75)
IQR = IQR2 - IQR1
IQR
# %%
Max = IQR2 + (IQR*3)
Min = IQR1 - (IQR*3)
# %%
print(Max)
print(Min)
# %%


def is_kor_outlier(total_sales):
    kor_score = total_sales['매출액']
    if kor_score > IQR2['매출액'] + 1.5 * IQR['매출액'] or kor_score < IQR1['매출액'] - 1.5 * IQR['매출액']:
        return True
    else:
        return False


# apply 함수를 통하여 각 값의 이상치 여부를 찾고 새로운 열에 결과 저장
total_sales['outlier'] = total_sales.apply(
    is_kor_outlier, axis=1)  # axis = 1 지정 필수

total_sales
# %%
total_trim = total.loc[total['outlier'] == False]

# 이상치여부를 나타내는 열 제거
del total_trim['outlier']

total_trim

# %%
IQR3 = total[['총자산', '영업외비용', '매출원가', '유동자산',
              '매출채권', '인건비', '자기자본']].quantile(q=0.25)
IQR4 = total[['총자산', '영업외비용', '매출원가', '유동자산',
              '매출채권', '인건비', '자기자본']].quantile(q=0.75)
IQR5 = IQR4 - IQR3
IQR5
# %%
max = IQR4 + (IQR5*3)
min = IQR3 - (IQR5*3)
# %%
print(max)
print(min)
# %%


def is_outlier(total):
    out_score = total[['총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
    if out_score > IQR3[['총자산', '영업외비용', '매출원가']] + 1.5 * IQR5[['총자산', '영업외비용', '매출원가']] or out_score < IQR3[['총자산', '영업외비용', '매출원가']] - 1.5 * IQR5[['총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]:
        return True
    else:
        return False


# apply 함수를 통하여 각 값의 이상치 여부를 찾고 새로운 열에 결과 저장
total['outlier'] = total.apply(is_outlier, axis=1)  # axis = 1 지정 필수

total
# %%
total_trim = total.loc[total['outlier'] == False]

# 이상치여부를 나타내는 열 제거
del total_trim['outlier']

total_trim
# %%
model = LinearRegression()
x = total_trim[['총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]  # 변수설정
y = total_sales['매출액']  # 타겟설정
# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=200)
model.fit(x_train, y_train)
# %%
print("Linear regression coeff : {}".format(model.coef_))
print("Linear regression intercept : {}".format(model.intercept_))
# %%
print("training Data evaluation : {}".format(model.score(x_train, y_train)))
print("test Data evaluation : {}".format(model.score(x_test, y_test)))
# %%

scaler = MinMaxScaler()
scaler.fit(x_train)
pd.DataFrame(x_train, columns=total.columns[:])
# %%

# %%

x = total[['총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]  # 변수설정
y = total_sales['매출액']  # 타겟설정
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
RF_model = DecisionTreeRegressor(max_depth=4, random_state=0)
RF_model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_test)
print('훈련 세트 정확도:{:.3f}'.format(RF_model.score(x_train, y_train)))
print('테스트 세트 정확도:{:.3f}'.format(RF_model.score(x_test, y_test)))

# %%














"""
# # %%
# df_details_2 = df_details[['사업자등록번호', 'CRI등급']]
# df_details_2.columns = ['사업자등록번호_마스킹', 'CRI등급']
# df_details_2
# # %%
# df_details_join_2017 = pd.merge(
#     df_2017_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

# df_details_join_2018 = pd.merge(
#     df_2018_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

# df_details_join_2019 = pd.merge(
#     df_2019_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')


# %%
# df_details_join_2017.ffill(inplace=True)
# %%
# CIR = df_details_join_2017['CRI등급'] == 'D'
# df_details_join_2017.drop(df_details_join_2017[CIR].index, inplace=True)
# # %%
# CIR = df_details_join_2017['CRI등급'] == 'NG'
# df_details_join_2017.drop(df_details_join_2017[CIR].index, inplace=True)
# # %%
# mask = df_details_join_2017['CRI등급'].str.contains('A')
# df_details_join_2017.loc[mask, 'CRI등급'] = 'A'
# # %%
# mask = df_details_join_2017['CRI등급'].str.contains('B')
# df_details_join_2017.loc[mask, 'CRI등급'] = 'B'
# # %%
# mask = df_details_join_2017['CRI등급'].str.contains('C')
# df_details_join_2017.loc[mask, 'CRI등급'] = 'C'


# # %%
# df_details_join_2018.ffill(inplace=True)
# # %%
# CIR = df_details_join_2018['CRI등급'] == 'D'
# df_details_join_2018.drop(df_details_join_2018[CIR].index, inplace=True)
# # %%
# CIR = df_details_join_2018['CRI등급'] == 'NG'
# df_details_join_2018.drop(df_details_join_2018[CIR].index, inplace=True)

# # %%
# mask = df_details_join_2018['CRI등급'].str.contains('A')
# df_details_join_2018.loc[mask, 'CRI등급'] = 'A'
# # %%
# mask = df_details_join_2018['CRI등급'].str.contains('B')
# df_details_join_2018.loc[mask, 'CRI등급'] = 'B'
# # %%
# mask = df_details_join_2018['CRI등급'].str.contains('C')
# df_details_join_2018.loc[mask, 'CRI등급'] = 'C'


# # %%
# df_details_join_2019.fillna(df_details_join_2019.mode(), inplace=True)
# # %%
# CIR = df_details_join_2019['CRI등급'] == 'D'
# df_details_join_2019.drop(df_details_join_2019[CIR].index, inplace=True)
# # %%
# CIR = df_details_join_2019['CRI등급'] == 'NG'
# df_details_join_2019.drop(df_details_join_2019[CIR].index, inplace=True)

# # %%
# mask = df_details_join_2019['CRI등급'].str.contains('A')
# df_details_join_2019.loc[mask, 'CRI등급'] = 'A'
# # %%
# mask = df_details_join_2019['CRI등급'].str.contains('B')
# df_details_join_2019.loc[mask, 'CRI등급'] = 'B'
# # %%
# mask = df_details_join_2019['CRI등급'].str.contains('C')
# df_details_join_2019.loc[mask, 'CRI등급'] = 'C'

# # %%
# df_details_join_2017 = df_details_join_2017[[
#     '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
# # %%
# df_details_join_2018 = df_details_join_2018[[
#     '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]
# # %%
# df_details_join_2019 = df_details_join_2019[[
#     '사업자등록번호_마스킹', '조사년도', 'CRI등급', '총자산', '영업외비용', '매출원가', '유동자산', '매출채권', '인건비', '자기자본']]

# %%
df_details_join_2017
# %%
df_details_join_2018
# %%
df_details_join_2019
# %%
total = pd.concat(
    [df_details_join_2017, df_details_join_2018, df_details_join_2019], axis=0)
A = total[total['CRI등급'] == 'A']
B = total[total['CRI등급'] == 'B']
C = total[total['CRI등급'] == 'C']
# %%
plt.hist(A['영업외비용'])
# %%
plt.hist(B['영업외비용'])
# %%
plt.hist(C['영업외비용'])
# %%
total
# %%
for i in total.columns:
    a = enumerate(total[i])

"""