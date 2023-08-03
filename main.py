# %%
import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from preprocessing.correlation import Correlation2020, Correlation2021
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

# %%
df = pd.read_csv('/Users/ojin-u/Desktop/개발/python/project/train_data/main.csv')
# %%
df_2017_2021 = df[['2017_매출액', '2018_매출액', '2019_매출액',
                   '2020_매출액', '2021_매출액']]
# %%
business_df = df['사업자등록번호_마스킹']
# %%
col_con = pd.concat([business_df, df_2017_2021], axis=1)

# %%
col_con.isnull().sum()

# %%
df_2017_2019 = df[['2019_유형자산 증가율', '2019_총자산 증가율', '2019_총자산', '2019_매출 증가율', '2019_자기자본', '2019_유동비율', '2019_자산총계', '2019_유동자산', '2019_자본총계', '2019_당기순이익(손실)', '2019_영업이익', '2019_부채총계', '2019_유동부채', '2019_당좌자산', '2019_판매비와관리비', '2019_차입금', '2019_매출총이익', '2019_매출액', '2019_인건비', '2019_이익잉여금', '2019_영업외비용', '2019_매출채권', '2019_매출원가', '2019_이자비용', '2019_투자자산', '2019_비유동부채', '2019_법인세비용', '2019_재고자산', '2019_제조경비', '2019_노무비', '2019_제조원가', '2019_대손상각비', '2019_원재료', '2019_자본잉여금', '2019_수출액3', '2019_수출액2', '2019_수출액1', '2019_당좌부채(당좌차월)', '2019_수출액4', '2019_수출액5', '2018_유형자산 증가율', '2018_총자산 증가율', '2018_매출 증가율', '2018_자기자본', '2018_총자산', '2018_유동비율', '2018_자산총계', '2018_유동자산',
                   '2018_당기순이익(손실)', '2018_영업이익', '2018_부채총계', '2018_유동부채', '2018_당좌자산', '2018_판매비와관리비', '2018_차입금', '2018_매출총이익', '2018_매출액', '2018_인건비', '2018_이익잉여금', '2018_영업외비용',
                   '2018_매출원가', '2018_이자비용', '2018_투자자산', '2018_비유동부채', '2018_법인세비용', '2018_재고자산', '2018_제조경비', '2018_노무비', '2018_제조원가', '2018_대손상각비', '2018_원재료', '2018_자본잉여금', '2018_수출액3', '2018_수출액2', '2018_수출액1', '2018_당좌부채(당좌차월)', '2018_수출액4', '2018_수출액5', '2017_총자산 증가율', '2017_매출 증가율', '2017_자기자본', '2017_총자산', '2017_유동비율', '2017_자산총계', '2017_유동자산', '2017_자본총계', '2017_당기순이익(손실)', '2017_영업이익', '2017_부채총계', '2017_유동부채', '2017_당좌자산', '2017_판매비와관리비', '2017_차입금', '2017_매출총이익', '2017_매출액', '2017_인건비', '2017_이익잉여금', '2017_영업외비용', '2017_매출채권', '2017_매출원가', '2017_이자비용', '2017_투자자산', '2017_비유동부채', '2017_법인세비용', '2017_재고자산', '2017_제조경비', '2017_노무비', '2017_제조원가', '2017_대손상각비', '2017_원재료', '2017_자본잉여금', '2017_수출액3', '2017_수출액2', '2017_수출액1', '2017_당좌부채(당좌차월)', '2017_수출액4', '2017_수출액5', "2020_매출액", '2021_매출액']]

# %%
df_2017_2019.drop(['2019_수출액4', '2017_수출액5', '2018_수출액5', '2019_수출액5', '2018_수출액4', '2017_수출액4', '2017_수출액1', '2019_당좌부채(당좌차월)',
                   '2019_수출액1', '2018_당좌부채(당좌차월)', '2018_수출액1', '2017_당좌부채(당좌차월)', '2017_수출액2', '2019_수출액2', '2018_수출액2', '2017_수출액3', '2018_수출액3', '2019_수출액3',
                   '2017_자본잉여금', '2018_자본잉여금', '2019_자본잉여금', '2017_원재료', '2018_원재료', '2019_원재료', '2018_대손상각비', '2017_대손상각비', '2019_대손상각비', '2017_비유동부채', '2017_법인세비용', '2018_제조경비',
                   '2017_제조원가', '2018_제조원가', '2017_재고자산', '2017_노무비', '2019_제조원가', '2017_제조경비', '2018_노무비', '2019_노무비'], axis=1, inplace=True)


df_2017_2019.isnull().sum().sort_values(ascending=False)

# %%
correlation2020 = Correlation2020()
correlation2021 = Correlation2021()
# %%
df_2020 = correlation2020.cc_2020(df_2017_2019)
df_2021 = correlation2021.cc_2021(df_2017_2019)


# %%
df_2020
# %%
a = dict(df_2020.iloc[:, 0])
b = a.values()
c = df_2017_2019[b]
c.isnull().sum().sort_values(ascending=False)
# %%
c['2019_매출액'].describe()
# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[:]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()
cor_2020
sns.clustermap(cor_2020, annot=True)
# %%
# type(cor_2020)


# sns.clustermap(cor_2020, annot=True)
# """
# 2019_매출액(V) - 2019_매출원가 : 1
# 2019_제조경비(V) - 2019_매출원가 : 1
# 2017_매출액(V) - 2017_매출액 : 0.99
# """


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[6:12]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()
sns.clustermap(cor_2020, annot=True)
"""
2017_매출원가(V) - 2018_매출원가 : 0.99
2017_매출채권(V) - 2019_매출채권 : 0.98
2019_유동부채(V) - 2018_유동부채 : 0.98
"""


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[12:18]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()

sns.clustermap(cor_2020, annot=True)
"""
2019_유동자산(V) - 2018_유동자산 = 0.99
2017_당좌자산(V) - 2017_유동자산(V) = 0.99
"""


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[18:24]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()

sns.clustermap(cor_2020, annot=True)
"""
2019_당좌자산(V) - 2018_당좌자산 = 0.99
2017_부채총계(V) - 2018_부채총계 = 0.99
2018_부채총계(V) - 2018_부채총계 = 0.99
"""


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[24:30]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()

sns.clustermap(cor_2020, annot=True)
"""
2018_자산총계(V) - 2018_총자산 - 2019_총자산(V) = 1
2018_재고자산(V) - 2019_재고자산 = 0.91
"""


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[30:]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()

sns.clustermap(cor_2020, annot=True)
"""
2018_자산총계(V) - 2018_총자산 - 2019_총자산(V) = 1
2019_재고자산(V) - 2018_재고자산 = 0.91
"""


# %%
df_2017_2019.drop(['2019_매출액', '2019_제조경비', '2017_매출액', '2017_매출원가', '2017_매출채권', '2019_유동부채', '2019_유동자산', '2017_당좌자산',
                   '2017_유동자산', '2019_당좌자산', '2017_부채총계', '2018_부채총계', '2019_총자산', '2018_자산총계', '2018_재고자산'], axis=1, inplace=True)


# %%
df_2017_2019
# %%

df_2017_2019.isnull().sum().sort_values(ascending=False)

# %%
correlation2020 = Correlation2020()
# %%
df_2020 = correlation2020.cc_2020(df_2017_2019)
# %%
a = df_2017_2019[[
    '2018_매출액',
    '2018_유동부채',
    '2019_영업외비용',
    '2018_유동자산',
    '2018_당좌자산',
    '2019_부채총계',
    '2018_총자산',
    '2019_자산총계'

]]
# %%
# 456 범위가 0월부터 100억원대 사이
# 0으로 채워도 평균이나 중위값의 큰 변화가 없다.
# 애초에 0값이 있어 min의 변화는 없음
# 매출액이나 총자산은 0으로 처리하면 안될것 같은데 부채는 없을 수
# 있다고 생각함

# %%
# sns.boxplot(data=a, x="age", y="class")

# %%
test1 = a[['2019_부채총계']]
test1.isnull().sum()
test1.fillna(0, inplace=True)
sns.boxplot(x=test1['2019_부채총계'])

# test.describe()


# %%
# 704
# 뭐 먹는데 돈썼겠찌

test2 = a[['2019_영업외비용']]
test2.fillna(test2.median(), inplace=True)
test2.describe()
# sns.boxplot(x=test['2019_영업외비용'])

# 522
# %%

test3 = a[['2018_총자산']]
q = test3.quantile(0.25)

test3.fillna(a, inplace=True)
test3.describe()
sns.boxplot(x=test3['2018_총자산'])
# 449
# %%

test4 = a[['2019_자산총계']]
q = test4.quantile(0.25)
test4.fillna(a, inplace=True)
test4
sns.boxplot(x=test4['2019_자산총계'], )

# %%

a = dict(df_2020.iloc[:, 0])
b = a.values()
c = df_2017_2019[b]
c.isnull().sum().sort_values(ascending=False)


# %%
cor_col_name = c.columns
cor_2020 = c[cor_col_name[:]]
cor_2020 = pd.concat([df_2017_2019[['2020_매출액']], cor_2020], axis=1)

cor_2020 = cor_2020.dropna()
cor_2020 = cor_2020.corr()

sns.clustermap(cor_2020, annot=True)


# %%
cor_2020


# %%
# 2017_

# 낮아도 중요하다고 생각하는 피쳐 넣어서 돌려보기

# 전처리 하기전 중간, 후 박스플롯을 그려가면서 하기
# 중간 모듈을 만들어 놓기


# 이제 다음단계가 서로관의 상관관계가 높으면 제거
# 너무 밀접하게 관계있으면 차지하는 비중이 높다.

# corr먼저 말고  뮤츄얼 인포리그레션(머신러닝에 맞춰져있다.) 이거 먼저하고
# 한국어로 정리 되어있을것이니 찾아봐라.
# 0.7로 거르기보단 corr

# 뮤츄얼로 나누고 피어슨 써서 관리 해서


# 피쳐선택(상호작용 확인)
# 박스플롯 이상치 차트뽑아내기
# 결측치
# 박스플롯 이상치 차트뽑아내기(이상치처리)
# 모델링

# discrete = [1, 2, 3]  # discrete column index
# mu = mutual_info_classif(train[in_var],
#                          train[target_name],
#                          discrete_features=discrete)

# X = df_2017_2019.copy()
# y = X.pop('2020_매출액')
# for colname in X.select_dtypes('object'):
#     X[colname], _ = X[colname].factorize()
# dicrete_features = X.dtypes == int


# def make_mi_scores(X, y, discrete_features):
#     mi_scores = mutual_info_regression(
#         X, y, discrete_features=discrete_features)

#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores


# mi_scores = make_mi_scores(X, y, discrete_features)
# mi_scores[::3]  # show a few features with their MI scores


# def plot_mi_scores(scores):
#     scores = scores.sort_values(ascending=True)
#     width = np.arange(len(scores))
#     ticks = list(scores.index)
#     plt.barh(width, scores)
#     plt.yticks(width, ticks)
#     plt.title("Mutual Information Scores")


# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)

# %%


""" 시각화
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
# 1, 1은 1x1 그리드를 생성
pos = np.arange(len(mu))
plt.barh(pos, mu)
plt.yticks(pos, in_var)
threshold = 0.01
for idx, tick in enumerate(axes.yaxis.get_major_ticks()):
    tick.label.set_fontsize(15)
    tick.label.set_rotation(15)
    if np.squeeze(mu)[idx] > threshold:
        tick.label.set_color("red")

plt.vlines(x=threshold, ymin=-1, ymax=len(mu))
plt.show()
"""


""" mutual info regression

# %%
df79 = df_2017_2019.fillna(df_2017_2019.median())

X = df79.drop(columns=['2020_매출액'])

y = df79['2020_매출액']

mutual_info_regression(
    X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)


# %%

for col in df_2017_2019.columns:
    df_2017_2019[col]

"""
