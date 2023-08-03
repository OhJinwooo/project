# %%
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
    '/Users/ojin-u/Desktop/개발/python/project/train_data/main.csv'
)
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


# %%
df_details_join_2017.dropna(inplace=True)
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
df_details_join_2018.dropna(inplace=True)
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
df_details_join_2019.dropna(inplace=True)
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
total = pd.concat([df_2017_last, df_2018_last, df_2019_last], axis=0)
# %%
a = enumerate(total['사업자등록번호_마스킹'])
a = dict(a)
len(a) / 3
total['총자산'][7000]
# %%

# %%
num = 2
total_di2ct = {}
for i in range(len(a) / 3):
    business_num = total['사업자등록번호_마스킹'][i]

    total_df = pd.DataFrame()
# %%
total = pd.merge([df_main['사업자등록번호_마스킹'], df_details_join_2017, df_details_join_2018,
                 df_details_join_2019], left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')

# %%
# year_dict = {}
# for i in df_details_join_2017.columns:
#     if
# %%
"""
# fin1 = financial_df.groupby(['사업자등록번호_마스킹', '조사년도'])['유동비율'].sum()
# fin2 = fin1.unstack('사업자등록번호_마스킹')
# fin2 = fin2.T
# fin2

"""
# %%

# %%
# dsafhdsal;k
# %%
df_2019_dict = dict(df_2019_m.iloc[:, 0])
df_2019_dict_val = df_2019_dict.values()
df_2019_val = df_2020[df_2019_dict_val]
df_2019_val_con = pd.concat([df_2019[['매출액']], df_2019_val], axis=1)

df_2019_val_con.isnull().sum().sort_values(ascending=False)
# %%
# df_2019_val_con.drop(['당좌부채(당좌차월)', '자본잉여금', '노무비', '제조경비',
#                      '제조원가', '재고자산', '비유동부채', '이자비용', '매출원가', '매출채권', '영업외비용', '이익잉여금', '인건비', '매출총이익', '판매비와관리비', '유동부채', '부채총계'], axis=1, inplace=True)
# %%
# df_2019_val_con.fillna(df_2019_val_con.mean(), inplace=True)
df_2019_val_con.dropna(inplace=True)
df_2019_val_con.isnull().sum().sort_values(ascending=False)
# %%
df_2019_val_con
# %%
corr = df_2019_val_con.corr()
sns.heatmap(corr, annot=True)
# %%
sns.boxplot(df_2019_val_con[['매출채권']])
# %%
plt.hist(df_2019_val_con[['매출채권']], bins=500)

# %%
# %%
df_2019_val_con.isnull().sum()


# %%
# 2019
# ---------------------------------------------------------
# %%

drop_2019_df = df_2019.drop(['유형자산 증가율',
                             '총자산 증가율', '매출 증가율',
                             '유동비율', '수출액1', '수출액2', '수출액3', '수출액4', '수출액5'], axis=1)


# %%
drop_2019_df
# %%
drop_2019_df.isnull().sum().sort_values(ascending=False)
# %%
correlation = Correlation2020()
# %%

df_2019_m = correlation.cc_2020(drop_2019_df)
# %%
df_2019_m


# %%
df_2019_last = df_2019[['사업자등록번호_마스킹', '매출원가', '매출채권',
                        '유동자산', '영업외비용', '당좌자산', '부채총계', '총자산']]
# %%
df_2019_last
# %%
df_details_2 = df_details[['사업자등록번호', 'CRI등급']]
df_details_2.columns = ['사업자등록번호_마스킹', 'CRI등급']
df_details_2
# %%
df_details_join = pd.merge(
    df_2019_last, df_details_2, left_on='사업자등록번호_마스킹', right_on='사업자등록번호_마스킹', how='left')
# %%
df_details_join
# %%
df_details_join.dropna(inplace=True)
# %%
CIR = df_details_join['CRI등급'] == 'D'
df_details_join.drop(df_details_join[CIR].index, inplace=True)
df_details_join
# %%
CIR = df_details_join['CRI등급'] == 'NG'
df_details_join.drop(df_details_join[CIR].index, inplace=True)
df_details_join

# %%
df_details_join
mask = df_details_join['CRI등급'].str.contains('A')
df_details_join.loc[mask, 'CRI등급'] = 'A'
# %%
df_details_join
mask = df_details_join['CRI등급'].str.contains('B')
df_details_join.loc[mask, 'CRI등급'] = 'B'
# %%
df_details_join
mask = df_details_join['CRI등급'].str.contains('C')
df_details_join.loc[mask, 'CRI등급'] = 'C'
df_details_join

# %%
df_details_join.groupby(['CRI등급', '사업자등록번호_마스킹', '매출원가',
                        '매출채권', '유동자산', '영업외비용', '당좌자산', '부채총계', '총자산'])
# %%
df_grouped = df_details_join.sort_values('CRI등급')
# %%
df_grouped
# %%
"""
# fin1 = financial_df.groupby(['사업자등록번호_마스킹', '조사년도'])['유동비율'].sum()
# fin2 = fin1.unstack('사업자등록번호_마스킹')
# fin2 = fin2.T
# fin2

"""
# %%

# %%

# %%
df_2019_dict = dict(df_2019_m.iloc[:, 0])
df_2019_dict_val = df_2019_dict.values()
df_2019_val = df_2020[df_2019_dict_val]
df_2019_val_con = pd.concat([df_2019[['매출액']], df_2019_val], axis=1)

df_2019_val_con.isnull().sum().sort_values(ascending=False)
# %%
# df_2019_val_con.drop(['당좌부채(당좌차월)', '자본잉여금', '노무비', '제조경비',
#                      '제조원가', '재고자산', '비유동부채', '이자비용', '매출원가', '매출채권', '영업외비용', '이익잉여금', '인건비', '매출총이익', '판매비와관리비', '유동부채', '부채총계'], axis=1, inplace=True)
# %%
# df_2019_val_con.fillna(df_2019_val_con.mean(), inplace=True)
df_2019_val_con.dropna(inplace=True)
df_2019_val_con.isnull().sum().sort_values(ascending=False)
# %%
df_2019_val_con
# %%
corr = df_2019_val_con.corr()
sns.heatmap(corr, annot=True)
# %%
sns.boxplot(df_2019_val_con)


# %%
# %%
df_2019_val_con.isnull().sum()


# %%
# -----------------------------------------------------------------------
total_df = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/main.csv')

total_df = total_df[['2017_매출원가', '2017_유동자산', '2017_영업외비용',
                     '2017_당좌자산', '2017_총자산', '2017_당기순이익(손실)',
                     '2018_매출원가', '2018_유동자산', '2018_영업외비용', '2018_당좌자산', '2018_총자산', '2018_당기순이익(손실)',
                     '2019_매출원가', '2019_유동자산',
                     '2019_영업외비용', '2019_당좌자산', '2019_총자산',
                     '2019_당기순이익(손실)', '2020_매출원가', '2020_유동자산', '2020_영업외비용', '2020_당좌자산', '2020_총자산', '2020_당기순이익(손실)', '2021_매출원가', '2021_유동자산', '2021_영업외비용', '2021_당좌자산', '2021_총자산', '2021_당기순이익(손실)']]


# %%
total_df.dropna(inplace=True)
total_df

# %%
sns.boxplot(total_df[['2019_매출원가', '2019_유동자산',
                     '2019_영업외비용', '2019_당좌자산', '2019_총자산',
                      '2019_당기순이익(손실)']])
# %%

a = total_df.columns.str.split('_', expand=True)
type(a)
# total_melted_df[['Year', 'Metric']
#                 ] = total_melted_df['Year_Metric'].
# %%
