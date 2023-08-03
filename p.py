# %%
import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

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
df_2020['조사년도'] = 2020
df_2021['조사년도'] = 2021
# %%
df_total = pd.concat([df_2017, df_2018, df_2019, df_2020, df_2021])
# %%
df_total
# %%
df_total = df_total.sort_values(['사업자등록번호_마스킹', '조사년도'])
# %%
df_total = df_total[['사업자등록번호_마스킹', '조사년도', '유형자산 증가율', '총자산 증가율',	'매출 증가율', '자기자본', '총자산', '유동비율', '자산총계',	'유동자산',
                     '자본총계', '유동자산', '자본총계', '당기순이익(손실)', '영업이익', '부채총계', '유동부채', '당좌자산', '판매비와관리비', '차입금', '매출총이익', '매출액', '인건비', '이익잉여금', '영업외비용', '매출채권', '매출원가', '이자비용', '투자자산', '비유동부채', '법인세비용', '재고자산', '제조경비', '노무비', '제조원가',  '대손상각비', '원재료', '자본잉여금', '수출액3', '수출액2', '수출액1', '당좌부채(당좌차월)',	'수출액4', '수출액5']]

# %%
len(df_total['사업자등록번호_마스킹'].unique())

# %%
df_total.to_csv('df_total.csv')
# %%
df_total
# %%
q = pd.DataFrame(df_total['사업자등록번호_마스킹'][4])
q.iloc[2:3].columns
int(q.iloc[2:3].values)

# %%

# %%
num = 1
total_df_dict = {}
for i in range(len(df_total['사업자등록번호_마스킹'].unique())):
    if len(df_total['사업자등록번호_마스킹'][i]) == 5:
        q = pd.DataFrame(df_total['사업자등록번호_마스킹'][i])
        total_df_dict[num] = int(q.iloc[2:3].values)
        num += 1
# %%
df = pd.DataFrame(total_df_dict, index=[0]).T
df.columns = ['사업자등록번호_마스킹']
# %%
df
# %%
