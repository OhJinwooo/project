# %%
import koreanize_matplotlib
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
df_2017_low = pd.DataFrame(low).T

# %%
df_2017_high.ffill(inplace=True)
df_2017_low.ffill(inplace=True)
df_2017.ffill(inplace=True)

# %%
# %%
# 부동 소수점 출력에 대한 정밀도 자릿수 설정
np.set_printoptions(precision=2)

# %%
# 랜덤 시드 번호 설정
np.random.seed(10)
# %%
# 00 랜덤 생성 x ~ N(10, 2)
mu, sigma = 10, 2
# %%
x = mu + sigma*df_2017_low[['매출액']].values

# %%
len(x)
# %%
plt.hist(x)
# %%
np.median(x)


# %%
Q1 = np.percentile(x, 25, axis=0)
# %%
Q1

# %%
Q3 = np.percentile(x, 75, axis=0)  # 5.700
# %%
Q3
# %%
IQR = Q3 - Q1
# %%
IQR
# %%
x_RobustScaler = RobustScaler().fit_transform(x)
# %%
x_RobustScaler[-10:]
# %%
np.median(x_RobustScaler)
# %%
np.mean(x_RobustScaler)
# %%
np.std(x_RobustScaler)
# %%
plt.hist(x_RobustScaler)
# %%
#
x_RobustScaler_zoonin = x_RobustScaler[x_RobustScaler < 3]
# %%
#
plt.hist(x_RobustScaler_zoonin, bins=np.arange(-3, 3, 0.1))
# %%
sns.boxplot(x_RobustScaler_zoonin)
# %%

# 이상치 범위
outlier_step = 3 * IQR
# %%
# 모든 이상치 인덱스를 연결할 리스트
outlier_list = []
# %%
len(x)
# %%
df_2017_high1 = df_2017_high['총자산'].values

# %%
# 각 피쳐에서 이상치 탐지
outlier_list_col = df_2017_high1[(
    x < Q1 - outlier_step) | (x > Q3 + outlier_step)].index
outlier_list.append(outlier_list_col)

# %%

# %%
# 모든 이상치 인덱스 연결
outlier_indices = np.concatenate(outlier_list, axis=0)
# %%
# Drop outliers
df_2017_high = df_2017_high.drop(np.unique(outlier_indices), axis=0)

# %%
df_2017_high
# %%

# Assume a dataset
data = df_2017_high['매출액']

# Calculate skewness
print("Skewness: ", stats.skew(data))
# %%
sns.distplot(df_2017_high['매출액'])
# %%
sns.distplot(df_2017_low['매출액'])

# %%
sns.distplot(np.log1p(df_2017_high['매출액']))
# %%
sns.distplot(np.log1p(df_2017_low['매출액']))
# %%
"""



# %%
# 부동 소수점 출력에 대한 정밀도 자릿수 설정
np.set_printoptions(precision=2)

# %%
# 랜덤 시드 번호 설정
np.random.seed(10)
# %%
# 00 랜덤 생성 x ~ N(10, 2)
mu, sigma = 10, 2
# %%
x = mu + sigma*low_total[['매출액_y']].values

# %%
len(x)
# %%
plt.hist(x)
# %%
np.median(x)


# %%
Q1 = np.percentile(x, 25, axis=0)
# %%
Q1

# %%
Q3 = np.percentile(x, 75, axis=0)  # 5.700
# %%
Q3
# %%
IQR = Q3 - Q1
# %%
IQR
# %%
x_RobustScaler = RobustScaler().fit_transform(x)
# %%
x_RobustScaler[-10:]
# %%
np.median(x_RobustScaler)
# %%
np.mean(x_RobustScaler)
# %%
np.std(x_RobustScaler)
# %%
plt.hist(x_RobustScaler)
# %%
#
x_RobustScaler_zoonin = x_RobustScaler[x_RobustScaler < 2]
# %%
#
plt.hist(x_RobustScaler_zoonin, bins=np.arange(-3, 3, 0.1))
# %%
sns.boxplot(x_RobustScaler_zoonin)

"""
