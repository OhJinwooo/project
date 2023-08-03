# %%
import koreanize_matplotlib
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.set_context('talk')


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
# %%
X = df_2017_high[['총자산', '매출액']].values
X

# %%
# dbscan = DBSCAN(eps=10.0, min_samples=5).fit(X)
dbscan = DBSCAN(eps=2.5, min_samples=13).fit(X)
dbscan
# %%
labels = dbscan.labels_

pd.Series(labels).value_counts()

# %%
plt.figure(figsize=(12, 12))

unique_labels = set(labels)
colors = ['#586fab', '#f55354']

for color, label in zip(colors, unique_labels):
    sample_mask = [True if l == label else False for l in labels]
    plt.plot(X[:, 0][sample_mask], X[:, 1][sample_mask], 'o', color=color)
plt.xlabel('총자산')
plt.ylabel('매출액')

# %%
