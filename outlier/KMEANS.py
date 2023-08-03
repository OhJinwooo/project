# %%
from sklearn.cluster import KMeans
import koreanize_matplotlib
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# %%
warnings.filterwarnings("ignore")


sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.set_context('talk')

sample_n = 50
rows = 5
columns = sample_n
total_sample = sample_n*rows
X = [[0 for _ in range(rows)] for _ in range(columns)]
Y = [[0 for _ in range(rows)] for _ in range(columns)]

# %%
for idx in range(rows):
    variance = abs(np.random.randn(1)*3)
    trans_x = np.random.randn(1)*10
    trans_y = np.random.randn(1)*10
    X[idx] = np.random.randn(sample_n)*variance + trans_x
    Y[idx] = np.random.randn(sample_n)*variance + trans_y

X_new = np.concatenate([X[0], X[1], X[2], X[3], X[4]], axis=0)
Y_new = np.concatenate([Y[0], Y[1], Y[2], Y[3], Y[4]], axis=0)

cluster_set = np.zeros((total_sample, 2))

for idx in range(len(cluster_set)):
    cluster_set[idx][0] = X_new[idx]
    cluster_set[idx][1] = Y_new[idx]

np.random.shuffle(cluster_set)
# %%
kmeans = KMeans(n_clusters=5)
y_pred = kmeans.fit_predict(cluster_set)
print(y_pred)
# %%
print(kmeans.labels_[0])
# KMean 클러스터링이 첫번째 샘플을 레이블링한 값 0

# %%
color_dic = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'blue'}

for idx in range(len(cluster_set)):
    color_idx = kmeans.labels_[idx]
    plt.scatter(cluster_set[idx][0], cluster_set[idx]
                [1], c=color_dic[color_idx])
    plt.xlabel('X')
    plt.ylabel('Y')

for idx in range(len(kmeans.cluster_centers_)):
    plt.scatter(kmeans.cluster_centers_[
                idx][0], kmeans.cluster_centers_[idx][1], c='black')
# 센트로이드는 따로 검정색 마크로 표시
# %%
# https://humankind.tistory.com/21 [출처]
