# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

# %%
df_2017 = pd.read_csv(
    '/Users/ojin-u/Desktop/개발/python/project/train_data/years/2017.csv')
# %%

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
# create some data
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500])
data
# %%
data = df_2017_high['총자산'].values
data = np.array(data)
data
# %%

# %%
# create a barplot

plt.bar(range(len(data)), data)

# %%
# use LOF to identify outliers

# specify the number of neighbors to use for LOF
lof = LocalOutlierFactor(n_neighbors=5)
# %%
outliers = lof.fit_predict(data.reshape(-1, 1))


# plot the outliers as red points on the barplot
# %%
plt.plot(np.where(outliers == -1), data[outliers == -1], 'ro')
# %%
# show the plot

plt.show()

# %%
