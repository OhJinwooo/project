
import pandas as pd
# from sklearn.feature_selection import mutual_info_regression
import numpy as np
import matplotlib.pyplot as plt


# class MutualInfoRegression:
#     comparisons = {}
#     comparisons_last = {}
#     comparisons.clear()
#     comparisons_last.clear()

#     def make_mi_scores(X, y, discrete_features):
#         mi_scores = mutual_info_regression(
#             X, y, discrete_features=discrete_features)

#         mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#         mi_scores = mi_scores.sort_values(ascending=False)
#         return mi_scores
#     mi_scores = make_mi_scores(X, y, discrete_features)
#     mi_scores[::3]  # show a few features with their MI scores

#     def plot_mi_scores(scores):
#         scores = scores.sort_values(ascending=True)
#         width = np.arange(len(scores))
#         ticks = list(scores.index)
#         plt.barh(width, scores)
#         plt.yticks(width, ticks)
#         plt.title("Mutual Information Scores")

#     plt.figure(dpi=100, figsize=(8, 5))
#     plot_mi_scores(mi_scores)
# def mutual_info(self, df):
#     for i in df.columns:


# MainProject
class Correlation2020:
    comparisons = {}
    comparisons_last = {}

    def cc_2020(self, df):
        self.comparisons.clear()
        self.comparisons_last.clear()
        for i in df.columns:
            if i != '매출액':
                temp_df = df[['매출액', i]].dropna()
                self.comparisons[i] = temp_df['매출액'].corr(temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2020_key_list = self.comparisons_last.keys()
        dict_2020_value_list = self.comparisons_last.values()

        dict_2020_df = pd.DataFrame(
            {'feature': dict_2020_key_list,
             '상관계수': dict_2020_value_list})

        return dict_2020_df.sort_values("상관계수", ascending=False)


class Correlation2021:
    comparisons = {}
    comparisons_last = {}
    comparisons.clear()
    comparisons_last.clear()

    def cc_2021(self, df):
        for i in df.columns:
            if i != '매출액':
                temp_df = df[['매출액', i]].dropna()
                self.comparisons[i] = temp_df['매출액'].corr(temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2021_key_list = self.comparisons_last.keys()
        dict_2021_value_list = self.comparisons_last.values()

        dict_2021_df = pd.DataFrame(
            {'feature': dict_2021_key_list,
             '상관계수': dict_2021_value_list})

        return dict_2021_df.sort_values("상관계수", ascending=False)


# SubProject
class SubCorrelation2021:
    comparisons = {}
    comparisons_last = {}
    comparisons.clear()
    comparisons_last.clear()

    def cc_a_net_profit(self, df):
        for i in df.columns:
            if i != '2021_당기순이익(손실)':
                temp_df = df[['2021_당기순이익(손실)', i]].dropna()
                self.comparisons[i] = temp_df['2021_당기순이익(손실)'].corr(
                    temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2020_key_list = self.comparisons_last.keys()
        dict_2020_value_list = self.comparisons_last.values()

        dict_2020_df = pd.DataFrame(
            {'feature': dict_2020_key_list,
             '상관계수': dict_2020_value_list})

        return dict_2020_df.sort_values("상관계수", ascending=False)

    def cc_gross_profit(self, df):
        for i in df.columns:
            if i != '2021_매출총이익':
                temp_df = df[['2021_매출총이익', i]].dropna()
                self.comparisons[i] = temp_df['2021_매출총이익'].corr(temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2021_key_list = self.comparisons_last.keys()
        dict_2021_value_list = self.comparisons_last.values()

        dict_2021_df = pd.DataFrame(
            {'feature': dict_2021_key_list,
             '상관계수': dict_2021_value_list})

        return dict_2021_df.sort_values("상관계수", ascending=False)

    def cc_sales_growth(self, df):
        for i in df.columns:
            if i != '2021_매출 증가율':
                temp_df = df[['2021_매출 증가율', i]].dropna()
                self.comparisons[i] = temp_df['2021_매출 증가율'].corr(temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2021_key_list = self.comparisons_last.keys()
        dict_2021_value_list = self.comparisons_last.values()

        dict_2021_df = pd.DataFrame(
            {'feature': dict_2021_key_list,
             '상관계수': dict_2021_value_list})

        return dict_2021_df.sort_values("상관계수", ascending=False)

    def cc_current_ratio(self, df):
        for i in df.columns:
            if i != '2021_유동비율':
                temp_df = df[['2021_유동비율', i]].dropna()
                self.comparisons[i] = temp_df['2021_유동비율'].corr(temp_df[i])

        for key in self.comparisons.keys():
            if self.comparisons[key] > 0.7:
                self.comparisons_last[key] = self.comparisons[key]

        dict_2021_key_list = self.comparisons_last.keys()
        dict_2021_value_list = self.comparisons_last.values()

        dict_2021_df = pd.DataFrame(
            {'feature': dict_2021_key_list,
             '상관계수': dict_2021_value_list})

        return dict_2021_df.sort_values("상관계수", ascending=False)


def aaaaaa(self, df):
    comparisons = {}
    comparisons_last = {}
    comparisons.clear()
    comparisons_last.clear()

    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            cor_col_name = df.columns
            cor_2020 = df[cor_col_name[j:i]]

            cor_2020 = cor_2020.dropna()
            cor_2020 = cor_2020.corr()
