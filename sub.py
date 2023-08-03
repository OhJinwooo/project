# %%

import pandas as pd
from preprocessing.correlation import SubCorrelation2021

# %%
df = pd.read_csv('/Users/ojin-u/Desktop/개발/python/project/train_data/main.csv')

# %%
df_2020 = df[[]]


df_2020.isnull().sum().sort_values(ascending=False)
# %%
subcorrelation2021 = SubCorrelation2021()

# %%
df_2020 = subcorrelation2021.cc_2020(df_2020)

# %%
df_2020
# %%
df_2021
# %%
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
