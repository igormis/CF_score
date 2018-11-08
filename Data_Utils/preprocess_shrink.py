# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
#from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

#import pickle
import numpy as np
import pandas as pd
from collections import Counter

# load in the data
df = pd.read_csv(r'..\Dataset\edited_rating.csv')
print("original dataframe size:", len(df))

N = df.userId.max() + 1 # number of users
M = df.news_idx.max() + 1 # number of news

user_ids_count = Counter(df.userId)
news_ids_count = Counter(df.news_idx)

# number of users and news we would like to keep
n = 1000
m = 200

user_ids = [u for u, c in user_ids_count.most_common(n)]
news_ids = [m for m, c in news_ids_count.most_common(m)]

# make a copy, otherwise ids won't be overwritten
df_small = df[df.userId.isin(user_ids) & df.news_idx.isin(news_ids)].copy()

# need to remake user ids and news ids since they are no longer sequential
new_user_id_map = {}
i = 0
for old in user_ids:
  new_user_id_map[old] = i
  i += 1
print("i:", i)

new_news_id_map = {}
j = 0
for old in news_ids:
  new_news_id_map[old] = j
  j += 1
print("j:", j)

print("Setting new ids")
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'news_idx'] = df_small.apply(lambda row: new_news_id_map[row.news_idx], axis=1)

print("max user id:", df_small.userId.max())
print("max news id:", df_small.news_idx.max())

print("small dataframe size:", len(df_small))
df_small.to_csv(r'..\Dataset\small_rating.csv', index=False)
