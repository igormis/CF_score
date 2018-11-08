# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
#from builtins import range, input


import pandas as pd

#The file should be in this format:
#"userId","newsId","rating","timestamp"
#where rating is between 1 (bad) and 5 (good)
df = pd.read_csv(r'..\Dataset\rating.csv')

# note:
# user ids are ordered sequentially from 1..N
# there should be no no missing numbers
# news ids are integers from 1..M
# NOT all news ids appear

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for news ids
unique_news_ids = set(df.newsId.values)
news2idx = {}
count = 0
for news_id in unique_news_ids:
  news2idx[news_id] = count
  count += 1

# add them to the data frame
# takes awhile
df['news_idx'] = df.apply(lambda row: news2idx[row.newsId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('..\Dataset\edited_rating.csv', index=False)