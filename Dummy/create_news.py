# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
train_data=pd.read_csv('train_CNN.csv', sep=";")
test_data=pd.read_csv('test_CNN.csv', sep = ";")
ratings = pd.read_csv('small_rating.csv')
ids = list(set(ratings.newsId.values))
ids_train = ids[0:int(len(ids)*0.8)]
ids_test = ids[int(len(ids)*0.8):]
col_names = ['newsId', 'Text']
train_data = train_data.title.sample(len(ids_train))
data_tuples = list(zip(ids_train,list(train_data)))
new_train = pd.DataFrame(data_tuples, columns=['newsId','Text'])
test_data = test_data.title.sample(len(ids_test))
data_tuples = list(zip(ids_test,list(test_data)))
new_test = pd.DataFrame(data_tuples, columns=['newsId','Text'])
    
new_train.to_csv(r'..\Dataset\train_news_CNN.csv', index=False)
new_test.to_csv(r'..\Dataset\test_news_CNN.csv', index=False)