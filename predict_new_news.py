# -*- coding: utf-8 -*-
from keras.models import load_model
from keras import backend as K
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

interpolate = interp1d([1,5],[0,40])
def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
CNN_news_model = load_model(r'.\Model\cnn_news', custom_objects={'RMSE': RMSE})

padding=0
with open(r'.\Model\news_padding.pkl', 'rb') as handle:        
    padding = pickle.load(handle)
    
NUM_WORDS=0
with open(r'.\Model\num_words.pkl', 'rb') as handle:        
    NUM_WORDS = pickle.load(handle)
with open(r'.\Model\tokenizer.pkl', 'rb') as handle:        
    tokenizer = pickle.load(handle)
with open(r'.\Model\num_words.pkl', 'rb') as handle:        
    NUM_WORDS = pickle.load(handle)

with open(r'.\Output\MF\medium.pkl', 'rb') as handle:        
    mu = pickle.load(handle)

with open(r'.\Output\MF\user_emb.pkl', 'rb') as handle:        
    user_emb = pickle.load(handle)

with open(r'.\Output\MF\user_bias.pkl', 'rb') as handle:        
    user_bias = pickle.load(handle)

def ret_news_vector(test_data):
    sequences_test=tokenizer.texts_to_sequences(test_data)
    X_test = pad_sequences(sequences_test,maxlen=padding)
    y_pred=CNN_news_model.predict(X_test)
    return y_pred[0]
#test_data=pd.read_csv(r'.\Dataset\test_news_CNN.csv')
import time
t = time.process_time()
res=ret_news_vector('Jury awards $55 million to Erin Andrews in lawsuit over nude video taken at hotel')
result_dict={}
for key,value in user_emb.items():
        result_dict[key]=np.exp(sum(value*res)+user_bias[key] + mu)*100/np.exp(5)
elapsed_time = time.process_time() - t
print(elapsed_time)

        