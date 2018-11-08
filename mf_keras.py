from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

def mf_keras(rating_file=r'.\Dataset\small_rating.csv', K=20, num_epochs=5, reg=0., dropout=0.2):
  #We need the rating file with (User_id, Infoitem_id and the score)
  #K is the dimensionality of the latent vector
  df = pd.read_csv(rating_file)
  #take 10000 random samples
  #df = df.sample(10000)
  a=df.newsIdx=49
  #number of users
  #N=len(set(df.userId.tolist()))
  #number of newsa
  #M=len(set(df.movie_idx.tolist()))
  
  
  N = df.userId.max() + 1 #number of cityfalcon users
  M = df.news_idx.max() + 1 # number of news

  #Split into train and test
  df=shuffle(df)
  cutoff = int(0.8*len(df))
  df_train = df.iloc[:cutoff]
  df_test = df.iloc[cutoff:]

  #this is the biased score in the training set
  mu = df_train.rating.mean()
  epochs = num_epochs
  reg=reg

  #Keras Model
  u = Input(shape=(1,))
  m = Input(shape=(1,))
  #User embeddings
  u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u) # (N, 1, K)
  #News Embeddings
  m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m) # (N, 1, K)

  u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
  m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (M, 1, 1)

  x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)

  x = Add()([x, u_bias, m_bias])
  x = Flatten()(x) # (N, 1)
    ##### side branch
  u_embedding = Flatten()(u_embedding) # (N, K)
  m_embedding = Flatten()(m_embedding) # (N, K)
  y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
  y = Dense(400)(y)
  y = Activation('relu')(y)
  y = Dropout(dropout)(y)
  y = Dense(1)(y)


  ##### merge
  x = Add()([x, y])
    
    
    
  model = Model(inputs=[u, m], outputs=x)
  model.compile(
    loss='mse',
    # optimizer='adam',
    # optimizer=Adam(lr=0.01),
    optimizer=SGD(lr=0.08, momentum=0.9),
    metrics=['mse'],
  )

  r = model.fit(
    x=[df_train.userId.values, df_train.news_idx.values],
    y=df_train.rating.values - mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
      [df_test.userId.values, df_test.news_idx.values],
      df_test.rating.values - mu
    )
  )
  print(model.summary())
  
  #get the user and news embeddings (hidden features)
  user_embedding_learnt = model.get_layer(name='embedding_1').get_weights()
  news_embeddings_learnt = model.get_layer(name='embedding_2').get_weights()
  user_embedding_learnt = np.squeeze(user_embedding_learnt, axis=(2,))
  news_embeddings_learnt = np.squeeze(news_embeddings_learnt, axis=(2,))
  
  #Get the biases
  user_bias_learnt = model.get_layer(name='embedding_3').get_weights()
  news_bias_learnt = model.get_layer(name='embedding_4').get_weights()
  user_bias_learnt = np.squeeze(user_bias_learnt, axis=(2,))
  news_bias_learnt = np.squeeze(news_bias_learnt, axis=(2,)) 
  #Calculate the score
  #score=np.matmul(user_embedding_learnt,np.transpose(news_embeddings_learnt))

  # plot losses
  plt.plot(r.history['loss'], label="train loss")
  plt.plot(r.history['val_loss'], label="test loss")
  plt.legend()
  plt.show()

  # plot mse
  plt.plot(r.history['mean_squared_error'], label="train mse")
  plt.plot(r.history['val_mean_squared_error'], label="test mse")
  plt.legend()
  plt.show()
  #save the model
  model.save(r'.\Model\MF_keras_model')
  user_emb={}
  user_biases={}
  for i in range(0,N):
      user_emb[i]=list(user_embedding_learnt[i])
      user_biases[i]=user_bias_learnt[i]
  news_emb={}
  news_biases={}
  for i in range(0,M):
      news_id=list(df.loc[df['news_idx'] == i, 'newsId'])[0]
      news_emb[news_id]=list(news_embeddings_learnt[i])
      news_biases[news_id]=news_bias_learnt[i]

  f = open(r'.\Output\MF\user_emb.pkl','wb')
  pickle.dump(user_emb,f)
  f.close()
  f = open(r'.\Output\MF\user_bias.pkl','wb')
  pickle.dump(user_biases,f)
  f.close()
  f = open(r'.\Output\MF\news_emb.pkl','wb')
  pickle.dump(news_emb,f)
  f.close()
  f = open(r'.\Output\MF\news_bias.pkl','wb')
  pickle.dump(news_biases,f)
  f.close()  
  f = open(r'.\Output\MF\medium.pkl','wb')
  pickle.dump(mu,f)
  f.close() 
#  with open(r'.\Output\MF\user_emb.csv', 'w') as f:
#      for i in range(0,N):
#          emb = str(list(user_embedding_learnt[i]))
#          f.write(str(i) + ', ' + emb + '\n')
#  with open(r'.\Output\MF\user_bias.csv', 'w') as f:
#      for i in range(0,N):
#          bias = str((user_bias_learnt[i]))
#          f.write(str(i) + ', ' + bias + '\n')  
#  with open(r'.\Output\MF\news_emb.csv', 'w') as f:
#      for i in range(0,M):
#          list(df.loc[df['news_idx'] == i, 'newsId'])[0]
#          emb = str(list(news_embeddings_learnt[i]))
#          f.write(str(i) + ', ' + emb + '\n')    
#  with open(r'.\Output\MF\news_bias.csv', 'w') as f:
#      for i in range(0,M):
#          list(df.loc[df['news_idx'] == i, 'newsId'])[0]
#          bias = str((news_bias_learnt[i]))
#          f.write(str(i) + ', ' + bias + '\n')
#  with open(r'.\Output\MF\medium.csv', 'w') as f:
#      f.write(str(mu))
mf_keras(r'.\Dataset\small_rating.csv', K=20, num_epochs=1, reg=0., dropout=0.2)



