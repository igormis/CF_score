# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 08:29:13 2018

@author: igorm@falcon 
This is a code in order to find a model that will fit the news with the hidden news vector obtained from SVD
"""
#ARRAYS AND LA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#DATA PROCESSING
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.corpus import stopwords
#WORD EMBEDDINGS
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

#EMBEDDING LAYER AND CNN
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape , Dropout, concatenate
from keras.models import Model
#from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
#from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import regularizers
from keras import backend as K
import stringdist
import pickle

EMBEDDING_FILE=r'.\Model\GoogleNews-vectors-negative300.bin'
EMB_SIZE=300
NUM_WORDS=20000
TRAIN_FILE=r'.\Dataset\train_news_CNN.csv'
TEST_FILE=r'.\Dataset\test_news_CNN.csv'
FS=4
NUM_F=50
DO=0.5
REG=0.01
LR = 0.01
B_SIZE = 32 
EPOCHS = 10

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_embedding_matrix(word_index, vocabulary_size,  word_emb_path = EMBEDDING_FILE, embedd_size=EMB_SIZE, num_words=NUM_WORDS):
	word_vectors = KeyedVectors.load_word2vec_format(word_emb_path, binary=True)
	embedding_matrix = np.zeros((vocabulary_size, embedd_size))
	for word, i in word_index.items():
	    if i>=num_words:
	        continue
	    try:
	        #Here we get the vectors given the word from word_vectors
	        #then we put the vector in the embedding matrix (on row i, where i is the index of the word)
	        embedding_vector = word_vectors[word]
	        embedding_matrix[i] = embedding_vector
	    except KeyError:
	        #If the word was not found we generate random network (change this)
	        embedding_matrix[i]=np.random.normal(-0.25,0.25, embedd_size)
	del(word_vectors)
	return embedding_matrix

def cnn_learn_word_emb(train_file=TRAIN_FILE, test_file =TEST_FILE, filter_sizes = FS, num_filters =NUM_F, drop = DO, reg=REG, learning_rate = LR, batch_s = B_SIZE, num_epochs = EPOCHS):
    #NewsID   NewsText   Output_20_dimensions
    train_data=pd.read_csv(train_file)
    #The test data will be used as new news that are published
    test_data=pd.read_csv(test_file)

    #Build the output from news_embedings from keras
    labels = {}
    with open(r'.\Output\MF\news_emb.pkl', 'rb') as handle:        
        labels = pickle.load(handle)
    
    #Get the size of the news_embeddings vector
    hidden_feature_size = len(next(iter(labels.values())))

    
    #From the train data obtain 20% for validation
    val_data=train_data.sample(frac=0.2,random_state=100)
    
    #The train data is 80%
    train_data=train_data.drop(val_data.index)
    
    #Clean the stop words from the news text
    texts=train_data.Text
    
	#Tokenize the news
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    tokenizer.fit_on_texts(texts)

	#sequence_train contains all the words in the training sequences using the index in word_index
    sequences_train = tokenizer.texts_to_sequences(texts)

	#sequence_valid contains all the words in the validation sequences using the index in word_index
    sequences_valid=tokenizer.texts_to_sequences(val_data.Text)

	#We get the dictionary: key{index}
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

	#(I will use different function for the padding)
    X_train = pad_sequences(sequences_train)
    X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])

	#Here the y_train should be 20 non-categorical float values
    y_train=[]
    for train_id in list(train_data.newsId.values):
        y_train.append(labels[train_id])
    y_train=np.asarray(y_train)
    y_val=[]
    for val_id in list(val_data.newsId.values):
        y_val.append(labels[val_id])
    y_val=np.asarray(y_val)
	#How to get the vectors (we can use also trained on our dataset)
	

	#THe length of distinct words in the whole dataset
    vocabulary_size=min(len(word_index)+1,NUM_WORDS)

	#We create the embedding matrix where the number of rows is the vocab_sze and embedding_sim is 300 
    embedding_matrix = build_embedding_matrix(word_index, vocabulary_size, word_emb_path=EMBEDDING_FILE, embedd_size = EMB_SIZE, num_words=NUM_WORDS)

	#We generate the first embedding layer
    sequence_length = X_train.shape[1]
    print(sequence_length)
    embedding_layer = Embedding(vocabulary_size,
                            EMB_SIZE,
                            weights=[embedding_matrix],
                            input_length=sequence_length,
                            trainable=True)
    print('Embedding done!')
    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)

    conv_0 = Conv1D(num_filters, filter_sizes,activation='relu',kernel_regularizer=regularizers.l2(reg), border_mode='same')(embedding)
	
    maxpool_0 = MaxPooling1D(sequence_length)(conv_0)

    standard_nn_layer = Flatten()(maxpool_0)
    dropout = Dropout(drop)(standard_nn_layer)
    predictions = Dense(hidden_feature_size)(dropout)
	
    model = Model(inputs, predictions)
    sgd = SGD(lr=learning_rate)
    model.compile(optimizer=sgd, loss=RMSE)#loss='cosine_proximity')
    
    callbacks = [EarlyStopping(monitor='val_loss')]
    model.fit(X_train, y_train, batch_size=batch_s, epochs=num_epochs, verbose=1, validation_data=(X_val, y_val),
         callbacks=callbacks)  # starts training
    #print(model.history.keys())
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save(r'.\Model\cnn_news')
    f = open(r'.\Model\news_padding.pkl', 'wb')
    pickle.dump(X_train.shape[1],f)
    f.close()
    f = open(r'.\Model\num_words.pkl', 'wb')
    pickle.dump(NUM_WORDS,f)
    f.close()
    f = open(r'.\Model\tokenizer.pkl', 'wb')
    pickle.dump(tokenizer,f)
    f.close()
	#Predict the output vector using the model
    sequences_test=tokenizer.texts_to_sequences(test_data.Text)
    X_test = pad_sequences(sequences_test,maxlen=X_train.shape[1])
    y_pred=model.predict(X_test)
    print(y_pred[0])
	
cnn_learn_word_emb()

	