#Artificial Neural Networks

#Installing Theano


#Installing Tensorflow


#Installing Keras - wraps Theano and Tensorflow

#Part 1 - Data Preprocessing
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential # initialize the ANN
from keras.layers import Dense #Create the layers in the ANN
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV #tunning the hyperparameters
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise

# Importing the dataset
with open(r'.\Output\Users\user_data_emb.pkl', 'rb') as handle:        
    X_in = pickle.load(handle)
with open(r'.\Output\MF\user_emb.pkl', 'rb') as handle:        
    y_out = pickle.load(handle)
X=[]
for key,value in X_in.items():
    X.append(value)
X=np.array(X)
input_d = X.shape[1]
Z=pairwise.cosine_similarity(X)

y=[]
for key,value in y_out.items():
    y.append(value)
y=np.array(y)
output_d = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
def baseline_model():
    #initialising the ANN
    classifier = Sequential()
    #Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(output_dim=500, init='uniform', activation='tanh', input_dim=input_d ))
    classifier.add(Dropout(p=0.3)) #increase if there is overfitting - but be careful with underfitting
    #Adding the second hidden layer
    classifier.add(Dense(output_dim=100, init='uniform', activation='tanh' ))
    classifier.add(Dropout(p=0.3))
    #Adding the output layer (Softmax more than two categories)
    classifier.add(Dense(output_dim=output_d, init='uniform', activation='sigmoid' ))
    
    #Compile the ANN - adam is stochastic gradient algorithm
    #THe loss function for two categories is binary_crossentropy
    #if there are more than two categories is categorical_crossentropy
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return classifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=5, verbose=0)
#Fitting the ANN to the training set
#classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100 )

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#Predict for a new customer - must be a nx1 vectorlike a single observation
#new_prediction = classifier.predict(sc.fit_transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#new_prediction = (new_prediction > 0.5)
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)