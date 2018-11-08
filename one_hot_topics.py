import pandas as pd

#The file should be consisted only of the existing topics
df = pd.read_csv(r'..\Dataset\topics.csv', header=None)
df.columns = ['Topics']

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

all_topics = df.Topics.values.tolist()
#Label encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(all_topics)

#One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#Create dictionary (key, value) = (topic, one-hot encoding vector)
topic_enc_dict={}
for i in range(0,len(all_topics)):
    topic_enc_dict[all_topics[i]]=onehot_encoded[i]
import pickle
#It generates one-hot econding for the topics
f = open(r'..\Dataset\topic_encodings.csv', 'wb')
pickle.dump(topic_enc_dict,f)
f.close()
