import pickle
import numpy as np
from scipy import spatial
with open(r'.\Output\Users\user_data_emb.pkl', 'rb') as handle:        
    others_vector = pickle.load(handle)
topics_one_hot = {}
with open(r'.\Dataset\topic_encodings.pkl', 'rb') as handle:        
    topics_one_hot = pickle.load(handle)
user_topics = ['Apple', 'Facebook', 'Trump']
user_vector=np.zeros(len(next(iter(others_vector.values()))))
#with open(r'.\Output\user_data_embeddings.csv', 'w') as f:
for topic in user_topics:
    try:
        user_vector += topics_one_hot[topic]   
    except:
        user_vector += np.zeros(len(next(iter(others_vector.values()))))
max_sim=0
for key,value in others_vector.items():
    sim= 1 - spatial.distance.cosine(user_vector, value)
    if sim > max_sim:
        max_sim=sim
        sim_user=key
  
with open(r'.\Output\MF\user_emb.pkl', 'rb') as handle:        
    others_emb = pickle.load(handle)
print(others_emb[sim_user])