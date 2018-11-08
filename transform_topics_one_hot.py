import pandas as pd
import numpy as np
import pickle 
df_users=pd.read_csv(r'.\Dataset\users_topics.csv', header=None)
df_users.columns = ['Id', 'Topics']

topics_one_hot = {}
with open(r'.\Dataset\topic_encodings.pkl', 'rb') as handle:        
    topics_one_hot = pickle.load(handle)
   
N = len(topics_one_hot)
columns = ['Id', 'Vector']
new_df = pd.DataFrame(columns=columns)
user_vector={}
#with open(r'.\Output\user_data_embeddings.csv', 'w') as f:
for row in df_users.values:
    i=0
    all_terms_per_user = str(row[1])
    terms = all_terms_per_user.split(';')
    term_vector = []
    user_vector[row[0]]=np.zeros(N)
    for i in range(0,len(terms)):
        user_vector[row[0]] += topics_one_hot[terms[i].strip()]
#Save the one hot encodings per user        
import pickle
f = open(r'.\Output\Users\user_data_emb.pkl','wb')
pickle.dump(user_vector,f)
f.close()



