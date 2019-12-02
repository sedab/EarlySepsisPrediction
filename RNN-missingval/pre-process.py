import pickle
import pandas as pd
import os
import numpy as np


os.chdir('/scratch/sb3923/time_series/data')

# load in data
filename = 'data.pickle'
with open(filename, 'rb') as f:
    vitals_df = pickle.load(f)
    labs_df = pickle.load(f)
    demogs_df = pickle.load(f)
    labels_df = pickle.load(f)
    
    
#concat all data
demogs_df['pat_id2']=demogs_df['pat_id']
demogs_df['hour2']=demogs_df['hour']
demogs_df=demogs_df.drop(['pat_id'], axis=1)
demogs_df=demogs_df.drop(['hour'], axis=1)

labs_df['pat_id3']=labs_df['pat_id']
labs_df['hour3']=labs_df['hour']
labs_df=labs_df.drop(['pat_id'], axis=1)
labs_df=labs_df.drop(['hour'], axis=1)


all_df_ = pd.concat([vitals_df, demogs_df], axis= 1)
all_df__ = pd.concat([all_df_, labs_df], axis= 1)
all_df = pd.concat([all_df__, labels_df['SepsisLabel']], axis= 1)

all_df=all_df.drop(['pat_id2'], axis=1)
all_df=all_df.drop(['pat_id3'], axis=1)
all_df=all_df.drop(['hour2'], axis=1)
all_df=all_df.drop(['hour3'], axis=1)


patients = all_df['pat_id'].unique()
empty_df = pd.DataFrame(np.nan, index=list(range(0,48)), columns=all_df.columns)


pat_data = pd.concat([all_df['pat_id']==patients[0],empty_df], axis= 0)[:48]
#fix the hour and the patient id
pat_data['hour']=list(range(0,48))
pat_data['pat_id']=all_df[all_df['pat_id']==patients[0]]['pat_id'][0]
all_pat_data=pat_data
n=0
for p in patients[1:]:
    if(len(all_df[all_df['pat_id']==p])>12):
        pat_data = pd.concat([all_df[all_df['pat_id']==p],empty_df], axis= 0)[:48]
        #fix the hour and the patient id
        pat_data['hour']=list(range(0,48))
        pat_data['pat_id']=all_df[all_df['pat_id']==p]['pat_id'].unique()[0]
        all_pat_data=pd.concat([all_pat_data,pat_data], axis= 1)
        #n=n+1
    
    
filename = '/scratch/sb3923/time_series/data/data_proc.pickle'
with open(filename, 'wb') as f:
    pickle.dump(all_pat_data, f)
    
    
    
    
    
