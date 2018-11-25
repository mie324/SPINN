import numpy as np
import spacy
import pandas as pd
# CHANGE SPLIT RATIO HERE
train_num = 820
val_num = 174
test_num = 261

task = 'location'
'''
pun_df = pd.read_table('sensegram_scores.tsv')
pun_df_shuffled = pun_df.sample(frac=1,random_state=77)

# split into 3 sets
train_df = pun_df_shuffled.iloc[0:train_num,:]
train_df = train_df.sample(frac=1,random_state=77)
train_df_1 = train_df[['score',task]]
train_df_1.to_csv('train1{}_sensegram.tsv'.format(task), sep='\t',index=False)  # must set index to False!!! otherwise format incorrect

val_df = pun_df_shuffled.iloc[train_num:(train_num+val_num),:]
val_df = val_df.sample(frac=1,random_state=77)
val_df_1 = val_df[['score',task]]
val_df_1.to_csv('validation1{}_sensegram.tsv'.format(task), sep='\t',index=False)

test_df = pun_df_shuffled.iloc[(train_num+val_num)::,:]
test_df = test_df.sample(frac=1,random_state=77)
test_df_1 = test_df[['score',task]]
test_df_1.to_csv('test1{}_sensegram.tsv'.format(task), sep='\t',index=False)
'''

pun_df = pd.read_pickle('sensegram_scores.pkl')
pun_df_shuffled = pun_df.sample(frac=1,random_state=77)

train_df = pun_df_shuffled.iloc[0:train_num,:]
train_df = train_df.sample(frac=1,random_state=77)
train_df_1 = train_df[['score',task]]
train_df_1.to_pickle('train1{}_sensegram.pkl'.format(task))  # must set index to False!!! otherwise format incorrect

val_df = pun_df_shuffled.iloc[train_num:(train_num+val_num),:]
val_df = val_df.sample(frac=1,random_state=77)
val_df_1 = val_df[['score',task]]
val_df_1.to_pickle('validation1{}_sensegram.pkl'.format(task))

test_df = pun_df_shuffled.iloc[(train_num+val_num)::,:]
test_df = test_df.sample(frac=1,random_state=77)
test_df_1 = test_df[['score',task]]
test_df_1.to_pickle('test1{}_sensegram.pkl'.format(task))