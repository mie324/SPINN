import numpy as np
import spacy
import pandas as pd
# CHANGE SPLIT RATIO HERE
train_num = 820
val_num = 174
test_num = 261

#task = 'detection'
task = 'location'

orig_data_df = pd.read_csv('sentences_balanced_2510.csv')   # save original tsv file to pd dataframe
orig_data_df['text']=orig_data_df['text'].str.replace('.', '', regex=True)
orig_data_df['text']=orig_data_df['text'].str.replace(',', '', regex=True)
orig_data_df['text']=orig_data_df['text'].str.replace(" ' ", "'", regex=True)
orig_data_df['text']=orig_data_df['text'].str.replace("?", "", regex=True)
orig_data_df['text']=orig_data_df['text'].str.replace("!", "", regex=True)
orig_data_df['text']=orig_data_df['text'].str.replace("''", "", regex=True)




# separate into 2 groups and shuffle
pun_df = orig_data_df.iloc[0:1255,:]  # contains 1255 puns
none_df = orig_data_df.iloc[1255::,:]   # 1255 non-puns
pun_df_shuffled = pun_df.sample(frac=1,random_state=77)
none_df_shuffled = none_df.sample(frac=1,random_state=77)

# split into 3 sets and merge back together
train_df = pd.concat([pun_df_shuffled.iloc[0:train_num,:], none_df_shuffled.iloc[0:train_num,:]], axis=0)
train_df = train_df.sample(frac=1,random_state=77)
# only use text and detection labels for SUBTASK 1
train_df_1 = train_df[['text',task]]
train_df_1.to_csv('train1{}.tsv'.format(task), sep='\t',index=False)  # must set index to False!!! otherwise format incorrect

val_df = pd.concat([pun_df_shuffled.iloc[train_num:(train_num+val_num),:], none_df_shuffled.iloc[train_num:(train_num+val_num),:]], axis=0)
val_df = val_df.sample(frac=1,random_state=77)
val_df_1 = val_df[['text',task]]
val_df_1.to_csv('validation1{}.tsv'.format(task), sep='\t',index=False)

test_df = pd.concat([pun_df_shuffled.iloc[(train_num+val_num)::,:], none_df_shuffled.iloc[(train_num+val_num)::,:]], axis=0)
test_df = test_df.sample(frac=1,random_state=77)
test_df_1 = test_df[['text',task]]
test_df_1.to_csv('test1{}.tsv'.format(task), sep='\t',index=False)
