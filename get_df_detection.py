import pandas as pd
import numpy as np
import torch
import sensegram
from wsd import WSD
from gensim.models import KeyedVectors

orig_data_df = pd.read_csv('sentences_balanced_revised.csv')   # save original tsv file to pd dataframe
pun_df = orig_data_df.iloc[0:1255,:]  # contains 1255 puns


sensegram_df = pd.DataFrame()

# make sure the sentences csv file is in the same directory!
sense_vectors_fpath = "model/wiki.txt.clusters.minsize5-1000-sum-score-20.sense_vectors"
word_vectors_fpath = "model/wiki.txt.word_vectors"

# Load models (takes long time)
sv = sensegram.SenseGram.load_word2vec_format(sense_vectors_fpath, binary=False)
wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")
context_words_max = 20 # change this paramters to 1, 2, 5, 10, 15, 20 : it may improve the results
context_window_size = 10 # this parameters can be also changed during experiments


def filter_scores(array):
    if array.size == 1:  # probably punctuation
        return np.negative(np.ones(10))
    if array.size > 10:   #too many scores, so let's just keep ten
        return array[:10]
    while array.size <10:  # too few scores, pad with -1
        array = np.append(array,np.array([-1]))
    return array


ignore_case = True
lang = "en"  # to filter out stopwords
for i in range(2510):
    sentence = orig_data_df['text'][i]
    detection = orig_data_df['detection'][i]
    word_list = sentence.split(' ')
    sentence_score_array = np.array([])
    for word in word_list:
        wsd_model = WSD(sv, wv, window=context_window_size, lang=lang,ignore_case=ignore_case)
        word_scores_list = wsd_model.disambiguate(sentence, word)[1]
            #print(word_scores_list)
        word_scores_list = sorted(word_scores_list,reverse=True)
        word_scores_arr = np.asarray(word_scores_list)
        word_scores_arr = filter_scores(word_scores_arr)
        sentence_score_array = np.concatenate((sentence_score_array,word_scores_arr),axis=0)
    sentence_score_array = np.reshape(sentence_score_array,(len(word_list),10))
    sensegram_df = sensegram_df.append({'score':sentence_score_array,'detection':detection},ignore_index=True)

    '''
    sentence_tensor = torch.from_numpy(sentence_score_array)
    sentence_tensor = sentence_tensor.view(1,len(word_list),10)
    if i==0:
        stacked_score_tensor = sentence_tensor
    else:
        stacked_score_tensor = torch.cat((stacked_score_tensor,sentence_tensor),0)
    '''

    print('{}\n'.format(i))

sensegram_df.to_csv('sensegram_scores_detection.tsv', sep='\t',index=False)
sensegram_df.to_pickle('sensegram_scores_detection.pkl')

#score_array = stacked_score_tensor.numpy()
#np.save('sensegram_score_arr.npy',score_array)


