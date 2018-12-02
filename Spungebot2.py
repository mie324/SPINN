'''
TYPE IN INPUT AND LOCATE PUN
'''
import torch

import torchtext
from torchtext import data
import spacy
import numpy as np
import pandas as pd

from string import punctuation
import sensegram
from wsd import WSD
from gensim.models import KeyedVectors


def tokenizer(strings):
    nlp = spacy.load('en')
    doc = nlp(strings)
    return doc

'''
EVAL WITH RNN MODEL
'''
def evaluate(model, data,length):
    model.eval()
    data = data.long()

    prediction = model(data, torch.tensor([length]))
    prediction = prediction.data.squeeze().numpy()
    prediction = np.argmax(prediction)
    return prediction


TEXT = torchtext.data.Field(sequential=True, tokenize='spacy', include_lengths=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
train, val, test = torchtext.data.TabularDataset.splits(
    path='./', train='train1location.tsv',
    validation='validation1location.tsv', test='test1location.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)], skip_header=True)
TEXT.build_vocab(train,val,test)
TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

sense_vectors_fpath = "model/wiki.txt.clusters.minsize5-1000-sum-score-20.sense_vectors"
word_vectors_fpath = "model/wiki.txt.word_vectors"
sv = sensegram.SenseGram.load_word2vec_format(sense_vectors_fpath, binary=False)
wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")
context_words_max = 20 # change this parameters to 1, 2, 5, 10, 15, 20 : it may improve the results
context_window_size = 10 # this parameters can be also changed during experiments


def filter_scores(array):
    if array.size == 1:  # probably punctuation
        return np.negative(np.ones(10))
    if array.size > 10:   #too many scores, so let's just keep ten
        return array[:10]
    while array.size <10:  # too few scores, pad with -1
        array = np.append(array,np.array([-1]))
    return array

def sensegram(context,word_list,context_window_size):
    sentence_score_array = np.array([])
    for word in word_list:
        wsd_model = WSD(sv, wv, window=context_window_size, lang='en', ignore_case=True)
        word_scores_list = wsd_model.disambiguate(context, word)[1]
        word_scores_list = sorted(word_scores_list, reverse=True)
        word_scores_arr = np.asarray(word_scores_list)
        word_scores_arr = filter_scores(word_scores_arr)
        sentence_score_array = np.concatenate((sentence_score_array, word_scores_arr), axis=0)
    sentence_score_array = np.reshape(sentence_score_array, (len(word_list), 10))
    sentence_df = pd.DataFrame()
    sentence_df = sentence_df.append({'score': sentence_score_array}, ignore_index=True)
    return sentence_df


'''
EVAL WITH SENSEGRAM LOCATION MODEL
'''
def evaluate_sensegram(model, sentence_df):
    model.eval()
    input = sentence_df.values[0][0]
    length = len(input)
    input2 = (torch.from_numpy(input)).float()
    input2 = input2.view(1,input2.shape[0],input2.shape[1])

    prediction = model.forward(input2, torch.tensor([length]))
    prediction = prediction.data.squeeze().numpy()
    prediction = np.argmax(prediction)
    return prediction


def evaluate_sg_detection(model,sentence_df):
    model.eval()
    input = sentence_df.values[0][0]
    length = len(input)
    input2 = (torch.from_numpy(input)).float()
    input2 = input2.view(1, input2.shape[0], input2.shape[1])

    prediction = model.forward(input2, torch.tensor([length]))
    prediction = prediction.data.squeeze().numpy()
    verdict = np.where(prediction>0.5,1,0)
    if int(verdict ==0):
        prediction = 1-prediction
    return float(prediction),int(verdict.squeeze())


while True:
    ans = ''
    while ans=='':
        context = input("Enter a sentence: \n ")
        ans = context.replace(',', '')
        ans = ans.replace('.', '')
        if len(ans.split(sep=" "))<4:
            print('sentence too short')
            ans=''
    ans = tokenizer(ans)
    ans = np.array(ans)
    sentence = ans
    inputs = []
    for i in range(len(ans)):
        inputs.append(TEXT.vocab.stoi[str(ans[i])])
    length = len(ans)
    inputs = np.array(inputs)
    inputs = np.reshape(inputs,(length,1))

    wsd_model = WSD(sv, wv, window=context_window_size, lang='en',ignore_case=True)
    word_list = context.split(' ')
    sentence_df = sensegram(context,word_list,context_window_size)


    list = [ 'rnn','sensegram_rnn']
    j = 0
    prob = 0.0


    model3 = torch.load('model_rnn_sgWSDdetection.pt')
    prob, label3 = evaluate_sg_detection(model3, sentence_df)
    if label3 == 1:
        x = 'pun'
    else:
        x = 'not pun'
    print('WSD model thinks the sentence is {}, with {}% confidence'.format(x, round(prob, 3) * 100))

    model1 = torch.load("model_{}location.pt".format(list[0]))
    label1 = evaluate(model1, torch.from_numpy(inputs),torch.tensor(length))
    print('{} model thinks the punning word is "{}".'.format(list[0].capitalize(),sentence[int(label1)]))

    model2 = torch.load("model_{}location.pt".format(list[1]))
    label2 = evaluate_sensegram(model2, sentence_df)
    print('{} model thinks the punning word is "{}".'.format(list[1].capitalize(),word_list[int(label2)]))



    print('\nTo look up the definition, you can access: https://www.google.ca/search?q={}%20definition\n'.format(sentence[int(label1)]))