"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""
import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import numpy as np

from string import punctuation


def tokenizer(strings):
    nlp = spacy.load('en')
    doc = nlp(strings)

    return doc


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

while True:
    ans = ''
    while ans=='':
        ans = input("Enter a sentence \n ")
        ans = ans.replace(',', '')
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
    list = [ 'rnn']
    j = 0
    prob = 0.0
    for i in list:
        j +=1
        model = torch.load("model_{}location.pt".format(i))
        label = evaluate(model, torch.from_numpy(inputs),torch.tensor(length))
        print('{} model think the punning word is "{}".'.format(i.capitalize(),sentence[int(label)]))
    print('To look up the definition you can access: https://www.google.ca/search?q={}%20definition'.format(sentence[int(label)]))