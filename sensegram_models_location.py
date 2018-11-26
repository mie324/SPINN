import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

vocab_size = 6247
torch.manual_seed(77)


# class Baseline(nn.Module):
#     def __init__(self, embedding_dim, vocab):
#         super(Baseline, self).__init__()
#         self.embedding_layer = nn.Embedding(vocab_size,embedding_dim) ## initialize
#         self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
#         self.fc1 = nn.Linear(embedding_dim,1)
#
#     def forward(self, x, lengths):
#         x = self.embedding_layer(x)
#         x = torch.mean(x,0)
#         x = self.fc1(x)
#         activation = nn.Sigmoid()
#         x = activation(x)
#         return x

#  BASELINE
class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(embedding_dim, 1, bias=True)

    def forward(self, x, lengths=None):
        x = self.embed(x)
        x = torch.sum(x, dim=0)
        lengths = 1 / lengths.float()
        x = torch.transpose(x.float(), 0, 1) * lengths
        x = torch.transpose(x, 0, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x.squeeze())
        return x


# CNN AND RNN COMBINED
class CRNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CRNN, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.fc1 = nn.Linear(n_filters, 1)
        self.gru = nn.GRU(n_filters, n_filters, dropout=0.1, batch_first=True)

    def forward(self, x, lengths):
        x = torch.transpose(x, 0, 1)  # make it into dim = (bs, sentence length)
        x = self.embedding_layer(x)  # make it into dim = (bs,sentence length, embedding dim)
        shape = x.shape
        x = x.view(shape[0], 1, shape[1], shape[2])
        x1 = self.conv1(x)  # do convolution using kernel size 2, get dim = (bs,filter num, feature map size,1)
        x2 = self.conv2(x)  # do convolution using kernel size 4
        relu = nn.ReLU()
        x1 = relu(x1).squeeze()
        x2 = relu(x2).squeeze()
        x = torch.cat((x1, x2), 2)
        x = torch.transpose(x, 1, 2)
        x = self.gru(x)
        x = x[1]
        x = x.squeeze()  # get dim = (bs,100)
        x = self.fc1(x)
        sigmoid = nn.Sigmoid()
        x = sigmoid(x)
        return x


# LSTM
class biRNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(biRNN, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(embedding_dim, hidden_dim, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, 1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.3, bidirectional=True)

    def forward(self, x, lengths):  # pass in x and x_length

        x = self.embedding_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        x_for, x_back = self.lstm(x)[1]  # just take the hidden state of the output
        x_for = x_for[1].squeeze()
        x_back = x_back[1].squeeze()
        x = torch.cat((x_for, x_back), 1)
        x = x.squeeze()

        x = self.fc1(x)
        activation = nn.Sigmoid()
        x = activation(x)
        return x


'''
        total = []
        for i in range(len(lengths)):
            shuff = []
            idx = torch.randperm(lengths[i])
            for j in range (len(idx)):
                shuff = shuff.append(int(x[i][idx[j]]))
            total.append(shuff)
        total = np.array(total)
        x = torch.from_numpy(total)
'''


# CNN MODEL
class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.fc1 = nn.Linear(n_filters * 2, 1)

    def forward(self, x, lengths):
        x = torch.transpose(x, 0, 1)  # make it into dim = (bs, sentence length)
        x = self.embedding_layer(x)  # make it into dim = (bs,sentence length, embedding dim)
        shape = x.shape
        x = x.view(shape[0], 1, shape[1], shape[2])
        x1 = self.conv1(x)  # do convolution using kernel size 2, get dim = (bs,filter num, feature map size,1)
        x2 = self.conv2(x)  # do convolution using kernel size 4
        relu = nn.ReLU()
        x1 = relu(x1)
        x2 = relu(x2)
        x1 = torch.max(x1, 2)[0]  # max pool of results using kernel size = 2,  get dim = (bs, filter num, 1)
        x2 = torch.max(x2, 2)[0]  # max pool of results using kernel size = 4
        x = torch.cat((x1, x2), 1)
        x = x.squeeze()  # get dim = (bs,100)
        x = self.fc1(x)
        sigmoid = nn.Sigmoid()
        x = sigmoid(x)
        return x


# RNN MODEL
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        #self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.LSTM(embedding_dim, hidden_dim, dropout=0.1,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 5)
        self.fc2 = nn.Linear(5, 1)
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths):  # pass in x and x_length
        #x = self.embedding_layer(x)
        #x = nn.utils.rnn.pack_padded_sequence(x,lengths)
        x = self.gru(x)
        x = x[0]  # take the hidden states from all of the cells (i.e. from every token in the sentence)
        total_len = x.shape[1]
        #x = torch.transpose(x, 0, 1)
        x = x.contiguous()
        x = x.view(-1, self.hidden_dim)
        tanh = nn.Tanh()
        x = tanh(x)
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)

        x = x.view(-1, total_len)
        x = F.softmax(x, dim=1)
        return x
