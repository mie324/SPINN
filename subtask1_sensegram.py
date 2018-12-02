import torch
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import argparse
import os
import numpy as np
from subtask1_model import *
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
torch.manual_seed(77)
eval_bs,bs,lr,MaxEpochs,model_type,num_filters,rnn_hidden_dim,embed_dim,save_csv = None,None,None,None,None,None,None,None,None

from test import *

'''
DEFINE DATASET CLASS
'''

class Sensegram_Data(data.Dataset):
    def __init__(self, score, location):
        self.score = score
        self.location = location

    def __len__(self):
        return len(self.score)

    def __getitem__(self, index):
        location = self.location[index]
        score = self.score[index]
        # e.g. get item at index 1 gives us sentence 1's score array of dim (sentence length, 10)
        return location, score


'''
CUSTOM COLLATE FUNCTION FOR DATALOADER
'''


def my_collate(batch):
    data = [item[1] for item in batch]
    # data [i][0] gives sentence i's 2d score matrix
    target = [item[0] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


'''
LOAD DATA INTO MODEL
'''


def load_data():
    global bs,eval_bs
    train_df = pd.read_pickle('train1detection_sensegram.pkl')
    val_df = pd.read_pickle('validation1detection_sensegram.pkl')
    test_df = pd.read_pickle('test1detection_sensegram.pkl')
    train_dataset = Sensegram_Data(train_df[['score']].values,train_df['detection'].values)
    val_dataset = Sensegram_Data(val_df[['score']].values,val_df['detection'].values)
    test_dataset = Sensegram_Data(test_df[['score']].values,test_df['detection'].values)
    train_loader = DataLoader(train_dataset,batch_size=bs,shuffle=True,collate_fn=my_collate)
    val_loader = DataLoader(val_dataset,batch_size=eval_bs,shuffle=False,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=eval_bs,shuffle=False,collate_fn=my_collate)
    return train_loader,val_loader,test_loader


'''
LOAD MODEL
'''


def load_model(learning_rate):
    global model_type, rnn_hidden_dim, embed_dim
    if model_type == 'baseline':
        model = Baseline(embed_dim)# will use TEXT.vocab
    elif model_type == 'rnn':
        model = RNN(embed_dim,rnn_hidden_dim)
    elif model_type == 'cnn':
        model = CNN(embed_dim, num_filters, np.array([2,4]))
    elif model_type == 'crnn':
        model = CRNN(embed_dim, num_filters, np.array([2,4]))
    elif model_type == 'birnn':
        model = biRNN(embed_dim, rnn_hidden_dim)
    elif model_type == 'rnn_sg':
        model = RNNSG(embed_dim, rnn_hidden_dim)
    loss_fxn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-04)
    return model, loss_fxn, optimizer



'''
EVALUATE RESULTS WITH VALIDATION SET
'''
def evaluate_model(model,loss_fxn,val_loader):
    global eval_bs
    total_val_corr = 0
    accum_loss = 0
    vali_samples = len(val_loader.dataset)
    model.eval()
    for i,(score,location) in enumerate(val_loader):
        x, y = score, location
        x_lengths = np.array([], dtype=int)

        for j in range(eval_bs):
            x_lengths = np.append(x_lengths, int(len(score[j][0])))

        max_length = max(x_lengths)
        x_pad = np.zeros((eval_bs, max_length, embed_dim))

        for j in range(eval_bs):
            orig_score_arr = score[j][0]
            pad = max_length - int(len(orig_score_arr))
            if pad == 0:
                x_pad[j, :, :] = orig_score_arr
            elif pad > 0:
                pad_arr = np.negative(np.ones((pad, 10)))
                new_score_arr = np.concatenate((orig_score_arr, pad_arr), axis=0)
                x_pad[j, :, :] = new_score_arr

        x_pad2 = (torch.from_numpy(x_pad)).float()
        predictions = model.forward(x_pad2, x_lengths)
        loss = loss_fxn(input=predictions.squeeze(), target=y.float())
        accum_loss += loss.item()

        # compare batch predictions to labels
        label_ans = predictions.data.squeeze().numpy()
        label_ans = np.where(label_ans > 0.5, 1, 0)
        actual_res = y.int().numpy()
        label_ans = np.where(label_ans == actual_res, 1, 0)
        total_val_corr += sum(label_ans)

    eval_accuracy = float(total_val_corr)/vali_samples
    return accum_loss/(i+1),eval_accuracy  # return this as the evaluation accuracy


def evaluate_model_test(model, loss_fxn, val_iter):
    global bs
    total_val_corr = 0
    accum_loss = 0
    vali_samples = len(val_iter.dataset)
    model.eval()
    Fpos = 0
    Fneg = 0
    Tpos = 0
    Tneg = 0
    confusion_mat = np.array([[0, 0],[0, 0]])
    for i, data in enumerate(val_iter):
        (x, x_lengths), y = data.text, data.detection
        predictions = model.forward(x, x_lengths)
        loss = loss_fxn(input=predictions.squeeze(), target=y.float())
        accum_loss += loss.item()

        # compare batch predictions to labels
        pred_res = predictions.data.squeeze().numpy()
        actual_res = y.int().numpy()
        for j in range(len(y)):
            if pred_res[j] <= 0.5 and actual_res[j] == 0:
                total_val_corr += 1
                Tneg += 1
            elif pred_res[j] <= 0.5 and actual_res[j] == 1:
                Fneg += 1
            elif pred_res[j] > 0.5 and actual_res[j] == 1:
                total_val_corr += 1
                Tpos += 1
            else:
                Fpos += 1
        confusion_mat = np.array([[Tpos,Fneg],[Fpos,Tneg]])

    eval_accuracy = float(total_val_corr) / vali_samples
    return accum_loss / (i + 1), eval_accuracy, confusion_mat  # return this as the evaluation accuracy


'''
TRAINING LOOP
'''
def main(args):
    dict_args = vars(args)
    global eval_bs,bs, lr, MaxEpochs, model_type, eval_every, num_filters, rnn_hidden_dim, embed_dim, save_csv
    bs = dict_args['train_batch_size']
    eval_bs = dict_args['eval_batch_size']
    lr = dict_args['lr']
    MaxEpochs = dict_args['epochs']
    model_type = dict_args['model']
    num_filters = dict_args['num_filt']
    rnn_hidden_dim = dict_args['rnn_hidden_dim']
    embed_dim = dict_args['emb_dim']
    eval_every = int(1640 / bs)
    save_csv = dict_args['save_csv']

    train_loader,val_loader,test_loader = load_data()
    model, loss_fxn, optimizer = load_model(lr)

    batch_done = 0
    batch_number_log = np.zeros((MaxEpochs, 1), dtype=float)
    train_accuracy_log = np.zeros((MaxEpochs, 1), dtype=float)
    vali_accuracy_log = np.zeros((MaxEpochs, 1), dtype=float)


    for epoch in range(MaxEpochs):
        accum_loss = float(0)
        total_correct = 0

        for i,(score,location) in enumerate(train_loader):
            x, y = score, location
            x_lengths = np.array([],dtype=int)

            for k in range(bs):
                x_lengths = np.append(x_lengths,int(len(score[k][0])))

            max_length = max(x_lengths)
            x_pad = np.zeros((bs,max_length,embed_dim))

            for j in range(bs):
                orig_score_arr = score[j][0]
                pad = max_length - int(len(orig_score_arr))
                if pad ==0:
                    x_pad[j,:,:] = orig_score_arr
                elif pad > 0:
                    pad_arr = np.negative(np.ones((pad,10)))
                    new_score_arr = np.concatenate((orig_score_arr,pad_arr),axis=0)
                    x_pad[j,:,:] = new_score_arr


            optimizer.zero_grad()
            x_pad2 = (torch.from_numpy(x_pad)).float()

            predictions = model.forward(x_pad2,x_lengths)

            loss = loss_fxn(input=predictions.squeeze(),target=y.float())
            accum_loss += loss.item()

            loss.backward()
            optimizer.step()

            # comopare batch predictions to labels
            label_ans = predictions.data.squeeze().numpy()
            actual_res = y.int().numpy()
            label_ans = np.where(label_ans>0.5,1,0)
            label_ans = np.where(label_ans == actual_res, 1, 0)
            total_correct += sum(label_ans)

            if batch_done == 0:
                print('\n ---------------- T R A I N I N G   I N   P R O G R E S S ----------------\n')

            batch_done += 1

        vali_loss, vali_accuracy = evaluate_model(model, loss_fxn, val_loader)
        train_loss = accum_loss/(i+1)
        train_accuracy = total_correct / len(train_loader.dataset)

        # Record validation and train accuracy, will be saved to csv
        vali_accuracy_log[epoch] = vali_accuracy
        batch_number_log[epoch] = batch_done
        train_accuracy_log[epoch] = train_accuracy

        print('\n Epoch {}, after {} total batches, accum loss is {}, validation accuracy is {} \n'.format(
            epoch + 1, batch_done, train_loss, vali_accuracy))
        print('------------- Training accuracy at end of epoch is {} -------------\n'.format(train_accuracy))
    torch.save(model,'model_{}{}.pt'.format(model_type,'WSDdetection'))


    test_loss,test_accuracy,cm = evaluate_model_test(model,loss_fxn,test_iter)
    print("================ RESULTS FOR MODEL: {} ================\n".format(model_type))
    # Use the results from the last epoch as loss and accuracy for training set:
    print("TRAIN SET: loss = {}, accuracy = {}".format(train_loss,train_accuracy))
    # Use the results from the last epoch as loss and accuracy for validation set:
    print("VALIDATION SET: loss = {}, accuracy = {}".format(vali_loss,vali_accuracy))
    print("TEST SET: loss = {}, accuracy = {}".format(test_loss, test_accuracy))


    plot_confusion_matrix('{}'.format(model_type), cm, ['pun','not pun'])

    # generate csv file and plot accuracy
    if save_csv == True:
        curr_time = datetime.datetime.now()
        df = pd.DataFrame({'steps': batch_number_log.squeeze(), "train_accuracy": train_accuracy_log.squeeze(),
                           'validation_accuracy': vali_accuracy_log.squeeze()})
        # use current date_time to label model name
        file_name = './plots/{}_{}.csv'.format(model_type,curr_time.strftime("%b%d_%H_%M_%S"))
        df.to_csv(file_name, index=False)
        plot_accuracy(file_name, test_accuracy)


'''
PLOT TRAINING AND VALIDATION ACCURACY
'''
def plot_accuracy(file,test_acc):
    csv_name1 = file
    acc_data1 = pd.read_csv(csv_name1)
    steps = acc_data1["steps"]
    train_acc1 = acc_data1['train_accuracy']
    val_acc1 = acc_data1['validation_accuracy']
    plt.figure()
    plt.suptitle("Validation & training accuracy vs. steps",fontweight='bold')
    plt.title('(test accuracy {})'.format(test_acc),fontsize=9)
    plt.plot(steps, val_acc1,steps, train_acc1)
    plt.xlabel("Number of steps")
    plt.ylabel('Accuracy')
    plt.legend(["Validation accuracy", "Training accuracy"], loc='best')
    plt.savefig("{}_plot.png".format(file[:-4]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batch-size', type=int, default=41)
    parser.add_argument('--eval-batch-size', type=int, default=29)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rnn_sg',
                        help="Model type: baseline,rnn,cnn, crnn, birnn, rnn_sg (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=10)
    parser.add_argument('--rnn-hidden-dim', type=int, default=10)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--save-csv', type=bool, default=True)

    args = parser.parse_args()

    main(args)



'''
SCORE = data.Field(sequential=True, use_vocab=False,include_lengths=True)
LOCATION_LABEL = data.Field(sequential=False, use_vocab=False)
fields = [('score', SCORE), ('location', LOCATION_LABEL)]
train, val, test = data.TabularDataset.splits(path = '',train='train1location_sensegram.tsv',
                                              validation='validation1location_sensegram.tsv',
                                              test='test1location_sensegram.tsv',
                                              format='tsv', fields=fields, skip_header=True)

train_iter= torchtext.data.Iterator(sort_key=lambda x: len(x.score), sort_within_batch=True, repeat=False,
                                    dataset=train, batch_size =bs, train=True, shuffle=True)
val_iter = torchtext.data.BucketIterator(sort_key=lambda x: len(x.score), sort_within_batch=True, repeat=False,
                                     dataset=val, batch_size=bs)
test_iter = torchtext.data.BucketIterator(sort_key=lambda x: len(x.score), sort_within_batch=True, repeat=False,
                                     dataset=test, batch_size=bs)

for data in train_iter:
    (x, x_lengths), y = data.score, data.location
    print('asdf')
'''
