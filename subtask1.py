import torch
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import argparse
import os
import numpy as np
from subtask1_model import *

torch.manual_seed(77)
bs,lr,MaxEpochs,model_type,num_filters,rnn_hidden_dim,embed_dim = None,None,None,None,None,None,None


bs = 64
def build_iter():
    global bs
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    DETECTION_LABEL = data.Field(sequential=False, use_vocab=False)
    fields = [('text', TEXT), ('detection', DETECTION_LABEL)]
    train, val, test = data.TabularDataset.splits(path = '',train='train1.tsv', validation='validation1.tsv',
                                                  test='test1.tsv', format='tsv', fields=fields, skip_header=True)

    train_iter= torchtext.data.BucketIterator(sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False,
                                    dataset=train, batch_size =64, train=True, shuffle=True)
    val_iter = torchtext.data.BucketIterator(sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False,
                                     dataset=val, batch_size=64)
    test_iter = torchtext.data.BucketIterator(sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False,
                                     dataset=test, batch_size=64)


    TEXT.build_vocab(train,val,test)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=300))
    return train_iter, val_iter, test_iter, TEXT.vocab



'''
LOAD MODEL
'''
def load_model(learning_rate,vocab):
    global model_type,rnn_hidden_dim,embed_dim
    if model_type == 'baseline':
        model = Baseline(embed_dim,vocab) # will use TEXT.vocab
    elif model_type == 'rnn':
        model = RNN(embed_dim,vocab,rnn_hidden_dim)
    elif model_type == 'cnn':
        model = CNN(embed_dim, vocab, num_filters, np.array([2,4]))
    elif model_type == 'crnn':
        model = CRNN(embed_dim, vocab, num_filters, np.array([2,4]))
    elif model_type == 'birnn':
        model = biRNN(embed_dim,vocab,rnn_hidden_dim)
    loss_fxn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model,loss_fxn,optimizer



'''
EVALUATE RESULTS WITH VALIDATION SET
'''
def evaluate_model(model,loss_fxn,val_iter):
    global bs
    total_val_corr = 0
    accum_loss = 0
    vali_samples = len(val_iter.dataset)
    model.eval()
    for i,data in enumerate(val_iter):
        (x, x_lengths), y = data.text, data.detection
        predictions = model.forward(x,x_lengths)
        loss = loss_fxn(input=predictions.squeeze(), target=y.float())
        accum_loss += loss.item()

        # compare batch predictions to labels
        pred_res = predictions.data.squeeze().numpy()
        actual_res = y.int().numpy()
        for j in range(len(y)):
            if pred_res[j] <= 0.5 and actual_res[j] == 0:
                total_val_corr += 1
            elif pred_res[j] > 0.5 and actual_res[j] == 1:
                total_val_corr += 1

    eval_accuracy = float(total_val_corr)/vali_samples
    return accum_loss/(i+1),eval_accuracy  # return this as the evaluation accuracy



'''
TRAINING AND EVALUATION
'''
def main(args):
    dict_args = vars(args)
    global bs, lr, MaxEpochs, model_type,eval_every,num_filters,rnn_hidden_dim,embed_dim
    bs = dict_args['batch_size']
    lr = dict_args['lr']
    MaxEpochs = dict_args['epochs']
    model_type = dict_args['model']
    num_filters = dict_args['num_filt']
    rnn_hidden_dim = dict_args['rnn_hidden_dim']
    embed_dim = dict_args['emb_dim']
    eval_every = int(1640 / bs)

    train_iter, val_iter, test_iter,vocab = build_iter()
    model,loss_fxn,optimizer = load_model(lr,vocab)
    batch_done = 0
    #max_vali_accuracy = 0
    #lowest_vali_loss = 0

    for epoch in range(MaxEpochs):
        accum_loss = float(0)
        total_correct = 0


        for i,data in enumerate(train_iter):
            (x, x_lengths), y = data.text, data.detection

            optimizer.zero_grad()
            predictions = model.forward(x,x_lengths)

            loss = loss_fxn(input=predictions.squeeze(),target=y.float())
            accum_loss += loss.item()

            loss.backward()
            optimizer.step()

            # comopare batch predictions to labels
            pred_res = predictions.data.squeeze().numpy()
            actual_res = y.int().numpy()
            for j in range(len(y)):
                if pred_res[j] <= 0.5 and actual_res [j] ==0:
                    total_correct +=1
                elif pred_res[j] > 0.5 and actual_res [j] ==1:
                    total_correct +=1

            if batch_done == 0:
                print('\n ---------------- T R A I N I N G   I N   P R O G R E S S ----------------\n')
            #if (batch_done + 1) % eval_every == 0:
                #vali_loss,vali_accuracy = evaluate_model(model, loss_fxn,val_iter)
                #vali_accuracy_log[epoch] = vali_accuracy
                #if vali_accuracy > max_vali_accuracy:
                    #max_vali_accuracy = vali_accuracy
                    #lowest_vali_loss = vali_loss
                #print('\n Epoch {}, after {} total batches, accum loss is {}, validation accuracy is {} \n'.format(
                    #epoch + 1, batch_done + 1, accum_loss / 100, vali_accuracy))
                #batch_number_log[epoch] = batch_done + 1
            batch_done += 1

        vali_loss, vali_accuracy = evaluate_model(model, loss_fxn, val_iter)
        train_loss = accum_loss/(i+1)
        train_accuracy = total_correct / len(train_iter.dataset)
        #train_accuracy_log[epoch] = train_accuracy
        print('\n Epoch {}, after {} total batches, accum loss is {}, validation accuracy is {} \n'.format(
            epoch + 1, batch_done, train_loss, vali_accuracy))
        print('------------- Training accuracy at end of epoch is {} -------------\n'.format(train_accuracy))


    torch.save(model,'model_{}.pt'.format(model_type))
    test_loss,test_accuracy = evaluate_model(model,loss_fxn,test_iter)
    print("================ RESULTS FOR MODEL: {} ================\n".format(model_type))
    # Use the results from the last epoch as loss and accuracy for training set:
    print("TRAIN SET: loss = {}, accuracy = {}".format(train_loss,train_accuracy))
    # Use the results from the last epoch as loss and accuracy for validation set:
    print("VALIDATION SET: loss = {}, accuracy = {}".format(vali_loss,vali_accuracy))
    print("TEST SET: loss = {}, accuracy = {}".format(test_loss, test_accuracy))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=82)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='birnn',
                        help="Model type: baseline,rnn,cnn, crnn, birnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=300)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)


