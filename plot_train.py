import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
PLOT TRAINING AND VALIDATION ACCURACY FOR RNN AND CNN, USING THE OUTPUT CSV SAVED EARLIER IN MAIN()
'''
def plot_accuracy(file1,file2):
    csv_name1 = file1 # rnn
    csv_name2 = file2 # cnn
    acc_data1 = pd.read_csv(csv_name1)
    acc_data2 = pd.read_csv(csv_name2)
    steps = acc_data1["steps"]
    train_acc1 = acc_data1['train_accuracy']
    train_acc2 = acc_data2['train_accuracy']
    val_acc1 = acc_data1['validation_accuracy']
    val_acc2 = acc_data2['validation_accuracy']
    plt.figure()
    plt.suptitle("Validation & training accuracy vs. steps",fontweight='bold')
    plt.title('(RNN test accuracy {} ; CNN test accuracy {})'.format(0.7356,0.7395),fontsize=9)
    plt.plot(steps, val_acc1,steps, train_acc1,steps,val_acc2,steps,train_acc2)
    plt.xlabel("Number of steps")
    plt.ylabel('Accuracy')
    plt.legend(["RNN validation accuracy", "RNN training accuracy","CNN validation accuracy", "CNN training accuracy"], loc='best')
    plt.savefig("./plots/{}_orig.png".format('combined'))

def main():
    plot_accuracy('./plots/rnn_1_0.735632183908046.csv','./plots/cnn_1_0.7394636015325671.csv')


if __name__ == '__main__':
    main()