import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

from util import *


def evaluate_confusion_matrix(path, model, val_loader, target_classes):
    """
    Run the model on the test set and generate the confusion matrix.

    Args:
        path: String to the output path of confusion matrix plot
        model: PyTorch neural network object
        val_loader: PyTorch data loader for the dataset
        target_classes: A list of strings denoting the name of the desired classes.
                        Should be a subset of the 'classes'
    Returns:
        cm: A NumPy array denoting the confusion matrix
    """
    val_labels = np.array([], dtype=np.int64)
    val_preds = np.array([], dtype=np.int64)

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = model(inputs)
        output_probs = F.sigmoid(outputs).detach().numpy()
        preds = (output_probs > 0.5)
        val_labels = np.concatenate((val_labels, labels))
        val_preds = np.concatenate((val_preds, preds))

    cm = confusion_matrix(val_labels, val_preds)
    plot_confusion_matrix(path, cm, target_classes)

    return cm


# Function based off
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(path, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        path: String to the output path of confusion matrix plot
        cm: A NumPy array denoting the confusion matrix
        classes: A list of strings denoting the name of the classes
        normalize: Boolean whether to normalize the confusion matrix or not
        title: String for the title of the plot
        cmap: Colour map for the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix_{}.png".format(path))

    return


def evaluate_visual_confusion_matrix(path, model, val_loader, target_classes):
    """
    Run the model on the test set and generates a 'visual' confusion matrix.
    We obtain the images with the highest confidence for each scenario.


    Args:
        path: String to the output path of confusion matrix plot
        model: PyTorch neural network object
        val_loader: PyTorch data loader for the dataset
        target_classes: A list of strings denoting the name of the desired classes.
                        Should be a subset of the 'classes'
    """
    current_cm = 0.5 * np.ones((2,2)) # Initialize it to the 'middle' value 0.5
    images = [[0,0],[0,0]]

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = model(inputs)
        output_probs = F.sigmoid(outputs).detach().numpy()
        preds = (output_probs > 0.5).astype(int)

        for j in range(len(labels)):
            if preds[j] == 0:
                if output_probs[j] < current_cm[labels[j]][preds[j]]:
                    current_cm[labels[j]][preds[j]] = output_probs[j]
                    images[labels[j]][preds[j]] = inputs[j]
            else:
                if output_probs[j] > current_cm[labels[j]][preds[j]]:
                    current_cm[labels[j]][preds[j]] = output_probs[j]
                    images[labels[j]][preds[j]] = inputs[j]

    fig = plt.figure()
    plt.suptitle("Visual Confusion Matrix (Highest Confidence)")
    plt.axis('off')
    col = 2
    row = 2
    for i in range(1, col*row + 1):
        r = math.floor((i-1)/row)
        c = (i-1) % col
        img = images[r][c].permute(1,2,0)
        # De-normalize the image from [-1,1] to [0,1]
        img = (img + 1) / 2
        fig.add_subplot(row, col, i)
        plt.title("Prob={0:.3f}".format(current_cm[r][c]))
        plt.ylabel("Label: {}".format(target_classes[r]))
        plt.xlabel("Pred: {}".format(target_classes[c]))
        plt.imshow(img)

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.savefig("visual_confusion_matrix_{}.png".format(path))

    return


def main():
    ########################################################################
    # Loads the configuration for the experiment from the configuration file
    config, learning_rate, batch_size, num_epochs, target_classes = load_config('configuration.json')

    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader, classes = get_data_loader(target_classes, batch_size)

    # Load the model
    net = Net()
    model_path = get_model_name(config)
    net.load_state_dict(torch.load(model_path))

    # Evaluate the model for the confusion matrix
    evaluate_confusion_matrix(model_path, net, test_loader, target_classes)

    # Visualize examples for false positive / true negative
    evaluate_visual_confusion_matrix(model_path, net, test_loader, target_classes)


if __name__ == '__main__':
    main()