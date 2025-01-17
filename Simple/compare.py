import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

labels = pd.read_csv('Data/training_set_metadata.csv')['target']

predictions = pd.read_csv('predictions.csv').set_index('object_id')
pred_argmax = predictions.values[:, :-1].argmax(axis=1)

class_map = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95])
pred_good = class_map[pred_argmax]

cm = confusion_matrix(labels, pred_good)

# Compare with labels
def plot_confusion_matrix(path, cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        path: String to the output folderin which to save the cm image
        cm: Numpy array of confusion matrix
        classes: List of strings of names of the classes
        normalize: Whether or not to normalize the cm
        title: Does what you think
        cmap: color map to use
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = [str(i) for i in [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)


plot_confusion_matrix('train_cm.png', cm, normalize=True)
