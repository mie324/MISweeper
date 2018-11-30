import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


def get_best_accuracy():
    with open("best.txt", 'r') as f:
        return float(f.read())


def save_best_accuracy(best_accuracy):
    with open("best.txt", 'w+') as f:
        f.write(str(best_accuracy))


def save_predictions(all_preds, object_ids):
    print("Saving predictions")

    sample_sub = pd.read_csv('Data/sample_submission.csv')
    submission_columns = list(sample_sub.columns)
    del sample_sub

    class_99 = np.ones(all_preds.shape[0])
    for i in range(all_preds.shape[1]):
        class_99 *= (1 - all_preds[:, i])

    final_predictions = pd.DataFrame(all_preds)
    final_predictions.insert(loc=0, column='object_id', value=object_ids)
    final_predictions.insert(loc=len(final_predictions.columns), column='class_99', value=class_99)
    final_predictions.columns = submission_columns

    final_predictions.to_csv('predictions.csv', header=True, mode='w', index=False)


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
