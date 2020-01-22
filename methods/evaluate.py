from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_classification(y_test, y_pred, labels, method):
    """

    A method to evaluate the classifier

    Args:
        y_test (numpy.ndarray): The array for y_test
        y_pred (numpy.ndarray): The array of y predictions from the classifier
        method (str): The name of the classification method being evaluated
        labels(numpy.ndarray): class labels
    Returns:
        cm(numpy.ndarray): the confusion matrix values
        roc_auc(int): the ROC value indicating classifier performance
    """
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    _plot_roc(fpr, tpr, roc_auc, method)
    plt.figure()
    _plot_cm(cm, method, labels)
    return cm, roc_auc


def _plot_roc(fpr, tpr, roc_auc, classification_method):
    """

    A method to plot the ROC Curve

    Args:
        fpr (numpy.ndarray): An array containing the false positive rate
        tpr (numpy.ndarray): An array containing the true positive rate
        roc_auc(int): the ROC value indicating classifier performance
        method (str): The name of the classification method being evaluated

    Returns:
        None
    """
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.suptitle("Classification Method : {}".format(classification_method))
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC CURVE")
    plt.show()


def _plot_cm(cm, classification_method, labels=None):
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title("Confusion Matrix: Classification Method : {}".format(
        classification_method))
    if labels is not None:
        ax.xaxis.set_ticklabels([labels[0], labels[1]])
        ax.yaxis.set_ticklabels([labels[1], labels[0]])
    else:
        ax.xaxis.set_ticklabels(['Class 0', 'Class 1'])
        ax.yaxis.set_ticklabels(['Class 0', 'Class 1'])
