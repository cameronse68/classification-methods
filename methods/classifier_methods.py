import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from methods.evaluate import evaluate_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import logging

from methods.classification_base import BaseClassifier

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SVM(BaseClassifier):

    def __init__(
          self,
          df, class_column, split_level=0.2, labels=None, kernel='rbf', class_weight='balanced'
    ):
        """
        A Support Vector Machine Classifier
        Args:
            kernel (str): The Kernel to use in the classifier
            class_weight (str): Class weight: options " uniform" or "balanced"

        Returns:
            y_pred (numpy.ndarray): the classification predictions
            cm(numpy.ndarray): the confusion matrix values
            roc_auc(in): the ROC value indicating classifier performance
        """
        self.kernel = kernel
        self.class_weight=class_weight

        super(SVM, self).__init__(
            df = df,
	        class_column = class_column,
	        split_level = split_level,

        )

    # def KNN(self, neighbors=5):
    #     """
    #     A K Nearest Neighbors Classifier
    #     Args:
    #         neighbors (int): The number of neighbors to use
    #     Returns:
    #         y_pred (numpy.ndarray): the classification predictions
    #         cm(numpy.ndarray): the confusion matrix values
    #         roc_auc(in): the ROC value indicating classifier performance
    #     """
    #     model = KNeighborsClassifier(n_neighbors=neighbors)
    #     model.fit(self.x_train, self.y_train)
    #     y_pred_knn = model.predict(self.x_test)
    #     cm, roc_auc = evaluate_classification(
    #         y_pred_knn, self.y_test, self.labels, method='KNN')
    #     return y_pred_knn, cm, roc_auc

    # def GaussNB(self):
    #     """
    #     A NaiveBayes Classifier
    #     Args:
    #          None
    #     Returns:
    #         y_pred (numpy.ndarray): the classification predictions
    #         cm(numpy.ndarray): the confusion matrix values
    #         roc_auc(in): the ROC value indicating classifier performance
    #     """
    #     clf = GaussianNB()
    #     clf.fit(self.x_train, self.y_train)
    #     y_pred = clf.predict(self.x_test)
    #     cm, roc_auc = evaluate_classification(
    #         y_pred, self.y_test, self.labels, method='Naive Bayes')
    #     return y_pred, cm, roc_auc
