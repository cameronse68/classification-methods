import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from methods.evaluate import evaluate_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Classification(object):

    def __init__(
            self,
            df,
            class_column,
            split_level=0.2,
            labels=None
    ):
        """
        Args:
            df(pandas.core.DataFrame): queried DataFrame
            class_column (str): the name of the column denoting the class
            split_level (int): the percent to split the data into train/test.
                               default is 80% train 20% test
            labels(numpy.ndarray): class labels
        """
        self.df = df
        self.class_column = class_column
        self.split_level = split_level

        try:
            self.x = self.df.drop(self.class_column, axis=1)
            self.y = self.df[self.class_column]
        except (KeyError, Exception) as e:
            logger.error("{}:".format(type(e)), exc_info=True)
            logger.error("Column Name not in DataFrame")
            raise

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.split_level)
        self.labels = labels

    @property
    def svm(self, kernel='rbf', class_weight='balanced'):
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
        svclassifier = SVC(kernel='rbf', class_weight='balanced')
        svclassifier.fit(self.x_train, self.y_train)
        y_pred_svc = svclassifier.predict(self.x_test)
        cm, roc_auc = evaluate_classification(
            y_pred_svc, self.y_test, self.labels, method='SVC')
        return y_pred_svc, cm, roc_auc

    def KNN(self, neighbors=5):
        """
        A K Nearest Neighbors Classifier
        Args:
            neighbors (int): The number of neighbors to use
        Returns:
            y_pred (numpy.ndarray): the classification predictions
            cm(numpy.ndarray): the confusion matrix values
            roc_auc(in): the ROC value indicating classifier performance
        """
        model = KNeighborsClassifier(n_neighbors=neighbors)
        model.fit(self.x_train, self.y_train)
        y_pred_knn = model.predict(self.x_test)
        cm, roc_auc = evaluate_classification(
            y_pred_knn, self.y_test, self.labels, method='KNN')
        return y_pred_knn, cm, roc_auc

    def GaussNB(self):
        """
        A NaiveBayes Classifier
        Args:
             None
        Returns:
            y_pred (numpy.ndarray): the classification predictions
            cm(numpy.ndarray): the confusion matrix values
            roc_auc(in): the ROC value indicating classifier performance
        """
        clf = GaussianNB()
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        cm, roc_auc = evaluate_classification(
            y_pred, self.y_test, self.labels, method='Naive Bayes')
        return y_pred, cm, roc_auc
