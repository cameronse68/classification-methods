
import logging 
from sklearn.model_selection import train_test_split
	
	
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
	
class BaseClassifier(object):
	
	def __init__(self,df, class_column, split_level=0.2, labels=None):
	
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
	    
	def fit(self,model):
	        model.fit(self.x_train, self.y_train)
	        predictions = model.predict(self.x_test)
	        self.predictions=predictions
	        return self, predictions 
