from data_handler import bagging_sampler
import copy
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        # self.estimators = [copy.deepcopy(base_estimator) for _ in range(n_estimator)]
        self.estimators = []
        for _ in range(n_estimator):
            self.estimators.append(copy.deepcopy(base_estimator))

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        #Use each base estimator to fit a model on a different bootstrap sample
        for estimator in self.estimators:
            random_X, random_y = bagging_sampler(X, y)
            # print(random_X, random_y)
            estimator.fit(random_X, random_y)

            

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        
        p = np.zeros(X.shape[0])
        
        #Take the majority vote of all the base estimators
        for estimator in self.estimators:
            p += estimator.predict(X)
        
        p = np.round(p/self.n_estimator)
    
        return p