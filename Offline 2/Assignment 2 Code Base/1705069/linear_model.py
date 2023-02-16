import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        learning_rate: learning rate for gradient descent
        lambda_: regularization parameter
        max_iter: maximum number of iterations
        w: weights
        b: bias
        threshold: threshold for classification
        """
        self.learning_rate = params['learning_rate']
        self.lambda_ = params['lambda_']
        self.max_iter = params['max_iter']
        self.w = np.array([])
        self.b = 1
        self.threshold = 0.5
        
        print("Logistic Regression initialized with learning rate: {}, lambda: {}, max_iter: {}".format(self.learning_rate, self.lambda_, self.max_iter))
        

    def sigmoid(self, z):
        """
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))
    
    def loss(self, X, y, w, b):
        """
        Calculate the loss function for logistic regression
        """
        m = X.shape[0]
        #Calculate the regularization term
        reg = (self.lambda_ / (2 * m)) * np.sum(w ** 2)
        return -np.sum(y * np.log(self.sigmoid(X.dot(w) + b)) +
                       (1 - y) * np.log(1 - self.sigmoid(X.dot(w) + b))) + reg

    
    def gradient(self, X, y, w, b):
        """
        Calculate the gradient of the loss function
        """
        m = X.shape[0]
        f_w = self.sigmoid(X.dot(w) + b)
        err = f_w - y
        dw = 1 / m * X.T.dot(err) + (self.lambda_ / m) * w
        db = 1 / m * np.sum(err)
        
        return db, dw
        
    def gradient_descent(self, X, y, w, b):
        """
        Perform gradient descent
        """
        #Number of training examples
        m = len(X)
        
        for i in range(self.max_iter):
            db, dw = self.gradient(X, y, w, b)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
        
        return w, b
        
    
    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        
        """
        Find the optimal weight vector w using gradient descent
        """
        
        initial_w = np.random.randn(X.shape[1])
        initial_b = -8
        
        self.w, self.b = self.gradient_descent(X, y, initial_w, initial_b)
        
        print("Logistic Regression model trained with w: {}, b: {}".format(self.w, self.b))
        
        
        
        

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        m = X.shape[0]
        p = np.zeros(m)
        
        for i in range(m):
            f_w = self.sigmoid(X[i].dot(self.w) + self.b)
            p[i] = f_w >= self.threshold
        
        return p
