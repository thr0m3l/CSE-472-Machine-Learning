import pandas as pd
import numpy as np

def normalize(df):
    #Perform min-max normalization on the data for all columns except the last column
    result = df.copy()
    for feature in df.columns:
        max_value = df[feature].max()
        min_value = df[feature].min()
        # print(feature, max_value, min_value)
        result[feature] = (df[feature] - min_value) / (max_value - min_value)
    return result

def load_dataset(filename):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    #Read the data from the csv file
    data = pd.read_csv(filename)
    
    data = normalize(data)
    

    zeros = data[data.iloc[:, -1] == 0]
    ones = data[data.iloc[:, -1] == 1]

    #Random undersampling
    zeros = zeros.sample(n=ones.shape[0], random_state=69)
    data = pd.concat([zeros, ones])
    
    #Output the data to temp csv file
    pd.DataFrame(data).to_csv("temp.csv", index=True, header=True)
    
    print(data.head())
    
    #Return a 2D feature matrix and a vector of class
    return data.iloc[:, :-1].values, data.iloc[:, -1].values

def shuffle_dataset(X, y):
    """
    function for shuffling the dataset
    :param X:
    :param y:
    """
    #shuffle the dataset
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]
    
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    #split the dataset into train and test, if shuffle is true, shuffle the dataset first
    
    #shuffle the dataset
    if shuffle:
        X, y = shuffle_dataset(X, y)
        
    X_train = X[:int(len(X) * (1 - test_size))]
    y_train = y[:int(len(y) * (1 - test_size))]
    X_test = X[int(len(X) * (1 - test_size)):]
    y_test = y[int(len(y) * (1 - test_size)):]
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # print(int(datetime.now().timestamp()))
    #Get a random seed
    randomize = np.arange(len(X))
    random_choice = np.random.choice(a=randomize, size=len(X), replace=True)
    print(random_choice)
    X_sample = X[random_choice]
    y_sample = y[random_choice]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
