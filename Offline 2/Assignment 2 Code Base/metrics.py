"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return np.mean(y_true == y_pred) * 100
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    precision_score = np.sum(y_true * y_pred) / np.sum(y_pred)
    return precision_score

def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    recall_score = np.sum(y_true * y_pred) / np.sum(y_true)
    return recall_score


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    #Calculate the F1 score
    f1_score = 2 * (precision_score(y_true, y_pred) * recall_score(y_true, y_pred)) / (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
    return f1_score
