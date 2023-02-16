import os
import glob
import pickle
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from train_1705069 import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    data_dir = os.path.join('./NumtaDB_with_aug', '')
    test_X = sorted(glob.glob(os.path.join(data_dir, 'test-b1', '*.png')))
    file_names = [os.path.basename(x) for x in test_X]

    #  print(test_X)
    # test_label = os.path.join(data_dir, 'training-d.csv')

    # Load all the images in the directory
    X_test_all = get_data(test_X,  resize_dim=28)
    X_test_all = X_test_all.reshape(X_test_all.shape[0], 28, 28, 1)

    print(X_test_all.shape)

    # Load the model from the pickle file
    cnn = pickle.load(open('cnn_model_1.pkl', 'rb'))

    # Predict the labels
    y_pred = cnn.predict(X_test_all)

    # Print the predicted labels from one hot encoding
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    # Now, calculate the accuracy of the model
    # acc = np.sum(y_pred == y_test_all)/y_test_all.shape[0]
    # print('Accuracy = ' + str(acc))

    # # Calculate the macro F1 score
    # f1 = f1_score(y_test_all, y_pred, average='macro')
    # print('Macro F1 Score = ', str(f1))

    # # Print the confusion matrix
    # conf_mat = confusion_matrix(y_test_all, y_pred)
    # print('Confusion Matrix = \n ', str(conf_mat))

    # Generate a csv file with the predicted labels along with the image names from test_X
    csv_file = pd.DataFrame({'FileName': file_names, 'Digit': y_pred})
    csv_file.to_csv('1705069_prediction.csv', index=False)
