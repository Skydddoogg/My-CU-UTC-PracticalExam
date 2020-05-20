import sys
sys.path.append("../")

import numpy as np
import os
from Problem3.config_path import data_path, result_path
from sklearn.model_selection import train_test_split

def get_splitted_data(validation = False):

    X_train = np.load(os.path.join(data_path, 'train', 'X.npy'))
    X_test = np.load(os.path.join(data_path, 'test', 'X.npy'))
    y_train = np.load(os.path.join(data_path, 'train', 'y.npy'))
    y_test = np.load(os.path.join(data_path, 'test', 'y.npy'))

    if validation:
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.20, random_state=0, stratify=y_train)
        
        print("Training samples: {0}".format(y_train.shape[0]))
        print("Test samples: {0}".format(y_test.shape[0]))
        print("Validation samples: {0}".format(y_valid.shape[0]))
        
        return X_train, X_test, X_valid, y_train, y_test, y_valid

    print("Training samples: {0}".format(y_train.shape[0]))
    print("Test samples: {0}".format(y_test.shape[0]))

    return X_train, X_test, y_train, y_test

def save_results(y_true, y_pred, prob, specified_name):

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    if not os.path.isdir(os.path.join(result_path, 'groundtruth')):
        os.mkdir(result_path + '/groundtruth')
        os.mkdir(result_path + '/prediction')
        os.mkdir(result_path + '/probability')

    np.save(os.path.join(result_path, 'groundtruth', specified_name + ".npy"), np.array(y_true))
    np.save(os.path.join(result_path, 'prediction', specified_name + ".npy"), np.array(y_pred))
    np.save(os.path.join(result_path, 'probability', specified_name + ".npy"), np.array(prob))