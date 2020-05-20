import os
import pickle
import sys
sys.path.append("../")

from Problem3.config_path import result_path

def save_history(history, specified_name):

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    if not os.path.isdir(os.path.join(result_path, 'history')):
        os.mkdir(os.path.join(result_path, 'history'))

    history.history['epoch'] = history.epoch
    pickle_history = open(os.path.join(result_path, 'history', specified_name + ".pickle"), "wb")
    pickle.dump(history.history, pickle_history)
    pickle_history.close()
