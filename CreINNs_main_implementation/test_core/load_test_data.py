from keras import datasets
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from keras.utils import to_categorical


def load_svhn_test():

    test_data = sio.loadmat('/staging/leuven/stg_00155/ood_datasets/SVHN/test_32x32.mat')
    x_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    y_test = test_data['y']
    # SVHN labels are 1-indexed, convert 10 to 0
    y_test[y_test == 10] = 0
    x_test = x_test / 255.0
    x_test = x_test.astype('float32')
    y_test = to_categorical(y_test, 10)
    
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    
    # val_samples = -100
    
    # x_test = x_test[val_samples:]
    # y_test = y_test[val_samples:]
    return x_test, y_test


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # Standardizing
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = (x_train - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    # val_samples = -100
    
    # x_test = x_test[val_samples:]
    # y_test = y_test[val_samples:]

    return (x_train, y_train), (x_test, y_test)
