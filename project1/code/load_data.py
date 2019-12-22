import os.path
import glob
import numpy as np
from cv2 import cv2


def load_data(path):
    '''
    Load image and label , at the same time resize image to (3,60,30)
    '''
    empty_list = os.listdir(path+'empty\\')
    occupied_list = os.listdir(path+'occupied\\')
    image = np.zeros((len(empty_list)+len(occupied_list), 3, 60, 30))
    label = np.zeros(len(empty_list)+len(occupied_list))
    i = 0

    for figure_path in empty_list:
        img = cv2.imread(path+'empty\\'+figure_path)
        # img = cv2.resize(img, (30, 15))
        img = np.resize(img, (3, 60, 30))
        image[i] = img/255.

        label[i] = 0
        i += 1

    for figure_path in occupied_list:
        img = cv2.imread(path+'occupied\\'+figure_path)
        # img = cv2.resize(img, (30, 15))
        img = np.resize(img, (3, 60, 30))
        image[i] = img/255.

        label[i] = 1
        i += 1

    return image, label


def load_train_data():
    '''
    train data:
    96 empty label
    285 occupied label
    '''
    return load_data('project1\\data\\train\\')


def load_test_data():
    '''
    test data:
    38 empty label
    126 ocupied label
    '''
    return load_data('project1\\data\\test\\')


load_test_data()
