import net
import torch
import data_preprocessing
import numpy as np
from cv2 import cv2


def predict(image):
    '''
    Return the num of the parking space in the image

    Paramaters
    ----------
    image:
            the image wait to be recognized
    '''
    parking_sapce_num, parking_space = data_preprocessing.crop(image)

    parking_space = np.reshape(parking_space, (parking_sapce_num, 3, 60, 30))
    parking_space = torch.Tensor(parking_space)

    car_detection = net.car_detection_net()
    car_detection.load_state_dict(torch.load('project1\\paramaters.pth'))

    res = car_detection(parking_space)
    res = np.argmax(res.data.numpy(), axis=1)
    return np.sum(res)
