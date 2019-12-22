import predict
from cv2 import cv2
import numpy as np


def monitor(image):
    '''
    Return the image with the num of parking space in the
    give image

    Paramaters
    ----------
    image:
            the frame of a video waited to be monitored
    '''

    num = predict.predict(image)
    cv2.putText(image, 'parking space:{}'.format(num), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 255))

    return image


cap = cv2.VideoCapture('project1\\parking_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img = monitor(frame)
        cv2.imshow('parking space monitor', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
