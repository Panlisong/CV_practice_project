from cv2 import cv2
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    '''
    Get the coordinate of the position which is clicked
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        # with open('project1\\park_coordinate.txt') as f:
        #     f.write(xy)
        img = cv2.imread("project1\\park_coordinate.jpg")
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


def get_coordinate():
    '''
    Through mouse clicking event to get the coordinate of the point
    '''
    img = cv2.imread("project1\\park_coordinate.jpg")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)

    while(True):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break

    cv2.waitKey(0)
    cv2.destroyAllWindow()


def crop(img):
    '''
    Return the num of parking space and the whole parking space coordinate
    in the img and the array's shape is (praking_sapce_num,30,60,3)

    Parameters
    ----------
    img: 
        the input image waited to be cropped
    '''
    x = np.zeros(44)
    y = np.zeros(44)
    i = 0

    with open('project1\\park_coordinate.txt') as f:
        for line in f:
            tmpx, tmpy = line.split(',')
            tmpy = tmpy.split('\n')[0]
            x[i] = int(tmpx)
            y[i] = int(tmpy)
            i += 1

    parking_space_num = 0
    for i in range(22):
        parking_space_num += int((y[2*i+1]-y[2*i])/15)

    parking_space_coordinate_x = np.zeros(parking_space_num)
    parking_space_coordinate_y = np.zeros(parking_space_num)
    image = np.zeros((parking_space_num, 30, 60, 3))
    a = np.zeros((15, 30, 3))
    k = 0
    for i in range(22):
        whole_gap = y[2*i+1]-y[2*i]
        gap = int(whole_gap/(int(whole_gap/15)))
        for j in range(int(whole_gap/15)):
            tmp = img[int(y[2*i]+gap*j):int(y[2*i]+gap*j+gap),
                      int(x[2*i]):int(x[2*i+1])]

            image[k] = cv2.resize(tmp, (60, 30))/255
            k += 1

    return parking_space_num, image
