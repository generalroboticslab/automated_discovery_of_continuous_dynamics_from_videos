'''
This script provides utility functions estimating the angle of a single pendulum from image.
Step 1. Extract the mass from the image.
Step 2. Do a rectangle fitting of the mass.
Step 3. Estimate the position of the mass. 
Certain images will be rejected if the mass does not exist or has a wrong shape.
'''

import cv2
import os
import numpy as np


'''
Extract the mass from the image.
Args:
    img: spring mass image in BGR format
Returns:
    seg: segmentation of the mass
'''
def seg_from_img(img):

    # pixel thresholds (in HSV)
    v_min = (0, 0, 0)
    v_max = (140, 140, 140)

    # extract the mass 
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    seg = cv2.inRange(img_hsv, v_min, v_max)


    return seg



'''
Fit mass in a rectangle.
Args:
    seg: segmentation of the mass
Returns:
    rej: (True/False) if the image is rejected
    rect: (Box2D structure) the fitted rectangle
'''
def fit_mass(seg):
    # find all contours
    contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # reject if no contours found
    if len(contours) == 0:
        return True, None
    # find the contour with the maximum area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    # reject if the contour is too small
    if area < 900:
        return True, None
    # rectangle fitting
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (width, height), _ = rect
    # reject if the rectangle is not close to a square
    if abs(height-width) > 5:
        return True, None
    # reject if the rectangle does not properly fit the contour
    if height*width > 1.5 * area:
        return True, None
    # reject if the rectangle's vertical placement is not centered
    if abs(cy- 64) > 5:
        return True, None
    
    return False, rect



'''
Estimate the position of mass from its fitted rectangle.
Args:
    rect: (Box2D structure) the fitted rectangle
Returns:
    angle: the estimated position in meters in range (-1,1)
    box: box points of the rectangle
    arrow: the arrow pointing along the displacement
'''
def estimate_position(rect):
    # box points
    box = cv2.boxPoints(rect)

    #print(box)
    # center, width and height
    (cx, cy), (width, height), angle = rect
    
    #get position
    position = cx

    #spring rest position 350 x
    rest = 63.5

    #get distance from rest position
    offset = position - rest

    #Scale from 128x128 to 800x800 to 1meter/250pxl
    # 800 / 128 * 1/ 250
    x = .025 * offset

    arrow = ((int(rest),64), (int(position), int(64)))
    return x, box, arrow


'''
Obtain the position of spring mass from image
Args:
    img: spring mass image in BGR format
Returns:
    rej: (True/False) if the image is rejected
    x: the estimated position in meters in range (-1,1)
    img_marked: image marked with the fitted rectangle,
    the direction vector and the estimated angle (BGR format)
'''
def obtain_position(img):
    img_marked = img.copy()
    seg = seg_from_img(img)
    rej, rect = fit_mass(seg)

    if not rej:
        position, box, arrow = estimate_position(rect)
        # mark the fitted rectangle
        #cv2.drawContours(img_marked, [np.int0(box)], 0, (0,0,255), 2)
        # mark the direction vector
        #cv2.arrowedLine(img_marked, arrow[0], arrow[1], (0, 0, 255), 1, tipLength=0.25)
        # mark the estimated angle in degrees

        if position != 0:
            cv2.putText(img_marked, "{:.2f}".format(position), (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(img_marked, "0", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    else:
        # mark the rejection
        cv2.putText(img_marked, 'Reject', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        position = np.nan
    
    return rej, position, img_marked