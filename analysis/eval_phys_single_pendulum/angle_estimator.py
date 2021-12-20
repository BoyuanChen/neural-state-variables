'''
This script provides utility functions estimating the angle of a single pendulum from image.
Step 1. Extract the pendulum from the image.
Step 2. Do a rectangle fitting of the pendulum.
Step 3. Estimate the angle of the pendulum. 
Certain images will be rejected if the pendulum does not exist or has a wrong shape.
'''

import cv2
import os
import numpy as np


'''
Extract the pendulum from the image.
Args:
    img: single pendulum image in BGR format
Returns:
    seg: segmentation of the pendulum
'''
def seg_from_img(img):
    # pixel thresholds (in HSV)
    v_min = (60, 0, 0)
    v_max = (255, 255, 255)
    # extract the pendulum
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    seg = cv2.inRange(img_hsv, v_min, v_max)

    return seg


'''
Fit pendulum in a rectangle.
Args:
    seg: segmentation of the pendulum
Returns:
    rej: (True/False) if the image is rejected
    rect: (Box2D structure) the fitted rectangle
'''
def fit_pendulum(seg):
    # find all contours
    contours, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # reject if no contours found
    if len(contours) == 0:
        return True, None
    # find the contour with the maximum area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    # reject if the contour is too small
    if area < 400:
        return True, None
    # rectangle fitting
    rect = cv2.minAreaRect(cnt)
    _, (width, height), _ = rect
    # reject if the rectangle is too close to a square
    if abs(height-width) < 35:
        return True, None
    # reject if the rectangle does not properly fit the contour
    if height*width > 1.5 * area:
        return True, None
    
    return False, rect


'''
Estimate the angle of pendulum from its fitted rectangle.
Args:
    rect: (Box2D structure) the fitted rectangle
Returns:
    angle: the estimated angle in radians in range (0, 2*pi)
    box: box points of the rectangle
    arrow: the arrow pointing along the angle
'''
def estimate_angle(rect):
    # box points
    box = cv2.boxPoints(rect)
    # center, width and height
    (cx, cy), (width, height), _ = rect
    # specify the direction vector of the pendulum
    if width < height:
        v_d = box[0] - box[1]
    else:
        v_d = box[0] - box[3]
    # reference vector
    v_ref = np.array([cx-64, cy-64])
    # choose the one from v_d and -v_d whose angle with v_ref
    # is less than pi
    if np.dot(v_d, v_ref) < 0:
        v_d = -v_d
    # counterclockwise angle from (0,-1) to v_d
    angle = np.arctan2(-v_d[1], v_d[0]) + np.pi/2
    if angle < 0:
        angle += 2*np.pi
    arrow = ((int(cx), int(cy)), (int(cx+0.7*v_d[0]), int(cy+0.7*v_d[1])))
    return angle, box, arrow


'''
Obtain the angle of single pendulum from image
Args:
    img: single pendulum image in BGR format
Returns:
    rej: (True/False) if the image is rejected
    angle: the estimated angle in radians in range (0, 2*pi)
    img_marked: image marked with the fitted rectangle,
    the direction vector and the estimated angle (BGR format)
'''
def obtain_angle(img):
    img_marked = img.copy()
    seg = seg_from_img(img)
    rej, rect = fit_pendulum(seg)

    if not rej:
        angle, box, arrow = estimate_angle(rect)
        # mark the fitted rectangle
        cv2.drawContours(img_marked, [np.int0(box)], 0, (0,0,255), 2)
        # mark the direction vector
        cv2.arrowedLine(img_marked, arrow[0], arrow[1], (0, 0, 255), 1, tipLength=0.25)
        # mark the estimated angle in degrees
        cv2.putText(img_marked, str(round(angle*180/np.pi)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        # mark the rejection
        cv2.putText(img_marked, 'Reject', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        angle = np.nan
    
    return rej, angle, img_marked