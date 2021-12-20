'''
This script provides utility functions estimating angles of a double pendulum from image.
Step 1. Extract each of the two pendulums from the image.
Step 2. Do a rectangle fitting of each pendulum.
Step 3. Estimate the angles of the pendulums. 
Certain images will be rejected if any pendulum does not exist, is hidden, or has a wrong shape.
'''

import cv2
import os
import numpy as np


'''
Extract the two pendulums from the image.
Args:
    img: double pendulum image in BGR format
Returns:
    seg_1: segmentation of the first pendulum
    seg_2: segmentation of the second pendulum
'''
def seg_from_img(img):
    # pixel thresholds (in HSV)
    v_min_1 = (0, 0, 0)
    v_max_1 = (255, 255, 140)
    v_min_2 = (60, 0, 0)
    v_max_2 = (255, 255, 255)
    # background color in HSV (RGB: [215, 205, 192])
    bg_color = [17, 27, 215]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the first pendulum
    seg_1 = cv2.inRange(img_hsv, v_min_1, v_max_1)
    # remove the first pendulum by replacing it with background pixels
    img_hsv[seg_1==255] = bg_color
    # extract the second pendulum
    seg_2 = cv2.inRange(img_hsv, v_min_2, v_max_2)

    return seg_1, seg_2


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
    if area < 125:
        return True, None
    # rectangle fitting
    rect = cv2.minAreaRect(cnt)
    _, (width, height), _ = rect
    # reject if the rectangle is too close to a square
    if abs(height-width) < 8:
        return True, None
    # reject if the rectangle does not properly fit the contour
    if height*width > 2 * area:
        return True, None

    return False, rect


'''
Estimate the angle of pendulum from its fitted rectangle.
Args:
    rect: (Box2D structure) the fitted rectangle
    ref_pt: reference point to determine the arrow direction
Returns:
    angle: the estimated angle in radians in range (0, 2*pi)
    box: box points of the rectangle
    arrow: the arrow pointing along the angle
'''
def estimate_angle(rect, ref_pt):
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
    v_ref = np.array([cx, cy]) - ref_pt
    # choose the one from v_d and -v_d whose angle with v_ref
    # is less than pi
    if np.dot(v_d, v_ref) < 0:
        v_d = -v_d
    # counterclockwise angle from (0,-1) to v_d
    angle = np.arctan2(-v_d[1], v_d[0]) + np.pi/2
    if angle < 0:
        angle += 2*np.pi
    arrow = ((int(cx), int(cy)), (int(cx+v_d[0]), int(cy+v_d[1])))
    return angle, box, arrow


'''
Obtain the angles of double pendulum from image
Args:
    img: double pendulum image in BGR format
Returns:
    rej: (True/False) if the image is rejected
    angles: the estimated angles in radians in range (0, 2*pi)
    img_marked: image marked with the fitted rectangles,
    the direction vectors and the estimated angles (BGR format)
'''
def obtain_angle(img):
    img_marked = img.copy()
    seg_1, seg_2 = seg_from_img(img)
    rej_1, rect_1 = fit_pendulum(seg_1)
    rej_2, rect_2 = fit_pendulum(seg_2)
    if rej_1:
        rej = True
        angles = []
        cv2.putText(img_marked, 'Reject arm 1', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    elif rej_2:
        rej = True
        angles = []
        cv2.putText(img_marked, 'Reject arm 2', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        rej = False
        img_center = np.array([64, 64])
        angle_1, box_1, arrow_1 = estimate_angle(rect_1, img_center)
        (cx, cy), _, _ = rect_1
        pd_1_end = 2 * np.array([cx, cy]) - img_center
        angle_2, box_2, arrow_2 = estimate_angle(rect_2, pd_1_end)
        angles = [angle_1, angle_2]
        # mark the fitted rectangles
        cv2.drawContours(img_marked, [np.int0(box_1)], 0, (0,0,255), 2)
        cv2.drawContours(img_marked, [np.int0(box_2)], 0, (0,0,255), 2)
        # mark the direction vectors
        cv2.arrowedLine(img_marked, arrow_1[0], arrow_1[1], (0, 0, 255), 1, tipLength=0.25)
        cv2.arrowedLine(img_marked, arrow_2[0], arrow_2[1], (0, 0, 255), 1, tipLength=0.25)
        # mark the estimated angles in degrees
        text = str(round(angle_1*180/np.pi)) + ' ' + str(round(angle_2*180/np.pi))
        cv2.putText(img_marked, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return rej, angles, img_marked