#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:14:49 2019

@author: shuvrajit
"""

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

#FACIAL_LANDMARKS_IDXS = OrderedDict([
#	("mouth", (48, 68)),
#	("right_eyebrow", (17, 22)),
#	("left_eyebrow", (22, 27)),
#	("right_eye", (36, 42)),
#	("left_eye", (42, 48)),
#	("nose", (27, 35)),
#	("jaw", (0, 17))
#])

mouth_idx = np.arange(48, 68)
right_eyebrow_idx = np.arange(17, 22)
left_eyebrow_idx = np.arange(22, 27)
right_eye_idx = np.arange(36,42)
left_eye_idx = np.arange(42, 48)
nose_idx = np.arange(27, 35)


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", mouth_idx),
    ("right_eye_eyebrow", np.append(right_eyebrow_idx, right_eye_idx)),
    ("left_eye_eyebrow", np.append(left_eyebrow_idx, left_eye_idx)),
	("nose", nose_idx),
])


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
args = vars(ap.parse_args())

args["shape_predictor"] = './data/aux/shape_predictor_68_face_landmarks.dat'
args["image"] = './data/YMU/images/001_1_n.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)



for (i, rect) in enumerate(rects):

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    for (name, idx_arr) in FACIAL_LANDMARKS_IDXS.items():

        clone = image.copy() 

        (x,y),radius = cv2.minEnclosingCircle(np.array([shape[idx_arr]]))  
        center = (int(x),int(y))  
        radius = int(radius) + 20   
        
        mask = np.zeros(clone.shape, dtype=np.uint8)  
        mask = cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)
        
        result_array = clone & mask
        result_array = result_array[center[1] - radius:center[1] + radius,
                            center[0] - radius:center[0] + radius, :]

        cv2.imshow("ROI", result_array)
        cv2.waitKey(0)
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
