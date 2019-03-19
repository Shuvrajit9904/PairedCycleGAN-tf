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
import os
import re

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

shape_pred = './data/aux/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_pred)


def parse_save(image, file,rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir):
    
    p = re.compile('(.*).jpg')
    out_file_init = p.match(file).group(1)
    
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
            print('shape, rad, center: ',result_array.shape, radius, center)
            print("reach: ", center[1] - radius, center[1] + radius)
            result_array = result_array[center[1] - radius:center[1] + radius,
                                center[0] - radius:center[0] + radius, :]
            print('2',result_array.shape)            
            out_file_name = output_dir + out_file_init + '_'+ name + '.jpg'
            if name == 'right_eye_eyebrow':
#                result_array = cv2.flip( result_array, 1 )
                print(result_array.shape)
                cv2.imwrite(out_file_name, cv2.flip( result_array, 1 ))
#                cv2.imshow("ROI", result_array)
#                cv2.waitKey(0)
    
            else:
#                print('k',result_array.shape)                
#                cv2.imwrite(out_file_name, result_array)
                pass
            
    

input_dir = './data/YMU/images/makeup_y/'
output_dir = './data/YMU/images/makeup_y_parsed/'
all_files = os.listdir(input_dir)
all_files.sort()
for file in all_files:
    input_file = input_dir + file
    image = cv2.imread(input_file)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)

#    parse_save(image, file,rects, predictor, FACIAL_LANDMARKS_IDXS, output_dir)
    break

#
#image = cv2.imread(args["image"])
image = cv2.imread('./data/YMU/images/152_2_y.jpg')
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
        
#        if name == 'right_eye_eyebrow':
#            cv2.imshow("right_eye_eyebrow", cv2.flip( result_array, 1 ))
#            cv2.waitKey(0)
#
#        else:
#            cv2.imshow("ROI", result_array)
#            cv2.waitKey(0)
        cv2.imshow("ROI", result_array)
        cv2.waitKey(0)
        
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
