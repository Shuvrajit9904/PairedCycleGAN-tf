#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:00:31 2018

@author: shuvrajit
"""

#import os
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image
import re
#from matplotlib import pyplot as plt


filename = 'data/YMU/Makeup_YMU.mat'

data = sio.loadmat(filename)

image_matrix = data['YMU_matrix']
image_file_names = data['YMU_filenames']
image_name_mat = {}

df = pd.DataFrame(image_matrix)

p_y = re.compile(".*_y.jpg")
p_n = re.compile(".*_n.jpg")


for column_idx in df:
    flat_data = np.asarray(df[column_idx])
    image_pixel = np.reshape(flat_data, (150, 130, 3), order = "F")
    image_name_mat[image_file_names[0][column_idx][0]] = image_pixel
    img = Image.fromarray(image_pixel, 'RGB')
    if p_y.match(image_file_names[0][column_idx][0]):
        img.save('data/YMU/images/makeup_y/'+image_file_names[0][column_idx][0])
    if p_n.match(image_file_names[0][column_idx][0]):
        img.save('data/YMU/images/makeup_n/'+image_file_names[0][column_idx][0])
    
    


