#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:00:31 2018

@author: shuvrajit
"""

import os
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image
#from matplotlib import pyplot as plt


filename = 'YMU/Makeup_YMU.mat'

data = sio.loadmat(filename)

image_matrix = data['YMU_matrix']
image_file_names = data['YMU_filenames']
image_name_mat = {}

df = pd.DataFrame(image_matrix)



for column_idx in df:
    flat_data = np.asarray(df[column_idx])
    image_pixel = np.reshape(flat_data, (130, 150, 3))
    image_name_mat[image_file_names[0][column_idx][0]] = image_pixel
    img = Image.fromarray(image_pixel, 'RGB')
    img.save('YMU/images/'+image_file_names[0][column_idx][0])
    
    


