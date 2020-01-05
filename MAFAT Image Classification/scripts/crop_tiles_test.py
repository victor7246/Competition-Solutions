#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:03:04 2018

@author: asengup6
"""

''' 
This script reads each test image and selects the object on interest given in the test.csv file. 
Saves all the test images numpy data.
'''

import numpy as np
import pandas as pd
from PIL import Image
import os 

labels_file = '../data/test.csv'
im_dir = '../data/test imagery'

results_format_file =  '../data/answer.csv'


pad = 5  

resize_tile_shape = (128,128)

with open (labels_file, 'rb') as f:
    labels_csv = pd.read_csv(f)


with open(results_format_file,'rb') as f:
    results_format = pd.read_csv(f)



main_class_set = list(results_format.keys()[0:2])
sub_class_set = list(results_format.keys()[2:17])
color_set = list(results_format.keys()[17:25])
features_set = list(results_format.keys()[25:])

num_all = len(results_format.keys())
num_ids = len(labels_csv)

tiles_array = np.zeros([num_ids, resize_tile_shape[0], resize_tile_shape[1], 3])


bb_X_lndex = [2, 4, 6, 8]
bb_y_lndex = [3, 5, 7, 9]

def prepere_classification(class_gt, class_set):
    class_dict = {}
    for i, c in enumerate(class_set):
        class_dict[c] = i
    class_gt_index = np.zeros(len(class_gt))
    # pdb.set_trace()
    for i, c in enumerate(class_gt):
        class_gt_index[i] = class_dict[c.lower()]

    class_gt_one_hot = np.zeros([len(class_gt), len(class_set)])

    class_gt_one_hot[np.arange(len(class_gt)), np.int16(class_gt_index)] = 1
    return class_gt_one_hot






def prepere_one_id(im_file, index):
    im = Image.open(im_file)
    bb_x = labels_csv.iloc[index, bb_X_lndex].values.astype(np.int32)
    bb_y = labels_csv.iloc[index, bb_y_lndex].values.astype(np.int32)  
    x_min = np.min(bb_x) - pad
    y_min = np.min(bb_y) - pad
    x_max = np.max(bb_x) + pad
    y_max = np.max(bb_y) + pad
    tile = im.crop([x_min, y_min, x_max, y_max])
    tile_resized = tile.resize(resize_tile_shape)
    tiles_array[index,:] = np.array(tile_resized)[:,:,0:3]




im_names = os.listdir(im_dir)

for i in range(num_ids):
    for im_name in im_names:
        if str(labels_csv['image_id'][i]) in im_name:
            im_file = os.path.join(im_dir,im_name)
            
    prepere_one_id(im_file, i)
    if np.mod(i,1000) == 0 :
        print (i)
np.save('../data/test_tiles_array', tiles_array)  




