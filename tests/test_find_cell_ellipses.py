from __future__ import division, print_function
import numpy as np
import scipy as sp
from scipy import ndimage
import cv2
import sys
from pprint import pprint
from random import randint
from nuclei.foci import *
import nuclei.girder as g
import nuclei.girder.time_lapse_data as reader
import pdb


pdb.set_trace()
source = reader.time_lapse_data()
source.load('5aaf02831fbb9006233ae6a2')
series_idx = 0
img1, item_obj = source.get_image(series_idx,0)
img_bw = img1.copy()
img_bw[:,:,0] = np.maximum(img1[:,:,1], img1[:,:,2])
img_bw[:,:,1] = img_bw[:,:,0]
img_bw[:,:,2] = img_bw[:,:,0]

ellipses1 = find_cell_ellipses(img_bw)

print(ellipses1)
