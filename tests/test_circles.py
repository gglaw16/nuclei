from __future__ import division, print_function
import numpy as np
import scipy as sp
from scipy import ndimage
import cv2
import sys
from pprint import pprint
from random import randint
import nuclei.girder as g
import nuclei.girder.time_lapse_data as reader
import pdb


print("we expect python3")
import sys
print(sys.version_info)


source = reader.time_lapse_data()
# 073117 BT549_Sv40_Bt549_SV40 NLS GFP 53BP1 mcherry_2017_07_31__18_31_54series000
source.load('5aaf02831fbb9006233ae6a2')
print("series: %d"%source.get_number_of_series())
print("time steps in series 1: %d"%source.get_series_length(1))
im, item_obj = source.get_image(1,10)

item_id = item_obj['_id']

# Compute circles and put them into into girder.
gray_image = im[:,:,0]
circles = cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,1,50, \
                           param1=50,param2=30,minRadius=50,maxRadius=100)
pdb.set_trace()
g.upload_circle_annotation(item_id, circles)

print('https://images.slide-atlas.org/#item/%s'%item_id)






