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

# test loading tracks from girder.


source = reader.time_lapse_data()
source.load('5aaf02831fbb9006233ae6a2')
series_idx = 0
num_time_steps = source.get_series_length(series_idx)
img1, item_obj = source.get_image(series_idx,0)

annotation = g.Annotation("tracks")
pdb.set_trace()
annotation.LoadFromItem(item_obj['_id'])

tracks = annotation.annot_obj


print(len(tracks))



