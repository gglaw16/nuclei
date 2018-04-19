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

# test generating and saveing tracks to girder.

tracks = []
item_ids = []

source = reader.time_lapse_data()
source.load('5aaf02831fbb9006233ae6a2')
series_idx = 0
num_time_steps = source.get_series_length(series_idx)
img1, item_obj = source.get_image(series_idx,0)
item_ids.append(item_obj['_id'])

img_bw = img1.copy()
img_bw[:,:,0] = np.maximum(img1[:,:,1], img1[:,:,2])
img_bw[:,:,1] = img_bw[:,:,0]
img_bw[:,:,2] = img_bw[:,:,0]
img1 = img_bw
ellipses1 = find_cell_ellipses(img_bw)
for t in range(1,num_time_steps):
    img2, item_obj = source.get_image(series_idx, t)
    item_ids.append(item_obj['_id'])

    img_bw = img2.copy()
    img_bw[:,:,0] = np.maximum(img1[:,:,1], img1[:,:,2])
    img_bw[:,:,1] = img_bw[:,:,0]
    img_bw[:,:,2] = img_bw[:,:,0]
    img2 = img_bw
    ellipses2 = find_cell_ellipses(img2)
    ellipse_pairs = match_ellipses(ellipses1, ellipses2)
    
    # lnsegs are the line segments we want to add to our poly lines.
    for pair in ellipse_pairs:
        track = find_track_that_ends_with_ellipse(tracks, pair[0])
        if not track:
            # The line doe not exist. This is the first line segment.
            # make a line wit a single point (the start)
            track = Track()
            track.add_ellipse(pair[0], t-1)
            tracks.append(track)
        # We found the line that ends with the new segment's first point.
        # Add the second point to the line
        track.add_ellipse(pair[1], t)
    img1 = img2
    ellipses1 = ellipses2

tmp = img2.copy()
for track in tracks:
    track.draw_in_frame(tmp, num_time_steps)

pdb.set_trace()
g.upload_tracks(tracks, item_ids)    


