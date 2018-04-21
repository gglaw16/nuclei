import girder_client
import nuclei.girder as g
import urllib.request
import numpy as np
import cv2
import math
import os
import pdb
import random


def match_ellipses(ellipses1, ellipses2):
    dict_e2_e1 = {}
    for e2 in ellipses2:
        prevdist = 300000
        pt2 = e2[0]
        for e1 in ellipses1:
            dist = ((e2[0][0]-e1[0][0])**2+(e2[0][1]-e1[0][1])**2)**.5
            if dist < prevdist:
                prevdist = dist
                closest = e1
        dict_e2_e1[e2] = closest

    dict_e1_e2 = {}
    for e1 in ellipses1:
        prevdist = 300000
        pt1 = e1[0]
        for e2 in ellipses2:
            dist = ((e1[0][0]-e2[0][0])**2+(e1[0][1]-e2[0][1])**2)**.5
            if dist < prevdist:
                prevdist = dist
                closest = e2
        dict_e1_e2[e1] = closest

    # find all the points which map back to themselves.
    # Make a list of segments (pairs of points)
    ellipse_pairs = []
    for e2 in dict_e2_e1.keys():
        e1 = dict_e2_e1[e2]
        # If the point maps back on itself ...
        if e2 == dict_e1_e2[e1]:
            ellipse_pairs.append([e1, e2])       

    return ellipse_pairs

# combine an image with a mask as a transparent overlay
# draw the mask on the image in red
# image and mask must have the same shape
def mask_on_image(mask, img):
    out = img.copy()
    if len(mask.shape) == 3:
        mask = mask[...,0]
    #out[mask>128,0] = 255
    #out[mask>128,1] = 0
    #out[mask>128,2] = 0
    out[...,0] = np.maximum(out[...,0], mask)
    mask = ~mask
    out[...,1] = np.minimum(out[...,1], mask)
    out[...,2] = np.minimum(out[...,2], mask)
    
    return out

def find_cell_ellipses(img):    
    threshold = 20
    ret,mask = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)

    # it is hard to see what the media does because the noise has such low amplitude.
    # you can see the dark noise speckles in the bright nucei get reduced.
    median = cv2.medianBlur(mask, 5)

    # Lets try to get rid of dark spots in nuclei with a closing filter.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)).astype(np.uint8)
    closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to get background 
    kernel = np.ones((3,3))
    sure_background = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel, iterations = 20)

    tmp = mask_on_image(closing, sure_background)

    #plt.imshow(tmp)
    #plt.show()
    # Distance threshold to separate cells

    tmp = closing[...,0]

    #print(tmp.shape)
    #print(tmp.dtype)


    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(tmp,cv2.DIST_L2,5)
    ret, sure_cells = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

    # Finding unknown region
    sure_cells = np.uint8(sure_cells)

    #print(sure_cells.shape)
    #plt.imshow(sure_cells)
    #plt.show()
    #print(sure_cells.shape)
    cell_mask = sure_cells.copy()
    background_mask = cv2.cvtColor(sure_background, cv2.COLOR_RGB2GRAY)

    unknown = cv2.subtract(background_mask, cell_mask)


    ret, markers = cv2.connectedComponents(cell_mask)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #print(markers.shape)
    #print(markers.dtype)

    #plt.imshow(markers)
    #plt.show()
    tmp = closing.copy()
    from pprint import pprint
    ws = cv2.watershed(tmp,markers)

    tmp = ws.copy()
    #print((np.min(tmp),np.max(tmp)))
    tmp = np.clip(tmp, 1, 255)-1
    #print((np.min(tmp),np.max(tmp)))
    tmp = tmp.astype(np.uint8)

    kernel = np.ones((3,3))
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_ERODE, kernel, iterations = 1)

    _,contours,_ = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
    ellipse_list = []
    for cont in contours:
        if len(cont) > 4:
            area = cv2.contourArea(cont)
            if area > 100:
                e = cv2.fitEllipse(cont)
                ellipse_list.append(e)

    return ellipse_list

#tells you whether a given point is above a given line
def is_above_line(a,b,x,y):
    yforxinline = a*x + b
    if(yforxinline > y):
        return(True)
    return(False)

class Track:
    def __init__(self):
        self.ellipses = []
        self.times = []
        # color to draw the track
        # todo: randomize colors
        # not sure about this format
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

    def p(self):
        length = len(self.ellipses)       
        print("Track (length = %d):"%length)
        for i in range(length):
              pt = self.ellipses[i][0]
              print("  %d: (%f, %f)"%(self.times[i],pt[0],pt[1]))
        
    #def get_ellipses(self):
    #    return self.ellipses
        
    #def get_times(self):
    #    return self.times
    
    def add_ellipse(self, ellipse, time):
        self.ellipses.append(ellipse)
        self.times.append(time)

    def get_ellipse(self, idx):
        return (self.ellipses[idx], self.times[idx])

    def get_length(self):
        return len(self.ellipses)
    
    def get_ellipse_from_time(self,time):
        for i in range(0,len(self.ellipses)):
            if(self.times[i] == time):
                return self.ellipses[i]
        return None
    
    def get_time_from_ellipse(self,ellipse):
        for i in range(0,len(self.times)):
            if(self.ellipses[i] == ellipse):
                return self.times[i]
        return None

    # right now this depends that the ellipses are added sorted in time.
    def get_start_time(self):
        if len(self.times) == 0:
            return None
        return self.times[0]

    # right now this depends that the ellipses are added sorted in time.
    def get_end_time(self):
        if len(self.times) == 0:
            return None
        return self.times[-1]

    # draw all the segments up to "time".  Draw the ellipse at time "time"
    def draw_in_frame(self, img, time):
        pt1 = self.ellipses[0][0]
        for i in range(1,len(self.ellipses)): 
            pt2 = self.ellipses[i][0]
            if(self.times[i] <= time):
                cv2.line(img,(int(pt1[0]),int(pt1[1])),(int(pt2[0]),int(pt2[1])),self.color,5)
            pt1 = pt2
        return img
    
    #merge another track to the beginning of this track
    def merge_track(self, track):
        start = min(self.get_start_time(), track.get_start_time())
        end = max(self.get_end_time(), track.get_end_time())
        new_ellipses = []
        new_times = []
        for t in range(start, end+1):
            e1 = self.get_ellipse_from_time(t)
            e2 = track.get_ellipse_from_time(t)
            if e1 != None:
                # TODO: merge ellipses if we have two
                new_ellipses.append(e1)
                new_times.append(t)
            elif e2 != None:
                new_ellipses.append(e2)
                new_times.append(t)
        self.ellipses = new_ellipses
        self.times = new_times
    
    #returns the time of the first ellipse higher than a line
    def cross_line(self, a, b):
        for ellipse in self.ellipses:
            if is_above_line(a,b,ellipse[0][0],ellipse[0][1]):
                return self.get_time_from_ellipse(ellipse)
        return -1

    def get_hex_color(self):
        hex_digits = '0123456789abcdef'
        hex_color = '#'
        for i in range(3):
            tmp = self.color[i]
            digit1 = int(math.floor(tmp/16))
            hex_color += hex_digits[digit1]
            digit2 = tmp%16
            hex_color += hex_digits[digit2]
        return hex_color
    
    
def get_frame(idx, data_dir):
    img1 = cv2.imread(data_dir + "channel1time%d.png"%idx)
    img2 = cv2.imread(data_dir + "channel2time%d.png"%idx)
    # combination of two channels.
    img_bw = np.maximum(img1, img2)
    return img_bw

def get_and_combine_frames(idx, data_dir):
    img1 = cv2.imread(data_dir + "channel1time%d.png"%idx)*[1,0,0]
    img2 = cv2.imread(data_dir + "channel2time%d.png"%idx)*[0,1,0]
    img3 = cv2.imread(data_dir + "channel3time%d.png"%idx)*[0,0,1]
    img_fin = img1 + img2 + img3
    return img_fin

def find_track_that_ends_with_ellipse(tracks, e):
    for track in tracks:
        last_ellipse, _ = track.get_ellipse(-1)
        if last_ellipse == e:
            return track
    return None


def rank_merge_tracks(track1, track2, rows=[]):
    #this returns the max dist between tracks
    prevdist = -1
    for time in range(track1.get_start_time(),track1.get_end_time() + 1):
        ell1 = track1.get_ellipse_from_time(time)
        for i in range(-5,5):
            ell2 = track2.get_ellipse_from_time(time+i)
            dist = getDistBtwEll(ell1,ell2, rows)
            if dist > prevdist:
                    prevdist = dist
    #in case its always None
    if prevdist == -1:
        return 50000
    return prevdist

# This modifies Euclidean distance by ignoring a rectangle of space around rows.
# coded for horizontal rows (It just shrinks the y dimension)
def getDistBtwEll(ell1, ell2, rows):
    if ell1 != None and ell2 != None:
        dx = ell2[0][0] - ell1[0][0]
        dy = ell2[0][1] - ell1[0][1]
        for row in rows:
            if (ell1[0][1] < row[0]*ell1[0][0]+row[1] and ell2[0][1] > row[0]*ell2[0][0]+row[1]) or \
               (ell2[0][1] < row[0]*ell2[0][0]+row[1] and ell1[0][1] > row[0]*ell1[0][0]+row[1]):
                dy = dy-100
        if (dy < 0):
            dy = 0.0
        dist = (dx**2 + dy**2)**.5
        return dist
    return 10000000
    
def find_best_track_to_merge(track1, tracks, rows=[]):
    # Todo: if neither end is in the spatial temporal center, return None.    
    best_track = None
    for track2 in tracks:
        if track2 != track1:
            score = rank_merge_tracks(track1, track2, rows)
            if best_track == None or best_score > score:
                best_score = score
                best_track = track2       
    if best_track == None:
        return None
    # arbitrary cutoff.
    if best_score > 200:
        return None
    return best_track

# Loop over all times
# img2 = get_frame(t)
def save_frame(tm, data_dir, combined_tracks):
    im = get_frame(tm, data_dir)
    for track in combined_tracks:
        track.draw_in_frame(im, tm)
    cv2.imwrite("test%02d.png"%tm, im)

# detect cirlces in the first channel of an image and fit 3 horizontal rows to the circle centers.
# returns [ [slope,y], [slope,y], [slope, y] ]
# todo make this more general (number of rows..
def compute_rows(img):
    gray_image = img[:,:,0]
    circles = cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,1,50, \
                               param1=50,param2=30,minRadius=50,maxRadius=100)
    row1x = []
    row2x = []
    row3x = []
    row1y = []
    row2y = []
    row3y = []
    row1r = []
    row2r = []
    row3r = []

    row1x.append(circles[0][0][0])
    row1y.append(circles[0][0][1])
    row1r.append(circles[0][0][2])

    for circle in circles[0]:
        if circle[1] < row1y[0]+row1r[0]+30 and circle[1] > row1y[0]-row1r[0]-30:
            row1x.append(circle[0])
            row1y.append(circle[1])
            row1r.append(circle[2])
        elif (row2x != [] and circle[1] < row2y[0]+row2r[0]+30 and circle[1] > row2y[0]-row2r[0]-30) or row2x == []:
            row2x.append(circle[0])
            row2y.append(circle[1])
            row2r.append(circle[2])
        else:
            row3x.append(circle[0])
            row3y.append(circle[1])
            row3r.append(circle[2])
            
    if row1y[1] < row2y[1] and row1y[1] < row3y[1]:
        row1 = np.polyfit(row1x, row1y,1)
        if row2y[1] < row3y[1]:
            row2 = np.polyfit(row2x, row2y,1)
            row3 = np.polyfit(row3x, row3y,1)
        else :
            row3 = np.polyfit(row2x, row2y,1)
            row2 = np.polyfit(row3x, row3y,1)
    elif row2y[1] < row3y[1] and row2y[1] < row1y[1]:
        row1 = np.polyfit(row2x, row2y,1)
        if row1y[1] < row3y[1]:
            row2 = np.polyfit(row1x, row1y,1)
            row3 = np.polyfit(row3x, row3y,1)
        else :
            row3 = np.polyfit(row1x, row1y,1)
            row2 = np.polyfit(row3x, row3y,1)
    else :
        row1 = np.polyfit(row3x, row3y,1)
        if row1y[1] < row2y[1]:
            row2 = np.polyfit(row1x, row1y,1)
            row3 = np.polyfit(row2x, row2y,1)
        else :
            row3 = np.polyfit(row1x, row1y,1)
            row2 = np.polyfit(row2x, row2y,1)

    return [row1, row2, row3]








