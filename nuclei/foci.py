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
        prevdist = 3000
        pt2 = e2[0]
        for e1 in ellipses1:

            dist = ((e2[0][0]-e1[0][0])**2+(e2[0][1]-e1[0][1])**2)**.5
            if dist < prevdist:
                prevdist = dist
                closest = e1

        dict_e2_e1[e2] = closest

    dict_e1_e2 = {}
    for e1 in ellipses1:
        prevdist = 3000
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
    threshold = 40
    ret,mask = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)

    # it is hard to see what the media does because the noise has such low amplitude.
    # you can see the dark noise speckles in the bright nucei get reduced.
    median = cv2.medianBlur(mask, 5)

    # Lets try to get rid of dark spots in nuclei with a closing filter.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)).astype(np.uint8)
    closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Dilate to get background 
    kernel = np.ones((3,3))
    sure_background = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel, iterations = 20)

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

#makes a b/w mask of a solid ellipse
def make_ellipse_mask(ellipse,time, data_dir):
    img = cv2.imread(data_dir + "channel1time%d.png"%time)
    black_img = np.zeros(img.shape, dtype=np.uint8)
    cv2.ellipse(black_img, ellipse, (1,1,1), -1)
    return(black_img)

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


def rank_merge_tracks(track1, track2):
    #this returns the max dist between tracks
    prevdist = -1
    for time in range(track1.get_start_time(),track1.get_end_time() + 1):
        ell1 = track1.get_ellipse_from_time(time)
        for i in range(-5,5):
            ell2 = track2.get_ellipse_from_time(time+i)
            if ell1 != None and ell2 != None:
                dist = ((ell2[0][0]-ell1[0][0])**2+(ell2[0][1]-ell1[0][1])**2)**.5
                if dist > prevdist:
                    prevdist = dist
    #in case its always None
    if prevdist == -1:
        return 50000
    return prevdist

def find_best_track_to_merge(track1, tracks):
    # Todo: if neither end is in the spatial temporal center, return None.    
    best_track = None
    for track2 in tracks:
        if track2 != track1:
            score = rank_merge_tracks(track1, track2)
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
    
#=========================================================================
# Girder stuff.

# Get a time sample image from girder.
# gc: girder_client authenticated to read the folder of interest.
# folder_id:  id of the series folder containing a lsit of images.
# time_idx: Integer index of the time sample requested.  Folder items names
#   must be formated "time04d%"%time_idx (for now).
# numpy array is returned.  Images now are stored as 3 cahnnel RGB.
def load_time_image(gc, folder_id, time_idx):
    time_step_name = "time%04d"%time_idx
    resp = gc.get("item?folderId=%s&name=%s"%(folder_id, time_step_name))
    if len(resp) == 0:
        return None
    item_id = resp[0]['_id']
    
    item_obj = gc.get("item/%s"%item_id)
    image_name = item_obj['name']
    resp = gc.get("item/%s/files"%item_id)
    png_id = None
    for file_info in resp:
        if os.path.splitext(file_info['name'])[1] == '.png':
            png_id = file_info['_id']
    if png_id:
        url = gc.urlBase + "file/" + png_id + "/download"
        #req = urllib.request.urlopen(url)
        headers = {'Girder-Token': gc.token}
        try:
            req = urllib.request.Request(url, None, headers)
            response = urllib.request.urlopen(req)
            #image = response.read()
            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            print('image loaded %s'%png_id)
            return (image, item_obj)
        except urllib.HTTPError:
            if err.code == 400:
                print("Bad request!")
            elif err.code == 404:
                print("Page not found!")
            elif err.code == 403:
                print("Access denied!")
            else:
                print("Something happened! Error code %d" % err.code)

    # get the image through the tile API
    #image = get_girder_large_cutout(gc, item_id, 1)
    return (None, None)











