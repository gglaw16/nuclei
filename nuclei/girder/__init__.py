import numpy as np
import scipy
import scipy.misc
import cv2
import math
import os
import sys
import urllib.request
import girder_client
from nuclei.girder.images_key import *
import pdb
# this may not be necessary to access tracks.
from nuclei.foci import *




def get_gc(gc=None):
    if gc is None:
        gc = girder_client.GirderClient(apiUrl= GIRDER_URL+'/api/v1')
        gc.authenticate(GIRDER_USERNAME, apiKey=GIRDER_KEY)
    return gc

#------------------------------------------------------------------------------
# Object to make writting and reading annotations easier (encapsulated).

class Annotation:
    name = "annotation"

    def __init__(self, name):
        self.annot_obj = {"elements":[],"name":name}

        
    def SetName(self, new_name):
        self.annot_obj['name'] = new_name
        
    def AddCircle(self, center, radius, color, line_width):
        element = {'type': 'circle', \
                   'center': [int(center[0]), int(center[1]), 0], \
                   'radius': int(radius), \
                   'lineColor': '#00ff00', \
                   'lineWidth': int(line_width)}
        self.annot_obj['elements'].append(element)
        
    def AddPolyline(self, points, color, line_width=0):
        if len(points) < 2:
            return
        element = {'type': 'polyline', \
                   'points': points, \
                   'lineColor': color, \
                   'lineWidth': int(line_width)}
        self.annot_obj['elements'].append(element)
        
    def SaveToItem(self, item_id, overwrite, gc=None):
        gc = get_gc(gc)
        annotation_id = None
        if overwrite:
            # see if we have an annotation by this name already.
            resp = gc.get("annotation?itemId=%s&name=%s"%(item_id, self.annot_obj['name']))
            if len(resp) > 0:
                annotation_id = resp[0]["_id"]

        if annotation_id:
            resp = gc.put("annotation/%s"%annotation_id, json=self.annot_obj)
        else:
            resp = gc.post("annotation", parameters={"itemId":item_id}, json=self.annot_obj)
        return resp["_id"]

    # Uses the name set previously.
    def LoadFromItem(self, item_id, gc=None):
        gc = get_gc(gc)
        annotation_id = None

        # First get the id from the name.
        resp = gc.get("annotation?itemId=%s&name=%s"%(item_id, self.annot_obj['name']))
        if len(resp) == 0:
            return False
        annotation_id = resp[0]["_id"]

        # now load the annotation
        resp = gc.get("annotation/%s"%annotation_id)
        self.annot_obj = resp['annotation']


# TODO: item ids should be stored in the track
# This is tricky. each image gets an annotation with all tracks upto that time.
# there are too many tracks to make separate annotation for each.
# Girder may allow adding annotation elements separately in the future.
def upload_tracks(tracks, item_ids, name='tracks', gc=None):
    for item_time, item_id in enumerate(item_ids):
        annotation = Annotation(name)
        for track in tracks:
            end_time = min(track.get_end_time(), item_time)
            points = []
            for t in range(track.get_start_time(), end_time+1):
                ellipse = track.get_ellipse_from_time(t)
                if ellipse != None:
                    # I do not think the ellispe willever be none, but ... this cannot hurt
                    x = int(ellipse[0][0])
                    y = int(ellipse[0][1])
                    points.append((x,y,0))
            annotation.AddPolyline(points, track.get_hex_color())
        annotation.SaveToItem(item_id, overwrite=True, gc=gc)


    
# Find circles in an image and upload to girder (as anntoations).
# gc: girder_client authenticated to write to the item massed in.
# item_obj: item to receive the annotations
# returns the circles found by opencv
def upload_circle_annotation(item_id, circles, gc=None):
    gc = get_gc(gc)
    annotation = Annotation("Circles")
    for i in circles[0]:
        annotation.AddCircle((i[0],i[1], 0),i[2],(0,255,0),2)
    annotation.SaveToItem(item_id, overwrite=True, gc=gc)


# Looks for a png in an item and then downloads it into a numpy array.
# it returns a pair (image, item_info)
def read_item_image(item_id, gc=None):
    item_obj = gc.get("item/%s"%item_id)
    image_name = item_obj['name']
    resp = gc.get("item/%s/files"%item_id)
    png_id = None
    for file_info in resp:
        if os.path.splitext(file_info['name'])[1] == '.png':
            png_id = file_info['_id']
    if png_id:
        (image, file_info) = read_file_image(png_id, gc)
        return (image, item_obj)
    return (None, None)


# downloads all the images in an item.  Returns a list of numpy images.
def read_item_images(item_id, gc=None):
    gc = get_gc(gc)
    resp = gc.get('item/%s/files'%item_id)
    images = []
    for file_obj in resp:
        im, item_obj = read_file_image(file_obj['_id'], gc)
        if item_obj != None:
            images.append(im)
    return images
    

# downloads an image into a numpy array without going to disk.
# Get an image from a girder file id
# TODO: support more than just pngs.
# it returns a pair (image, file_info)
def read_file_image(file_id, gc=None):
    gc = get_gc(gc)
    file_info = gc.get("file/%s"%file_id)
    if os.path.splitext(file_info['name'])[1] == '.png':
        url = gc.urlBase + "file/" + file_id + "/download"
        headers = {'Girder-Token': gc.token}
        try:
            req = urllib.request.Request(url, None, headers)
            response = urllib.request.urlopen(req)
            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            #print('image loaded %s'%file_id)
            return (image, file_info)
        except urllib.request.HTTPError:
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


# Upload the image to girder (for debugging)
def upload_image(image, item_name, destination_folder_id,
                 gc=None, stomp=True):
    gc = get_gc(gc)
    resp = gc.get("item?folderId=%s&name=%s&limit=50&offset=0&sort=lowerName&sortdir=1" \
                  %(destination_folder_id, item_name))
    if not stomp or len(resp) == 0:
        girder_item = gc.createItem(destination_folder_id, item_name,
                                    "debugging image")
    else:
        girder_item = resp[0]
        resp = gc.get("item/%s/files?limit=500"% girder_item['_id'])
        for f in resp:
            gc.delete("file/%s"%f['_id'])
    gc.addMetadataToItem(girder_item['_id'], {'lightbox': 1})

    output_path = '/tmp'
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tmp_file_name = os.path.join(output_path, 'image.png')
    scipy.misc.imsave(tmp_file_name, image)
    # upload the file into the girder item
    gc.uploadFileToItem(girder_item['_id'], tmp_file_name)


# Upload an array of images to girder light box.
def upload_images(images, item_name, destination_folder_id,
                  num=None, gc=None, stomp=True, filenames=None):
    gc = get_gc(gc)
    resp = gc.get("item?folderId=%s&name=%s&limit=50&offset=0&sort=lowerName&sortdir=1" \
                  %(destination_folder_id, item_name))
    if not stomp or len(resp) == 0:
        girder_item = gc.createItem(destination_folder_id, item_name,
                                    "training images for debugging")
    else:
        girder_item = resp[0]
        resp = gc.get("item/%s/files?limit=500"% girder_item['_id'])
        for f in resp:
            gc.delete("file/%s"%f['_id'])
    gc.addMetadataToItem(girder_item['_id'], {'lightbox': 1})

    total = len(images)
    if total == 0:
        return
    tmp = total
    if num != None and num < total:
        tmp = num

    for i in range(tmp):
        image = images[i]
        output_path = '/tmp'
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if filenames == None:
            tmp_file_name = os.path.join(output_path, '%03d.png'%i )
        else:
            tmp_file_name = os.path.join(output_path, filenames[i] )
        scipy.misc.imsave(tmp_file_name, image)
        # upload the file into the girder item
        gc.uploadFileToItem(girder_item['_id'], tmp_file_name)


def get_collection_item_id_from_name(collection_name, item_name, gc=None):
    gc = get_gc(gc)
    resp = gc.get('collection?text=%s&limit=1&offset=0'%collection_name)
    assert len(resp) > 0, "%d collection not found"%collection_name
    collection_id = resp[0]['_id']
    return get_decendant_item_id(collection_id, 'collection', item_name, gc)


# give names of the collection and folder and return the first id found
def get_collection_folder_id_from_name(collection_name, folder_name, gc=None):
    gc = get_gc(gc)
    resp = gc.get('collection?text=%s&limit=1&offset=0'%collection_name)
    assert len(resp) > 0, "%s collection not found"%collection_name
    collection_id = resp[0]['_id']
    return get_decendant_folder_id(collection_id, 'collection', folder_name, gc)


# given a folder name and parent folder id return:
# [image_ids], folder_id
def get_image_ids_from_folder_name(folder_name, parent_folder_id, gc=None):
    gc = get_gc(gc)
    # get the folder id from its name.
    folder_id = get_decendant_folder_id(digital_globe_images_folder_id,
                                        'folder', folder_name, gc)
    assert folder_id, ("Could not find folder %s"%folder_name)
    return get_image_ids_from_folder_id(folder_id, gc), folder_id


def get_image_ids_from_folder_id(folder_id, gc=None):
    gc = get_gc(gc)
    # get a list of images in the folder
    image_ids = []
    resp = gc.get("item?folderId=%s&limit=1000&offset=0&sort=lowerName&sortdir=1"%folder_id)
    for item in resp:
        # todo: fix this bug.  Small images do not have this so are skipped.
        if "largeImage" in item:
            image_ids.append(item['_id'])
    return image_ids


# create a named annotation or return its id.
def create_or_find_annotation(image_id, name, gc=None):
    gc = get_gc(gc)
    annot_id = get_annotation_id_from_name(image_id, name, gc)
    if annot_id:
        return annot_id
    else:
        annot = {"elements":[],"name":name}
        resp = gc.post("annotation", parameters={"itemId":image_id}, json=annot)
        return resp["_id"]

# Get an annotation id from its name.
# returns None if doesnot exist for item
def get_annotation_id_from_name(image_id, name, gc):
    gc = get_gc(gc)
    resp = gc.get("annotation?itemId=%s&name=%s" % (image_id, name))
    if len(resp) > 0:
        #print('"%s":"%s"' % (image_id, resp[0]["_id"]))
        return resp[0]["_id"]
    else:
        return None



# returns the first decendant image with a matching name.
def get_decendant_item_id(ancestor_id, ancestor_type, item_name, gc=None):
    gc = get_gc(gc)
    if ancestor_type == 'folder':
        # look for the item
        resp = gc.get('item?folderId=%s&name=%s&limit=50'%(ancestor_id, item_name))
        if len(resp) > 0:
            return resp[0]['_id']

    resp = ['do while']
    offset = 0
    while len(resp) > 0:
        resp = gc.get('folder?parentType=%s&parentId=%s&limit=50&offset=%d'%(ancestor_type,ancestor_id,offset))
        offset += 50
        for folder in resp:
            item_id = get_decendant_item_id(folder['_id'], 'folder', item_name, gc)
            if item_id:
                return item_id
    return None


# returns the first decendant folder with a matching name.
def get_decendant_folder_id(ancestor_id, ancestor_type, folder_name, gc=None):
    gc = get_gc(gc)
    resp = ['do while']
    offset = 0
    while len(resp) > 0:
        resp = gc.get('folder?parentType=%s&parentId=%s&limit=50&offset=%d'%(ancestor_type,ancestor_id,offset))
        offset += 50
        for folder in resp:
            if folder['name'] == folder_name:
                return folder['_id']
            folder_id = get_decendant_folder_id(folder['_id'], 'folder', folder_name, gc)
            if folder_id:
                return folder_id
    return None

"""
def get_girder_cutout(gc, image_id, left, top, width, height, scale = 1):
    if scale == 1:
        chip_url = GIRDER_URL + "/api/v1/item/" + image_id + "/tiles/region?" + \
                   ("left=%d&top=%d&" % (left, top)) + \
                   ("regionWidth=%d&regionHeight=%d" % (width,height)) + \
                   "&units=base_pixels&encoding=JPEG&jpegQuality=95&jpegSubsampling=0"
    else:
        # does not shrink like I want.
        chip_url = GIRDER_URL + "/api/v1/item/" + image_id + "/tiles/region?" + \
                   ("maginfication=%f&" % (40 * scale)) + \
                   ("left=%d&top=%d&" % (left, top)) + \
                   ("regionWidth=%d&regionHeight=%d" % (width,height)) + \
                   "&units=base_pixels&encoding=JPEG&jpegQuality=95&jpegSubsampling=0"
                      

    #print(chip_url)
    req = urllib2.Request(chip_url)
    req.add_header('Girder-Token', gc.token)
    # intermitently does not work. Repeat and it does (not really)
    retry = 1
    while retry > 0:
        #print(chip_url)
        try:
            resp = urllib2.urlopen(req)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            #print('image loaded')
            return image
        except urllib2.HTTPError, err:
            if err.code == 400:
                print("Bad request!")
            elif err.code == 404:
                print("Page not found!")
            elif err.code == 403:
                print("Access denied!")
            else:
                print("Something happened! Error code %d" % err.code)
            retry -= 1
            #time.sleep(1)
    return None
 """

    
