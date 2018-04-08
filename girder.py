import numpy as np
import scipy
import scipy.misc
import cv2
import os
import sys
import girder_client
from key import *
import pdb



def get_gc(gc=None):
    if gc is None:
        gc = girder_client.GirderClient(apiUrl= GIRDER_URL+'/api/v1')
        gc.authenticate(GIRDER_USERNAME, apiKey=GIRDER_KEY)
    return gc


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


# Batch is numpy (batch, comps, dimy, dimx)
# Upload the images to girder (for debugging)
# Crude, but at least the chips do not change when the annotation changes.
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
 

    
