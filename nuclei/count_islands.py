# Algorithm for finding peaks / islands
# It is less sensitve to spatial constraints or intensity contraints.
# It seeds segmetnation with local maximum.  It then filters these based on
# bunch of criteria.
# Each local maxima peak is through of as a tim of an island.
# This algorithm is an iterative process of createing islands from peaks.
# this highest peak is turned into an island by thresholding to give an
# island of height "island_height".
# The prime peak, and any other peak part of the island are removed.
# If The island generated bleeds over to a higher peak, no island is saved.

import numpy as np
import cv2
import scipy.signal
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


# min value is shifted to 0.  All values belop min are clamped to zero.
def clip(img, min_value):
    # change this to isolate the spots after filtering
    out = np.clip(img, min_value, 255) - min_value
    return out

def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    All non peak pixels are set to zero.
    The result is returned. (each non zero pixel is a peak or tied for a peak).
    Peak pixels retain their input value.
    """
    peaks = np.array(image)
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)
    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    peaks[maximum_filter(peaks, footprint=neighborhood) != peaks] = 0
    return peaks

def fill_max_island(image, peaks, islands, island_height, area_max):
    """
    this is gauranteed to find an island or return false.
    Islands and peaks are modified (largest peak is moved to islands)
    Returns true if there are more peaks to process.
    image: the image used to find peaks.
    peaks: Same as image, but non maxima pixels are set to 0.
    island_heights: pass in islands found so far. New island added to it.
      call by reference return value.
    area_max:  Wide islands are filtered out. Only pointing islands are kept.
    """
    while True:
        # find the location and value of the highest peak
        ix = np.argmax(np.max(peaks, axis=0))
        iy = np.argmax(peaks[:,ix])
        peak_height = peaks[iy, ix]
        if peak_height == 0:
            # No more peaks in the peaks image.
            return False
        # Threshold at the waterline
        ret,mask = cv2.threshold(image, peak_height-island_height, 255, cv2.THRESH_BINARY_INV)
        # Use floodfill/connectivity to get rid of all islands not containing the main peak.
        # mask for flood fill has to be padded with 1 layers.
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 1).astype(np.uint8)
        island = np.zeros(image.shape, dtype=np.uint8)
        cv2.floodFill(island, mask, (ix, iy), 1)
        # only keep this island if it does not bleed over into a higher island.
        if np.max(image*island) <= peak_height:
            # compute the island area (pixels have value 1).
            # Even diffuse blobs will have one peak.  This gets rid of peaks that are too smooth.
            # Area is a measure of sharpness. Even a small island can be part of a large
            # mostly submerged mountain.
            island_area = np.sum(island)
            if island_area < area_max:
                # This island is a keeper.
                # Add the island to the output map.
                # I had a hard time with inplace addition.
                islands[...] = islands + (island*peak_height)
                # remove the peak (and any lower peaks also part of the island) from the peak map.
                peaks[island != 0] = 0
                return True
        # This island did not pass the test.  Just remove the peak from the peaks iamge.
        peaks[island != 0] = 0
        # keep looping until an island passes, or no peaks are left.    


def peaks_to_islands(image, peaks, island_height, area_max):
    # image is the original image,  c is the image where every non local maximum is set to 0.
    # Flood fill is used to grow peaks into islands of height c. Any peaks that are not
    # separated by a valley of depth c are merged.
    # reuturns (island_map, island_count)
    islands = np.zeros(peaks.shape, dtype=np.uint8)
    count = 0
    # keep generating island, peak by peak.
    while fill_max_island(image, peaks, islands, island_height, area_max):
        count = count + 1
        # This should not be necessary, but infinite loops are painful.
        if count > 100:
            return islands, count

    return islands, count


def count_islands(image, peak_absolute_min, spacing_min, island_height, area_max):
    clipped_image = clip(image, peak_absolute_min)
    ksize = (2*spacing_min) - 1
    #print((ksize,ksize), clipped_image.shape, clipped_image.dtype)
    median_image = scipy.signal.medfilt(clipped_image, kernel_size=(ksize,ksize))
    # Detects every local maximum. They need to be filtered to give anything useful.
    peaks = detect_peaks(median_image)
    return peaks_to_islands(median_image, peaks, island_height, area_max)


        
