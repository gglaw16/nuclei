from __future__ import print_function
import nuclei.girder as g
import pdb

# load all the meta data from a czi file on girder
class time_lapse_data():

    def __init__(self):
        self.series_folders = []
        self.series = []
        
    def load(self, girder_folder_id, gc=None):
        gc = g.get_gc(gc)
        resp = gc.get('folder?parentType=folder&parentId=%s&limit=100&sort=lowerName&sortdir=1'%girder_folder_id)
        for folder in resp:
            if 'series' in folder['name']:
                self.series_folders.append(folder)
                self.load_series(folder['_id'], gc)

    def load_series(self, girder_folder_id, gc=None):
        gc = g.get_gc(gc)
        resp = gc.get('item?folderId=%s&limit=100&sort=lowerName&sortdir=1'%girder_folder_id)
        series = []
        for item in resp:
            if 'time' in item['name']:
                series.append(item)
        self.series.append(series)
                
    def get_number_of_series(self):
        return len(self.series)

    def get_series_length(self, series_idx):
        if series_idx < 0 or series_idx >= len(self.series):
            return 0
        return len(self.series[series_idx])

    def get_series_folder(self, series_idx):
        if series_idx < 0 or series_idx >= len(self.series_folders):
            return None
        return self.series_folders[series_idx]
    
    def get_image(self, series_idx, time_idx, gc=None):
        gc = g.get_gc(gc)
        if series_idx < 0 or series_idx >= len(self.series):
            return None
        series = self.series[series_idx]
        if time_idx < 0 or time_idx >= len(series):
            return None
        image, img_obj = g.read_item_image(series[time_idx]['_id'], gc)
        return image, img_obj

    def get_series_stack(self, series_idx, gc=None):
        gc = g.get_gc(gc);
        folder = self.get_series_folder(series_idx)
        resp = gc.get('item?folderId=%s&name=stack'%folder['_id'])
        if len(resp) == 0:
            return None
        return resp[0]
    
if __name__ == "__main__":
    # test the class.
    source = time_lapse_data()
    pdb.set_trace()
    source.load("5aaf02831fbb9006233ae6a2")
    print("series: %d"%source.get_number_of_series())
    print("time steps in series 1: %d"%source.get_series_length(1))
    img, _ = source.get_image(1,10)
    print(img.shape)
