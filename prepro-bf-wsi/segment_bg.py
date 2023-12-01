import os
import argparse
import numpy as np
import openslide
import cv2
import glob
import time
from datetime import date
from multiprocessing import Pool


class TissueSegmentor:
    """ Generates a tissue mask from a WSI.

    Attributes: 
        wsi_file: pathname of WSI file
        thresh: ratio of mask in tile above which the tile is "masked" 
            (default is 0.20)
        mask_level: magnification level to create mask (0 is high mag.)
        data_dir: directory in which to save output

    Methods:
        get_tile_coords(): Generates array of (top left) tile 
            coordinates for a WSI (at level 0). 
            Ex: [ [x0,y0], [x1,y1]...[xN,yN] ]
        apply_mask(): Retrieves a mask of the tile area.
        get_tile_labels(): Determine tile-level labels from annotations. 
        save_tiles(): Saves tiles according to output method. 
        get_tile_img(): Extracts a numpy array of a tile image.
        relevel(): Converts coordinates or tile size from one 
            magnification level to another.
        get_wsi_mask(): Produces a downsampled mask of the WSI.
        is_masked(): Determines whether image passes masking threshold.

    Usage:
    >>> gen_obj = TissueSegmentor(wsi_file)
    >>> gen_obj.run()
    """
    def __init__(self, wsi_file: str, save_thumb: bool=False, 
                data_dir: str=""): 

        self.wsi_file = wsi_file
        self.save_thumb = save_thumb
        self.data_dir = data_dir

        self.wsi_obj = openslide.OpenSlide(wsi_file)
        self.wsi_mask = np.ndarray=np.array([])
        self.mask_level = self.wsi_obj.get_best_level_for_downsample(64)
    

    def run(self):
        print(f"\nProcessing WSI: {self.wsi_file}")
        if self.save_thumb: self.save_thumb_img()
        self.make_wsi_mask()
        self.save_wsi_mask()


    def save_thumb_img(self):
        """ Save WSI thumbnail image to file. """
        out_path = os.path.join(self.data_dir, "thumbnails")
        wsi_name = os.path.basename(self.wsi_file).split(".")[0]
        outfile = os.path.join(out_path, wsi_name + "_thumb.png")
        os.makedirs(out_path, exist_ok=True)
        size_dim = np.array(self.wsi_obj.dimensions)//50
        wsi_pil = self.wsi_obj.get_thumbnail(size_dim)
        wsi_pil.save(outfile)
        print(f"\tSaved thumbnail to file: {outfile}")
        return None


    def make_wsi_mask(self) -> np.ndarray:
        """ Produces a downsampled tissue mask of the WSI. """
        start_time = time.time()  # start timer
        # Downsample image
        mask_dim = self.wsi_obj.level_dimensions[self.mask_level]
        ds_wsi = self.wsi_obj.read_region((0, 0), self.mask_level, mask_dim)
        ds_wsi = np.array(ds_wsi.convert("RGB"))

        # Blur image
        img = cv2.blur(ds_wsi, (30,30))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Threshold image and return mask
        lower_thresh = np.array([75, 20, 75], dtype=np.uint8)
        upper_thresh = np.array([170, 175, 255], dtype=np.uint8)
        self.wsi_mask = cv2.inRange(img, lower_thresh, upper_thresh)

        seg_time_elapsed = time.time() - start_time   
        print("".join([f"\tMask generated in ", 
                        f"{seg_time_elapsed // 60:.0f}m ",
                        f"{seg_time_elapsed % 60:.0f}s "]))
        return None
    

    def save_wsi_mask(self):
        """ Save mask to file. """
        out_path = os.path.join(self.data_dir, "masks")
        wsi_name = os.path.basename(self.wsi_file).split(".")[0]
        outfile = os.path.join(out_path, wsi_name + "_mask.png")
        os.makedirs(out_path, exist_ok=True)
        cv2.imwrite(outfile, self.wsi_mask, 
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        print(f"\tSaved mask to file: {outfile}")
        return None


#Multi-processing function
def run_wsi_file(wsi_file):
    gen_obj = TissueSegmentor(wsi_file, 
                              save_thumb=args.save_thumb,
                              data_dir=args.data_dir)
    gen_obj.run()

    print(f"\tTissue segmentation completed.")

# Argparse
parser = argparse.ArgumentParser(description='Generate tissue segmentation masks')
parser.add_argument('--data_dir', type=str, required=True, 
					help='Directory containing dataset subdirectories, 1 up from "slides"')
parser.add_argument('--save_thumb', default=False, action='store_true',
					help='Save wsi thumbnails as png files')
parser.add_argument('--skip_completed', default=False, action='store_true',
					help='Skip processing wsi files with saved masks')
parser.add_argument('--cores', default=1, type=int,
					help='Number of cores to use durning processing')


if __name__ == "__main__":
    args = parser.parse_args()
    p=Pool(args.cores)
    # Get list of WSI files
    wsi_file_list = []
    wsi_dir = os.path.join(args.data_dir, "slides")
    for f in os.listdir(wsi_dir):
        if f.endswith((".ndpi", ".svs", ".tif", ".tiff")):
            wsi_name = os.path.basename(f).split(".")[0]
            mask_dir = os.path.join(args.data_dir, "masks")
            outfile = os.path.join(mask_dir, wsi_name+"_mask.png")
            if args.skip_completed and os.path.exists(outfile): # don't run processed WSIs 
                continue
            else:
                wsi_file_list.append(os.path.join(wsi_dir, f))

    # Run tile generator
    p.map(run_wsi_file, wsi_file_list)
