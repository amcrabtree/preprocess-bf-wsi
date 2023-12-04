import time
import os
import cv2
import openslide
import torch
import glob
import numpy as np


class CoordGenerator:
    """ 
    Generates tile coordinates for a WSI, removing background slides 
    if a mask file is provided.

    Attributes: 
        pt_file: PyTorch data file containing slide info
        out_pt_file: PyTorch data file to save output (default overwrites pt_file) 
        mask_dir: directory in which to save output
        tile_size: pixel size of tiles (can only be square tiles)
        tile_level: magnification level, default is 0 (highest mag)
        mask_thresh: minimum ratio of white pixels a mask tile needs to be saved
        pt_dict: PyTorch data dictionary
        mask_path_list: list of paths to mask files corresponding to slides
        tile_coords: nested list of non-blank tile coords for each slide

    Methods:
        find_mask_files(): makes list of wsi files from pt_file
        get_tile_coords(): saves mask to file
        write_file(): saves mask to file
        run(): runs all of the above

    Usage:
        >>> coord_maker = CoordGenerator(pt_file, mask_dir, tile_size=256)
        >>> coord_maker.run()
        >>> coord_maker.write_file(pt_file)
    """
    def __init__(self, 
                 pt_file: str, 
                 out_pt_file: str="",
                 mask_dir: str="", 
                 tile_size: int=256, 
                 tile_level: int=0):
        self.pt_file = pt_file
        self.out_pt_file = out_pt_file if out_pt_file != "" else pt_file
        self.mask_dir = mask_dir
        self.tile_size = tile_size
        self.tile_level = tile_level
        self.mask_thresh = 0.2  # if > 20% of the tile contains tissue, save tile
        self.pt_dict = ""
        self.mask_path_list = []
        self.tile_coords = []

    def run(self):
        self.pt_dict = torch.load(self.pt_file)
        self.find_mask_files()
        self.get_tile_coords()
        self.write_file()

    def find_mask_files(self):
        """ Make list of mask file paths. """
        for wsi_path in self.pt_dict['slide_path']: 
            wsi_file = os.path.basename(wsi_path)
            wsi_name = wsi_file.split(".")[0]
            match_list = glob.glob(f"{self.mask_dir}/*{wsi_name}*.png")
            if len(match_list) == 0:
                print(f"WARNING: Mask file for slide does not exist: {wsi_file}")
                self.mask_path_list.append("")
            else:
                mask_file = match_list[0]
                mask_path = os.path.join(self.mask_dir, mask_file)
                self.mask_path_list.append(mask_path)
        return None
    
    def get_tile_coords(self):
        self.pt_dict['tile_size'] = self.tile_size
        self.pt_dict['tile_coords'] = []
        for i, slide_path in enumerate(self.pt_dict['slide_path']):
            wsi_obj = openslide.OpenSlide(slide_path)
            wsi_w, wsi_h = wsi_obj.level_dimensions[self.tile_level] 
            mask_file = self.mask_path_list[i]
            if mask_file != "":
                small_mask = cv2.imread(mask_file)
                wsi_mask = cv2.resize(small_mask, (wsi_w, wsi_h))
            else: 
                wsi_mask = np.full((wsi_h, wsi_w), 255, dtype='uint8')
            
            # Generate tile coords
            wsi_tile_list = []
            if self.tile_level == 0: 
                t_size_lev0 = self.tile_size
            else:
                t_size_lev0 = round(size / wsi_obj.level_downsamples[new_level])
            x_range, y_range = [0, wsi_w], [0, wsi_h]
            for x in range(x_range[0], x_range[1]-t_size_lev0, t_size_lev0):
                for y in range(y_range[0], y_range[1]-t_size_lev0, t_size_lev0):
                    # Calculate masked ratio (white) and compare to threshold
                    masked_tile = wsi_mask[y:(y + t_size_lev0), x:(x + t_size_lev0)]
                    tile_mask_ratio = np.count_nonzero(masked_tile) / masked_tile.size
                    if tile_mask_ratio > self.mask_thresh: wsi_tile_list.append((x,y))
            self.pt_dict['tile_coords'].append(wsi_tile_list)
        return None

    def write_file(self, pt_file: str=""):
        if pt_file != "": self.out_pt_file = pt_file
        torch.save(self.pt_dict, self.out_pt_file)
        return None
