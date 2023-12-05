import os
import argparse
import numpy as np
from datetime import date
import cv2
import openslide
import torch
import time
import shutil


class TileImgSaver:
    """ Saves tile images to disk. 
    
    Attributes: 
        wsi_dir: WSI directory
        data_dir: directory in which to save output files
        coord_file: tile coordinate file exist
        out_dir: output directory for tile images or embeddings
        head_node: directory to temporarily save files, for speed 
                (for example, on head node, ex: '/tmp/slides')
    
    Methods:
        parse_pt_file(): Parse PyTorch data file 
        save_tile_imgs(): Save tiles as image files

    Usage:
    >>> saver_obj = TileSaver(coord_file)
    >>> saver_obj.run()
    """
    def __init__(self, 
                 pt_file: str, 
                 tile_dir: str="", 
                 dbx: bool=False): 

        self.pt_file = pt_file
        self.tile_dir = tile_dir
        self.dbx = dbx
        self.head_node = "/tmp/slides" if dbx else ""
        
        self.slides = []
        self.labels = []
        self.tile_coords = []
        self.tile_size = 0
        self.tile_level = 0

    def run(self):
        self.make_dirs()
        self.parse_pt_file()
        self.save_tile_imgs(img_compression=0)
        return None
            
    def make_dirs(self):
        os.makedirs(self.tile_dir, exist_ok=True)
        if self.head_node != "": 
            os.makedirs(self.head_node, exist_ok=True)
        return None 
    
    def parse_pt_file(self):
        pt_dict = torch.load(self.coord_file)
        self.slides = pt_dict['slide_path']
        if 'tile_labels' in pt_dict.keys():
            self.labels = pt_dict['tile_labels']
        self.tile_coords = pt_dict['tile_coords']
        self.tile_size = pt_dict['tile_size']
        self.tile_level = pt_dict['tile_level']
        return None

    def save_tile_imgs(self, img_compression: int=0):
        
        # Open WSI as OpenSlide object
        for wsi_idx, wsi_path in enumerate(self.slides):
            #print(f"\nProcessing {wsi_path}\n")
            since = time.time()
            # Copy slide to head node, if using Dbx
            if self.head_node != "":
                shutil.copy2(wsi_path, self.head_node)
                wsi_path = os.path.join(self.head_node, os.path.basename(wsi_path))
            wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
            wsi_obj = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_path))
            
            # Make tiles for each WSI
            for tile_idx, tile_loc in enumerate(self.grid[wsi_idx]):
                # Store output file path
                outdir = self.head_node if self.head_node != "" else self.tile_dir
                filename = f"wsi_{wsi_name}"
                filename = filename + f"_loc_{tile_loc[0]}-{tile_loc[1]}"
                if len(self.labels) > 0:
                    filename = filename + f"_label_{self.labels[wsi_idx][tile_idx]}"
                outfile = os.path.join(outdir, filename + ".png")

                # Save tile image
                opens_img = wsi_obj.read_region((tile_loc[0], tile_loc[1]), 
                                                self.tile_level, 
                                                (self.tile_size, self.tile_size))
                opens_img = opens_img.convert("RGB")  # remove alpha channel
                # opens_img.save(outfile)  # save using PIL library
                tile_img = np.array(opens_img)
                cv2.imwrite(outfile, tile_img, [int(cv2.IMWRITE_PNG_COMPRESSION),
                                                img_compression])
                
                # transfer tile to ADLS, if necessary
                if self.head_node != "":
                    shutil.copy2(outfile, self.tile_dir)
                    os.remove(outfile)

            # Remove slide from head node, if using Dbx
            if self.head_node != "": os.remove(wsi_path)
            time_elapsed = time.time() - since
            print(f'\tSlide tiled in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        print(f"\nCompleted saving files.\n")
        return None
