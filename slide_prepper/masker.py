"""
For installation of Segment Anything, see: https://github.com/facebookresearch/segment-anything.git
"""
import time
import os
import cv2
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import openslide
import torch


class Masker:
    """ 
    Generates a tissue mask from a WSI.

    Attributes: 
        wsi_df: dataframe containing 'slide_path' in which all files will be masked
        wsi_dir: directory containing all files needing masked (wsi_df will be ignored)
        mask_dir: directory in which to save output
        skip_completed: 'True' will skip mask files if they exist already
        method: mask method (Options are either 'HSV' or 'SAM')
        sam_model_type: type of SAM model (default is vit_h)
        sam_checkpoint: PyTorch file containing SAM model weights

    Methods:
        get_wsi_list(): makes list of wsi files from wsi_dir or wsi_df
        setup_sam(): sets up SAM parameters and model weights
        make_wsi_mask(): saves mask to file
        run(): runs all of the above

    Usage:
    >>> mask_maker = Masker(wsi_path, mask_dir)
    >>> mask_maker.run()
    """
    def __init__(self, 
                 wsi_df: str="", 
                 wsi_dir: str="", 
                 mask_dir: str="", 
                 skip_completed: bool=True,
                 method: str="SAM",
                 sam_model_type: str='vit_h',
                 sam_checkpoint: str='sam_vit_h_4b8939.pth'): 
        self.wsi_df = wsi_df
        self.wsi_dir = wsi_dir
        self.mask_dir = mask_dir
        self.skip_completed = skip_completed
        self.method = method
        self.sam_model_type = sam_model_type
        self.sam_checkpoint = sam_checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.png_compression = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]

        self.wsi_path_list = []
        self.wsi_obj_list = []
        self.wsi_np = np.array([])
        self.wsi_mask = np.array([])

    def run(self):
        self.get_wsi_list()
        os.makedirs(self.mask_dir, exist_ok=True)
        if self.method == "SAM": self.setup_sam()
        for i in range(len(self.wsi_path_list)): self.make_wsi_mask(i)
        return None
    
    def get_wsi_list(self):
        if self.wsi_dir != "":
            for f in os.listdir(self.wsi_dir):
                    if f.endswith((".ndpi", ".svs", ".tif", ".tiff")):
                        self.wsi_path_list.append(os.path.join(self.wsi_dir, f))
        else: 
            self.wsi_path_list = self.wsi_df['slide_path'].tolist()
        return None
    
    def setup_sam(self):
        # Specify model weight file
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        # Create mask generator
        self.mask_generator_ = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.96,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    
    def make_wsi_mask(self, slide_idx):
        """ Produces a downsampled tissue mask of the WSI using SAM. """
        start_time = time.time()  # start timer
        
        # Load downsampled slide as numpy array
        wsi_path = self.wsi_path_list[slide_idx]
        wsi_obj = openslide.OpenSlide(wsi_path)
        mask_level = wsi_obj.get_best_level_for_downsample(64)
        size_dim = wsi_obj.level_dimensions[mask_level]
        wsi_pil = wsi_obj.get_thumbnail(size_dim)
        wsi_np = np.array(wsi_pil)

        # Turn into black & white mask using HSV
        if self.method == "HSV":
            # Blur image
            img = cv2.blur(np.array(wsi_np), (30,30))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Threshold image and return mask
            lower_thresh = np.array([75, 20, 75], dtype=np.uint8)
            upper_thresh = np.array([170, 175, 255], dtype=np.uint8)
            final_mask_np = cv2.inRange(img, lower_thresh, upper_thresh)
        # Turn into black & white mask using SAM
        else: 
            mask_dict = self.mask_generator_.generate(wsi_np)
            final_mask_bw = np.zeros(mask_dict[0]['segmentation'].shape)
            mask_height = final_mask_bw.shape[0]
            for m in mask_dict:
                mask_bool = m['segmentation']
                mask_bw = mask_bool.astype(np.uint8) * 255
                if (mask_bw[10,10] == 0) and (mask_bw[mask_height-10,10] == 0):
                    final_mask_bw += mask_bw
            final_mask_bw[final_mask_bw > 255] = 255
            final_mask_np = final_mask_bw

        seg_time_elapsed = time.time() - start_time   
        print("".join([f"\tMask generated in ", 
                        f"{seg_time_elapsed // 60:.0f}m ",
                        f"{seg_time_elapsed % 60:.0f}s "]), flush=True)

        # Save mask to file.
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        outfile = os.path.join(self.mask_dir, wsi_name + "_mask.png")
        cv2.imwrite(outfile, final_mask_np, self.png_compression)
        return None