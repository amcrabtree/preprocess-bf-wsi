import numpy as np
import openslide
import cv2

class BgSegmentor:
    """ 
    Segments tissue from slide background in a WSI. 

    Attributes: 
        wsi_file: pathname of WSI file

    Methods:
        make_mask(): Produces a downsampled mask of the WSI.

    Usage:
        >>> segmentor = BgSegmentor(wsi_file)
        >>> mask = segmentor.make_mask(method="HSV")
    """
    def __init__(self, wsi_file: str): 
        self.wsi_file = wsi_file
        self.wsi_obj = openslide.OpenSlide(wsi_file)
        self.mask_level = self.wsi_obj.get_best_level_for_downsample(64)

    def make_mask(self, method: str="HSV") -> np.ndarray:
        """ 
	Produces a downsampled tissue mask of the WSI. 
        """
        # Downsample image
        mask_dim = self.wsi_obj.level_dimensions[self.mask_level]
        ds_wsi = self.wsi_obj.read_region((0, 0), self.mask_level, mask_dim)
        ds_wsi = np.array(ds_wsi.convert("RGB"))

	if method == "HSV":
	    # Blur image
	    img = cv2.blur(ds_wsi, (30,30))
	    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	
	    # Threshold image and return mask
	    lower_thresh = np.array([75, 20, 75], dtype=np.uint8)
	    upper_thresh = np.array([170, 175, 255], dtype=np.uint8)
	    wsi_mask = cv2.inRange(img, lower_thresh, upper_thresh)
	else: 
	    print("ERROR: Mask method argument is invalid.")
	    exit(1)
        return wsi_mask
