import torch

class SlideLevelPthReader:
    """ 
    Reads a slide-level pth data file to aid data checking.

    Attributes: 
        pt_file: PyTorch file containing data for slide-level labels
        pt_dict: slide-level data dictionary
        dict_keys: all keys in PyTorch data file or dictionary

    Methods:
        test_input(): ensures at least one of the two possible inputs are provided
        read_file(): reads in the pytorch data file
        pull_slide_info(): gathers slide info for the provided slide index

    Usage:
        >>> pt_reader = SlideLevelPthReader(pt_filename)
        >>> pt_reader.read_file()
        >>> slide_2_info = pt_reader.pull_slide_info(slide_idx=2)
        >>> print(slide_2_info['slide_path'])
    """
    def __init__(self, 
                 pt_file: str="", 
                 pt_dict: dict={}):
        self.pt_file = pt_file
        self.pt_dict = pt_dict
        self.dict_keys = []
        self.test_input()

    def test_input(self):
        if (len(self.pt_dict) == 0) and (self.pt_file == ""):
            print("ERROR: load SlideLevelPthReader with pt_file or pt_dict")
            exit()
        return None

    def read_file(self):
        self.pt_dict = torch.load(self.pt_file)
        self.dict_keys = self.pt_dict.keys()
        return None

    def pull_slide_info(self, slide_idx: int):
        if len(self.pt_dict) == 0: self.read_file()
        slide_dict = {'slide_path': self.pt_dict['slide_path'][slide_idx]}
        if 'labels' in self.dict_keys:
            slide_dict['labels'] = self.pt_dict['labels'][slide_idx]
        if 'tile_size' in self.dict_keys:
            slide_dict['tile_size'] = self.pt_dict['tile_size']
            slide_dict['tile_coords'] = self.pt_dict['tile_coords'][slide_idx]
        if 'ae_arch' in self.dict_keys:
            slide_dict['ae_arch'] = self.pt_dict['ae_arch']
            slide_dict['ae_file'] = self.pt_dict['ae_file']
            slide_dict['ae_features'] = self.pt_dict['ae_features'][slide_idx]
        return slide_dict
