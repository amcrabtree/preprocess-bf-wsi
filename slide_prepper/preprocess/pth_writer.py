import os
import pandas as pd
import torch

class SlideLevelPthWriter:
    """ 
    Writes a slide-level pth data file to use in ML workflows.

    Attributes: 
        wsi_df: 
        wsi_dir: 
        pt_file: PyTorch file containing data for slide-level labels
        pt_dict: slide-level data dictionary

    Methods:
        update_df(): subsets provided wsi_df by slides in wsi_dir
        make_dict(): makes data dictionary given slide info
        write_file(): writes data dictionary to PyTorch file

    Usage:
        >>> pt_writer = SlideLevelPthWriter(pt_filename)
        >>> pt_writer.write_file()
    """
    def __init__(self, 
                 wsi_df: pd.DataFrame=pd.DataFrame(), 
                 wsi_dir: str=""):
        self.wsi_df = wsi_df
        self.wsi_dir = wsi_dir
        self.pt_file = ""
        self.pt_dict = {}

    def update_df(self, subset_by_dir: str=""):
        if subset_by_dir != "":
            self.wsi_dir = subset_by_dir
        # Get list of WSI files in wsi_dir, if given
        if self.wsi_dir != "":
            wsi_file_list = []
            for f in os.listdir(self.wsi_dir):
                if f.endswith((".ndpi", ".svs", ".tif", ".tiff")):
                    wsi_file_list.append(os.path.join(self.wsi_dir, f))
            # Make new wsi_df if wsi_df is empty
            if len(self.wsi_df) == 0:
                for wsi_path in wsi_file_list: 
                    self.wsi_df['slide_path'] = wsi_path
            # Inner join if wsi_df is not empty
            else: 
                self.wsi_df = self.wsi_df[self.wsi_df['slide_path'].isin(wsi_file_list)]
        return None

    def make_dict(self):
        # Add slide paths to dict
        self.pt_dict['slide_path'] = self.wsi_df['slide_path'].values.tolist()
        # Add labels to dict, if they exist
        colnames = self.wsi_df.columns.tolist()
        col_drop_list = ['slide_path']
        if len(colnames) > len(col_drop_list):
            label_df = self.wsi_df.drop(columns=col_drop_list)
            label_list = []
            for row in range(len(label_df)):
                label_dict = {}
                colnames = [ele for ele in colnames if ele not in col_drop_list]
                for col in colnames:
                    label_dict[col] = label_df[col].iloc[row]
                label_list.append(label_dict)
            self.pt_dict['labels'] = label_list
        return None

    def write_file(self, pt_filename: str="train_lib.pth"):
        self.update_df()
        self.make_dict()
        torch.save(self.pt_dict, pt_filename)
        return None
