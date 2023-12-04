import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class DataSplitter:
    """ Generates tile coordinates from a WSI.

    Attributes: 
        pt_file: PyTorch file containing data
        n_folds: number of data folds to produce
        split_df: dataframe containing split info

    Methods:
        split_and_fold(): Divide dataset folds between train/val/test 
        run(): runs all of the above
        write_file(csv_file): writes CSV file containing split info

    Usage:
        >>> splitter = DataSplitter(pt_file, n_folds=5)
        >>> splitter.run()
        >>> splitter.write_file(csv_file)
    """
    def __init__(self, pt_file: str, n_folds: int=1): 
        self.pt_file = pt_file
        self.n_folds = n_folds
        self.split_df = pd.DataFrame()
    
    def run(self):
        self.split_and_fold()

    def split_and_fold(self) -> pd.DataFrame:
        slides_list = torch.load(self.pt_file)['slide_path']
        slides_list = [os.path.basename(x) for x in slides_list]
        # Split datasets 
        dict_list = []
        for n in range(self.n_folds):
            idx_list = [i for i in range(len(slides_list))]
            X_train, X_test, y_train, y_test = train_test_split(idx_list, slides_list, 
                                                                test_size=0.4)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
                                                            test_size=0.5)
            for idx in X_train: 
                dict_list.append({'fold':n+1, 'subset':'train', 'slide_idx':idx, 
                                'slide_file':slides_list[idx]})
            for idx in X_val: 
                dict_list.append({'fold':n+1, 'subset':'val', 'slide_idx':idx, 
                                'slide_file':slides_list[idx]})
            for idx in X_test: 
                dict_list.append({'fold':n+1, 'subset':'test', 'slide_idx':idx, 
                                'slide_file':slides_list[idx]})
        # Load into pandas dataframe 
        self.split_df = pd.DataFrame(dict_list)
        return self.split_df
    
    def write_file(self, csv_file: str=""): 
        if csv_file != "": self.csv_file = csv_file
        self.split_df.to_csv(self.csv_file, index=False)
