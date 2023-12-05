import os
import numpy as np
import cv2
import openslide
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import time
import shutil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatExtractor:
    """ Extracts tile image feature vectors and saves to pytorch data file. 
    
    Attributes: 
        pt_file: path of PyTorch data file
        model: model class as used in torchvision.models (default = 'resnet50')
        weights: weights class as used in torchvision.models (default
                    = 'ResNet50_Weights.IMAGENET1K_V2')
        weights_file: use if in Databricks. You must download the weights 
                        file from the web and upload to Azure blob storage. 
        num_workers: number of workers (see pytorch doc)
        batch_size: batch size for feature generation
        dbx: 'True' if running on databricks (copies slide to head node at 
            '/tmp/slides/')

    Methods:
        extract_feat(pt_out): Converts all tiles from a WSI into a set of 
                        features and saves to pytorch file. 
                        pt_out: saves new pytorch data file, default is overwrite
        run(pt_out): runs all the above
                    pt_out: saves new pytorch data file, default is overwrite
        load_model(): loads model architecture and weights

    Usage:
        >>> extractor = FeatExtractor(PT_FILE)
        >>> extractor.run()
    """
    def __init__(self, 
                 pt_file: str, 
                 model: str, 
                 weights: str, 
                 weights_file: str, 
                 num_workers: str=1, 
                 batch_size: int=128, 
                 dbx: bool=False): 
        self.pt_file = pt_file
        self.model_name = model
        self.weights_name = weights
        self.model = eval(f'torchvision.models.{model}')
        self.weights = eval(f'torchvision.models.{weights}')
        self.transform = eval(f'torchvision.models.{weights}.transforms()')
        self.weights_file = weights_file
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dbx = dbx
        self.head_node = "/tmp/slides" if dbx else ""

        self.tile_coords = []
        self.tile_level = 0
        self.tile_size = 0
        
    def run(self, pt_out: str=""):
        if self.dbx: os.makedirs(self.head_node, exist_ok=True)
        if pt_out == "": pt_out = self.pt_file
        self.extract_feat(pt_out)
        return None
    
    def load_model(self):
        if self.weights_file == "":
            model = self.model(weights=self.weights)
        else: 
            model = self.model(weights=None)
            state_dict = torch.load(self.weights_file)
            model.load_state_dict(state_dict)
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
        for p in model.parameters(): p.requires_grad = False
        model.to(device)
        model.eval() 
        self.model = model
        return None

    def extract_feat(self, pt_out: str):
        # Load model
        self.load_model()
        model = self.model

        # Load data
        pt_dict = torch.load(self.pt_file)
        slide_path_list = pt_dict['slide_path']
        torch.set_grad_enabled(False) 
        pt_dict['feat_vectors'] = []
        for slide_idx, slide_path in enumerate(pt_dict['slide_path']):
            since = time.time()
            # Copy slide to head node, if using Dbx
            if self.dbx: 
                shutil.copy2(slide_path, self.head_node)
                tmp_slide_path = os.path.join(self.head_node, 
                                            os.path.basename(slide_path))
                pt_dict['slide_path'][slide_idx] = tmp_slide_path

            # Load slide
            dataset = PthFeatData(pt_dict=pt_dict, 
                                  slide_idx=slide_idx, 
                                  transform=self.transform)
            pin_memory = False if self.num_workers == 0 else True
            dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    pin_memory=pin_memory, 
                                    shuffle=False, 
                                    drop_last=False)

            # Run images through model to obtain list of probs
            total_n_tiles = len(pt_dict['tile_coords'][slide_idx])
            feat_list = []
            for step, images in enumerate(dataloader):
                images = images.to(device)
                outputs = model(images).cpu().numpy()
                feat_list.append(outputs)
            feat_array = np.array(feat_list, dtype=object)
            feat_np = np.concatenate(feat_array, axis=0, 
                                    dtype=object).astype(np.float32)
            feat_np = feat_np.reshape(total_n_tiles, 2048)

            # Save features to dictionary 
            pt_dict['feat_vectors'].append(feat_np.astype(np.float32))

            # Remove slide from head node, if using Dbx
            if self.dbx: os.remove(tmp_slide_path)

            # Print time info
            time_elapsed = time.time() - since
            print(f'\tGenerated tile feature vectors',
                    f'in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.', 
                    flush=True)
        
        # Save pth output file
        pt_dict['slide_path'] = slide_path_list
        pt_dict['feat_arch'] = self.model_name
        pt_dict['feat_weights'] = self.weights_name
        torch.save(pt_dict, pt_out)
        return None  
    

class PthFeatData(Dataset):    # <- this can probably be deleted
    """ Dataset for extracting tile image features (usually from ResNet50)

    Attributes:
        pt_dict: PyTorch data dictionary
        slide_idx: index of desired slide in pt_dict
        transform: PyTorch transform
    """
    def __init__(self, 
                 pt_dict: str, 
                 slide_idx: int, 
                 transform: None):
        self.pt_dict = pt_dict
        self.slide_idx = slide_idx
        self.transform = transform
        
        self.wsi_path = pt_dict['slide_path'][slide_idx]
        self.wsi_obj = openslide.OpenSlide(self.wsi_path)
        self.tile_coords = pt_dict['tile_coords'][slide_idx]
        self.level = pt_dict['tile_level']
        self.tile_size = pt_dict['tile_size']


    def __len__(self):
        """ Returns number of observations (images/labels) """
        return len(self.tile_coords)

    def __getitem__(self, idx):
        """ Returns an image for a given index number. """
        # return image as numpy array
        x, y = self.tile_coords[idx]
        if x == 0 and y == 0:
            image = np.zeros([self.tile_size, self.tile_size, 3], 
                                dtype=np.uint8)
        else: 
            image = self.wsi_obj.read_region((x, y), self.level, 
                                                (self.tile_size, self.tile_size))
            image = image.convert("RGB") # removing alpha channel

        # transform image 
        image = self.transform(image)
        return image
