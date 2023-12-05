# Slide Prepper
*Preprocessing tools for bright-field Whole Slide Images (WSIs), such as H&amp;E-stained slides*

The process of preparing digitized slide images for machine learning operations is anything but trivial. In this repo, we have contained scripts for the following operations: 

- Writing, reading, and modifying PyTorch data files
- Generating tissue mask PNG images
- Generating tile coordinates
- Extracting feature vectors for each tile using classic CNNs
- Saving tiles as PNG images 

#### 1. Write PyTorch (.pth) file from list of slides

The list of slides can either be in the form of a Pandas dataframe (eg from a sample csv file) or from a directory containing slide images. If using a directory path, the images cannot be contained within subfolders and must have one of the following extensions: .tif, .tiff, .svg, .ndpi

```python
from slide_prepper.pth_writer import SlideLevelPthWriter

# If using a sample sheet:
wsi_df = pd.read_csv(SLIDE_CSV)
pt_writer = SlideLevelPthWriter(wsi_df=wsi_df)
pt_writer.write_file("output_file.pth")

# Alternative, if using a slide directory:
pt_writer = SlideLevelPthWriter(wsi_dir=WSI_DIR)
pt_writer.write_file("data_file.pth")
```

#### Inspect PyTorch slide-level data file

It is a good idea to check the data file to ensure it has been written as desired. The SlideLevelPthReader class enables easy printing of slide information when provided a slide index. The slide index is the position the slide is in the original slide dataframe or directory. 

```python
from slide_prepper.pth_reader import SlideLevelPthReader

pt_reader = SlideLevelPthReader("data_file.pth")
print(pt_reader.pull_slide_info(slide_idx=2))
```

#### 2. Segment slide background from tissue-containing areas

For each slide, a mask of tissue areas is created, which is used in determining tile coordinates during downstream data preparation. These masks are scaled down, ideally to 1/64th the size of the original slide image, but less downscaling is performed if level is not available. 

```python
from slide_prepper.masker import Masker

# HSV thresholding
masker = Masker(wsi_dir=WSI_DIR, 
                mask_dir=MASK_DIR, 
                skip_completed=True, 
                method='HSV')
masker.run()

# Alternatively, SAM thresholding (problematic for TCGA slides because of image artifacts)
masker = Masker(wsi_dir=WSI_DIR, 
                mask_dir=MASK_DIR, 
                skip_completed=True, 
                method='SAM', 
                sam_model_type='vit_h',
                sam_checkpoint=SAM_WEIGHTS_FILE)
masker.run()
```

#### 3. Add tile coordinates to pt data file

Now that masks are available, tile coordinates can be appended to the PyTorch data file. Note that if the masks are not available or the path to the mask directory is incorrect, all tiles in the image will be saved. However, there will be a warning printed for any slides that do not have a mask file available. 

```python
from slide_prepper.coord_generator import CoordGenerator

coord_maker = CoordGenerator(pt_file="data_file.pth", 
                             mask_dir=MASK_DIR, 
                             tile_size=256)
coord_maker.run()

# Ensure tile coordinates are available in data file (print first 10 coordinates):
print(pt_reader.pull_slide_info(slide_idx=2)['tile_coords'][0:10])
```
#### 4. Extract image features for each tile 

Runs each tile image through a standard CNN to create a feature vector for each tile. See PyTorch documentation on available [models](https://pytorch.org/vision/stable/models.html). 

```python
from slide_prepper.feat_extract import FeatExtractor

extractor = FeatExtractor(pt_file="data_file.pth", 
                          model='resnet50',
                          weights='ResNet50_Weights.IMAGENET1K_V2')
extractor.run(pt_out="data_feat_file.pth")

# Alternatively, you can overwrite your old data file by excluding arguments from the run method:
extractor.run()
```

#### 5. Divide dataset between train/val/test and make folds

Creates a data sheet defining which slides are in training, validation, and test sets in each fold. This can be updated manually and used in downstream training. 

```python
from slide_prepper.split_folds import DataSplitter

splitter = DataSplitter("data_file.pth", n_folds=1)
splitter.run()
splitter.write_file("data_splits.csv")
```

#### Optional: Save tile image files (WARNING: needs debugging)

This can be used on any PyTorch file that contains tile coordinates. The tile image names will only contain labels if the 'tile_labels' key is present in the PyTorch data file. 

```python
from slide_prepper.save_tile_imgs import TileImgSaver

img_saver = TileSaver(pt_file=PT_FILE, 
                      tile_dir=TILE_DIR)
img_saver.run()
```

