# Slide Prepper
*Preprocessing tools for bright-field Whole Slide Images (WSIs), such as H&amp;E-stained slides*

The process of preparing digitized slide images for machine learning operations is anything but trivial. In this repo, we have contained scripts for the following operations: 

- **segment_bg**: Separates tissue-containing areas from blank areas of the slide
- **make_coords**: Generates tile coordinates which are saved as dictionary-like PyTorch files
- **save_imgs**: Generates tiles as PNG images
