# Annotation Procedure

### Runing the script
- Place .svs files to be processed into `dataset/data/final`
- Execute `src/features/extract_roi.py` with Python 3 (check that downscale=4)
- Find the resultant slide-wise segregated ROIs in `dataset/data/roi`

### Verification, Labeling and Data Entry

- Segregate the image files into 4 categories, referring to the mapping spreadsheet:
    - `P`: Positive
    - `N`: Negative
    - `E`: Extra, Irrelevant
    - `NAR`: Valid ROI, but Not Relevant for Analysis 
- Update the mapping spreadsheet column - `filepath` with the file path of the ROI image

       
#### **If Multiple ROIs are extracted into the same image**

Assuming the image file is names: `file-region_4.tiff`,
1. Number each ROI within the image.    
    **Precedence**: Vertical Position (Top, then bottom) (then) Horizontal Position (Left, then Right)
    (i.e) complete all ROIs at a particular height - left to right, then move to the next height level
2. Crop out each ROI into a separate file
3. Name them as `file-region_4_x.tiff` where `x` is the ROI number, assigned to it in step-1.
4. Place the cropped out ROIs into the correspodning category.
5. Place the original image (containing multiple ROIs) into the `E` category folder.

