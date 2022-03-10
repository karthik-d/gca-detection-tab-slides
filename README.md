# GCA-Detection-TAB-Slides

Deep Neural Network to automate the detection of Giant Cell Arteritis from digital pathology slides of Temporal Artery Biopsy.

## Working Notes

 Link to doc notes: [here](https://docs.google.com/document/d/1EI5U-VP_N0la0jteKMeJGxWmzn7jbAj8no2Khai4MwM/edit)      
 Link to sample ROIs: [here](https://drive.google.com/drive/folders/1w94vfqY0z4Gr2lwi8LnAQiKYCEJDIUHV?usp=sharing)
 
## Progress Tracking

1. IBM Code adapted:
    - [X] retain original file names post conversions
    - [X] perform 16x, 8x & w/o downscaling (update immediately)
    - [X] only retain the unflitered and roi (black background) files
2. Create a code to pick each ROI and filename_01, filename_02, ... (segmentation)
    - [X] extract roi
3. Labelling
    - [ ] Map from annotations file
    - [ ] Verify labelled dataset

## Procedure to Extract ROI

Check out the video demo [here](https://drive.google.com/file/d/1Hb6pySfqGVqKtTUiBUeS54muv80gTESg/view?usp=sharing)

Please download and extract the zipped archive with the scripts from [here](https://drive.google.com/file/d/1hw_JGWN8uumKNLK6Cvu9nqv9Ff-BuPSM/view?usp=sharing).

### Install dependencies

Please ensure that you are using a **macOS** or **Linux** development environment as some dependencies are OS specific.
Use Anaconda to manage your packages and **Python 3 (version >= 3.6.0 recommended)**.

#### **(Recommended)** Using an Anaconda environment
- Use the dependency file `dep-file-conda.txt` [from here](https://raw.githubusercontent.com/karthik-d/TAB-Slides/main/dep-file-conda.txt) or from the zip-folder `GCA-Detection/`
- **Either** create a new environment with all dependencies by running   
`conda create --name myenv --file dep-file-conda.txt`
- **Or** install to an existing environment by running   
`conda install --name myenv --file dep-file-conda.txt`

    **Finally**, switch to the created/modified environment by running   
    `conda activate myenv`


#### **(Not preferred)** Using pip
- Use the dependency file `dep-file-conda.txt` [from here](https://raw.githubusercontent.com/karthik-d/TAB-Slides/main/dep-file-pip.txt) or from the zip-folder `GCA-Detection/`
- Install all dependencies by running   
`pip install -r dep-file-pip.txt`


### Load the data

1. Navigate to `GCA-Detection/dataset/data/final`
2. Move/Copy all .svs files from which ROIs need to be extracted into this folder.

    The directory should finally look something like:   
    ```
        GCA-Detection
        |_ dataset
            |_ data
            |_final
                |_ Neg_13829$2020-025-5$US$SCAN$OR$001 -003.svs
                |_ Postivie_13829$2000-005-5$US$SCAN$OR$001 -003.svs
                |_ mixed_13829$2000-050-10$US$SCAN$OR$001 -001.svs
                |_ .
                |_ .
        |_ src
            |_ .
            |_ .
    ```

### Execute the extraction script

1. Navigate to `GCA-Detection/src/features`
2. Set the downsampling level:    
    - Edit the file `extract_roi.py`
    - Set the `downscale_level` argument in the function `roi.multiprocess_extract_roi_from_filtered()` to a suitable value.    
    Please refer to the instructions in the file.   
    The same instructions are reproduced here for convenience:   

        ```
        Set downscale_level to:
        
        0, if ROIs must be from - TOP SLIDE (Highest Resolution) 
        1, if ROIs must be from - x4 DOWNSCALED SLIDE
        2, if ROIs must be from - x16 DOWNSCALED SLIDE
        3, if ROIs must be from - x64 DOWNSCALED SLIDE
        
        NOTE: Only one of 0, 1, 2 or 3 must be specified. 
        Other downscaling levels have been pruned to improve execution time.
        ```
3. Execute the `extract_roi.py` file using the command  
        ```
        python extract_roi.py
        ```
4. Extracted ROIs will be generated into `GCA-Detection/dataset/data/roi/`

### Result Description

- The extracted ROIs will be contained in `GCA-Detection/dataset/data/roi/`

- A subdirectory will be generated in `../roi` for each input `.svs` image in `GCA-Detection/dataset/data/final/` with the same name as the input image.

    Each of these subdirectories contains:
    - All the ROIs extracted from that slide, named as `<slide_name>_region_<roi_num>.tiff`
    - A subdirectory called `related-imgs` containing a _thumbnail_ and the _label_ meta-images of the slide.
