# Procedure to Extract ROI

Check out the video demo [here](https://drive.google.com/file/d/1dOIPB8BWaovi4Cj0zAWIxnHStyxyEoBY/view?usp=sharing)!

Feel free to watch the it at 2x speed since most of the video is just waiting for downloads to complete or the scripts to finish processing.

**NOTE:** The video only demonstrates how to run the ROI extraction scripts from the svs files already downloaded into the local system. Navigate [here](https://github.com/karthik-d/TAB-Slides/blob/main/docs/03-annotation_procedure.md) for the ROI annotation (manual labelling) instructions. Furthermore, the svs files used for the demo are named Mixed, Neg and Pos; as a result, the ROI will be prefixed by the same name. The labelling process is **NOT AUTOMATED by this script**.

Please download and extract the zipped archive with the scripts from [here](https://drive.google.com/file/d/1C9q9iBO6B72okd34DB3xHJp5LrY76eSb/view?usp=sharing).

## Install dependencies

Please ensure that you are using a **macOS** or **Linux** development environment as some dependencies are OS specific.
Use Anaconda to manage your packages and **Python 3 (version >= 3.6.0 recommended)**.

### **(Recommended)** Using an Anaconda environment
Use the dependency file `dep-file-conda.txt` [from here](https://raw.githubusercontent.com/karthik-d/TAB-Slides/main/dep-file-conda.txt) or from the zip-folder `GCA-Detection/`
- **Either** create a new environment with all dependencies by running   
`conda create --name myenv --file dep-file-conda.txt`
- **Or** install to an existing environment by running   
`conda install --name myenv --file dep-file-conda.txt`
- **Or** use a .yml file to create an environment with all dependencies by running   
`conda env create -f gca.yml`

    **Finally**, switch to the created/modified environment by running   
    `conda activate myenv`


### **(Not preferred)** Using pip
- Use the dependency file `dep-file-conda.txt` [from here](https://raw.githubusercontent.com/karthik-d/TAB-Slides/main/dep-file-pip.txt) or from the zip-folder `GCA-Detection/`
- Install all dependencies by running   
`pip install -r dep-file-pip.txt`


## Load the data

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

## Execute the extraction script

1. Navigate to `GCA-Detection/src/features`
2. Set the downsampling level:    
    - Edit the file `extract_roi.py`
    - Set the `downscale_level` argument in the function `roi.multiprocess_extract_roi_from_filtered()` to a suitable value (**Use 1 here**).    
    Please refer to the instructions in the file. The same instructions are reproduced here for convenience:   

        ```
        Set downscale_level to:
        
        0, if ROIs must be from - TOP SLIDE (Highest Resolution) 
        1, if ROIs must be from - x4 DOWNSCALED SLIDE - USE THIS
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

## Result Description

- The extracted ROIs will be contained in `GCA-Detection/dataset/data/roi/`

- A subdirectory will be generated in `../roi` for each input `.svs` image in `GCA-Detection/dataset/data/final/` with the same name as the input image.

    Each of these subdirectories contains:
    - All the ROIs extracted from that slide, named as `<slide_name>_region_<roi_num>.tiff`
    - A subdirectory called `related-imgs` containing a _thumbnail_ and the _label_ meta-images of the slide.
