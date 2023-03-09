"""
Constructs a data directory
from: `SRC_DIR` which must be of the form:

    [SRC_DIR]/ (typically, called ds_phase_N_raw)
        - <wsi-name>/
            - Y/
            - N/
            - (others ...) - ignored
        - <wsi-name>/
            - .
            - .
            - .

at the specified `DESTN_PATH` of the form:
    
    - [DESTN_PATH]/  (typically, called ds_phase_N)
        - assorted/
            - Y/
                *.tiff
                .
                .
            - N/
                *.tiff
                .
                .
"""

#TODO: Use global config for setting paths

import os
import glob
import shutil
from pathlib import Path


# SET THESE PARAMETERS

#--enter
SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_4_raw"
))

#--enter
DESTN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_4"
))

#--enter
classes_to_extract = [ 'Y', 'N' ]

# ---------------------------------


def assort_classwise():
    """
    Driver function for assorting
    """

    class_path_map = {
        class_: os.path.join(DESTN_PATH, 'assorted', class_) 
        for class_ in classes_to_extract
    }

    # Create destination template
    for class_, path_ in class_path_map.items():
        Path(path_).mkdir(
            parents=True,
            exist_ok=False
        )
    print("\nDestination template created")

    # Assort data files
    for wsi_name in os.listdir(SRC_PATH):
        
        copy_ctr = 0
        wsi_path = os.path.join(SRC_PATH, wsi_name)
        for class_name in os.listdir(wsi_path):
            
            # check if valid class name
            if class_name in classes_to_extract:
                class_path = os.path.join(wsi_path, class_name)
                # Move all .tiff files
                for filename in glob.glob(os.path.join(class_path, "*.tiff")):
                    shutil.copy2(
                        src=os.path.join(class_path, filename),
                        dst=class_path_map.get(class_name)
                    )
                    copy_ctr += 1
        
        print(f"\nCopied {copy_ctr} files for {wsi_name}")