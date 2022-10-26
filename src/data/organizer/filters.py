import os
import pandas as pd
from pathlib import Path
import glob
import shutil


#--enter
SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_3_raw",
    "labeled"
))

#--enter
DESTN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_3",
    "filtered"
))

#--enter
classes_to_extract = [ 'Y', 'N' ]

# ---------------------------------


def filter_by_roiname(filenames_file, retain_listed=True):

    # Assort data files
    filter_files = list(map(lambda x: x.strip(), open(filenames_file, 'r').readlines()))
    master_ctr = 0
    for wsi_name in os.listdir(SRC_PATH):
        
        copy_ctr = 0
        wsi_suffix = wsi_name
        for class_name in os.listdir(os.path.join(SRC_PATH, wsi_suffix)):
            
            # check if valid class name
            if class_name in classes_to_extract:
                class_suffix = os.path.join(wsi_suffix, class_name)
                # Move all .tiff files
                for filename in glob.glob(os.path.join(SRC_PATH, class_suffix, "*.tiff")):
                    
                    if (
                        (os.path.basename(filename) in filter_files and retain_listed) 
                        or os.path.basename(filename) not in filter_files and not retain_listed
                    ): 

                        # create destination if it doesn't exist
                        if not os.path.isdir(os.path.join(DESTN_PATH, class_suffix)):
                            Path(os.path.join(DESTN_PATH, class_suffix)).mkdir(
                                parents=True,
                                exist_ok=False
                            )

                        shutil.copy2(
                            src=os.path.join(filename),
                            dst=os.path.join(DESTN_PATH, class_suffix)
                        )
                        copy_ctr += 1
        
        print(f"\nCopied {copy_ctr} files for {wsi_name}")
        master_ctr += copy_ctr 
    
    print(f"Total Files Copied: {master_ctr}")