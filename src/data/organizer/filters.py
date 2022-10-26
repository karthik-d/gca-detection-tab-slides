import os
import pandas as pd
from pathlib import Path
import glob


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
    "ds_phase_3"
))

#--enter
classes_to_extract = [ 'Y', 'N' ]

# ---------------------------------

class_path_map = {
    class_: os.path.join(DESTN_PATH, 'assorted', class_) 
    for class_ in classes_to_extract
}


def filter_by_roiname(filenames_file, retain_listed=True):

    # Create destination template
    for class_, path_ in class_path_map.items():
        Path(path_).mkdir(
            parents=True,
            exist_ok=True
        )

    print("\nDestination template created")

    # Assort data files
    master_ctr = 0
    for wsi_name in os.listdir(SRC_PATH):
        
        copy_ctr = 0
        wsi_path = os.path.join(SRC_PATH, wsi_name)
        for class_name in os.listdir(wsi_path):
            
            # check if valid class name
            if class_name in classes_to_extract:
                class_path = os.path.join(wsi_path, class_name)
                # Move all .tiff files
                for filename in glob.glob(os.path.join(class_path, "*.tiff")):
                    # shutil.copy2(
                    #     src=os.path.join(class_path, filename),
                    #     dst=class_path_map.get(class_name)
                    # )
                    copy_ctr += 1
        
        print(f"\nCopied {copy_ctr} files for {wsi_name}")
        master_ctr += copy_ctr 
    
    print(f"Total Files Copied: {master_ctr}")

    # for filename in open(filenames_file, 'r').readlines():
    #     filename = filename.strip()