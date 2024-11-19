"""
Constructs directories for the split (by ratio specified)
from: `SRC_DIR` which must of the form:

    [SRC_DIR]/
        - Y/
            *.tiff
            .
            .
        - N/
            *.tiff
            .
            .

at: the specified `DESTN_PATH` of the form:
    
    - [DESTN_PATH]/
        - <split-1>/
            - Y/
                *.tiff
            - N/
                *.tiff
        - <split-2>/
            - Y/
                *.tiff
            - N/
                *.tiff
        - .
        - .
"""

#TODO: Use global config for setting paths

import os
import glob
import shutil
from pathlib import Path
import random


# SET THESE PARAMETERS

#--enter
SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_1",
    "assort"
))

#--enter
DESTN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_1"
))

#--enter
data_splits = ['train', 'valid']

#--enter
split_fractions = [0.8, 0.2]

#--enter
classes_to_split = ['P', 'N']

# ------------------------------


def split_pooled():

    # Ensure valid ratios
    assert sum(split_fractions)==1, "Split Fractions must sum to 1"
    assert len(data_splits)==len(split_fractions), "Split fractions count must match data splits count"

    split_path_map = {
        split_: os.path.join(DESTN_PATH, 'splits', split_) 
        for split_ in data_splits
    }

    # Create destination template
    for split_, path_ in split_path_map.items():
        for class_name in classes_to_split:
            Path(
                os.path.join(path_, class_name)
            ).mkdir(
                parents=True,
                exist_ok=False
            )
    print("\nDestination template created")
    
    # Split data files
    for class_name in os.listdir(SRC_PATH):

        if class_name not in classes_to_split:
            continue
            
        print(f"\nProcessing {class_name}...")
        class_path = os.path.join(SRC_PATH, class_name)

        all_files = glob.glob(os.path.join(class_path, "*.tiff"))
        num_files = len(all_files)
        print("Files in class:", num_files)

        # Shuffle files randomly
        random.shuffle(all_files)

        start_idx = 0
        for split_name, split_ratio in zip(data_splits, split_fractions):
            
            # Find files to move by ratio
            end_idx = start_idx + int(num_files * split_ratio) 
            files_for_split = all_files[start_idx : end_idx]

            # Move the files for split
            split_ctr = 0
            for filename in files_for_split:
                shutil.copy2(
                    src=os.path.join(class_path, filename),
                    dst=os.path.join(split_path_map.get(split_name), class_name)
                )
                split_ctr += 1

            # Update indices
            start_idx = end_idx
            print(f"Split {split_ctr} for {split_name}")

        # Copy remainder to last split - integer approximation
        remainder_files = all_files[end_idx:]
        for file_name in files_for_split:
            shutil.copy2(
                src=os.path.join(class_path, filename),
                dst=os.path.join(split_path_map.get(split_name), class_name)
            )
        print(f"Split remaining {len(remainder_files)} for {split_name}\n")