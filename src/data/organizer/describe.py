"""
Generates basic data sample enumeration in the $PWD as "data_description.csv" file, and as STDOUT for 
`SRC_DIR` which must be of the form:

    [SRC_DIR]/
        - Slide-X/
            - Y/
                *.tiff
                .
                .
            - N/
                *.tiff
                .
                .
          .
          .
          .
"""

import csv
import os
import pandas as pd

# SET THESE PARAMETERS

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
label_names = ['Y', 'N', 'E', 'NAR']


def describe_datafolder(display=True):
    print(f"[INFO] Describing data folder: {SRC_PATH} ...")

    stats_file = csv.writer(open("data_description.csv", "w"))
    stats_file.writerow(["slide_name", "roi_name", "filepath", "label"])

    for slide_name in sorted(os.listdir(SRC_PATH)):
        slide_path = os.path.join(SRC_PATH, slide_name)

        for label in label_names:
            label_path = os.path.join(slide_path, label)

            for roi_name in sorted(os.listdir(label_path)):
                roi_path = os.path.join(label_path, roi_name)

                stats_file.writerow([slide_name, roi_name, roi_path, label])

    if display:
        data_df = pd.read_csv("data_description.csv")
        print(data_df.groupby(['label']).size())



