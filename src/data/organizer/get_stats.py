"""
Generates basic stats in the $PWD as "stats.csv" file, and as STDOUT for 
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
    "ds_phase_3_raw",
    "labeled"
))


stats_file = csv.writer(open("stats.csv", "w"))
csv.writerow(["slide_name", "roi_name", "filepath", "label"])

for slide_name in sorted(os.listdir(SRC_PATH)):
    