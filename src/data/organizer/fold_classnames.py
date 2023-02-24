"""
Fold class names 
from alt_names = [[<alt_name_1_for_class_1>, ...], [<alt_name_1_for_class_2>, ...], ... ] 
to   fold_names = [<fold_name_for_class_1, ...] 
for `SRC_DIR` which must be of the form:

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
import numpy as np

# SET THESE PARAMETERS

#--enter
# SRC_PATH = os.path.abspath(os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
#     "dataset",Generates basic data sample enumeration in the $PWD as "data_description.csv" file, and as STDOUT
#     "data",
#     "roi",
#     "ds_phase_3_raw",
#     "labeled"
# ))

SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*4), 
    "dataset",
    "annotations",
    "phase-on-07Feb23"
))

#--enter
alt_names = [['P', 'Y']]

#--enter
fold_names = ['Y']


def _fold_classnames(data_path):
	print(f"[INFO] Folding class names in data folder: {SRC_PATH} ...")

	count_renames = 0
	for slide_name in sorted(os.listdir(SRC_PATH)):
		slide_path = os.path.join(SRC_PATH, slide_name)

		for alts, fold in zip(alt_names, fold_names):

			curr_label_names = os.listdir(slide_path)
			if fold in curr_label_names:
				continue
			
			for label_name in curr_label_names:
				if label_name in alts:
					os.rename(
						os.path.join(slide_path, label_name),
						os.path.join(slide_path, fold)
					)
					count_renames += 1
	
	print("[INFO] Folded {} label names".format(count_renames))


def fold_classnames(data_path=None):
	"""
	Use data_path=None to set data source as env-var or global-var
	"""

	if data_path is None:
		return _fold_classnames(SRC_PATH)
	else:
		return _fold_classnames(data_path)



