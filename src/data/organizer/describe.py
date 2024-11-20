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
import numpy as np

# SET THESE PARAMETERS

#--enter
# SRC_PATH = os.path.abspath(os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
#     "dataset",
#     "data",
#     "roi",
#     "ds_phase_3_raw",
#     "labeled"
# ))

#--enter (will be overriden by function args)
SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_4"
))

#--enter
label_names = ['Y', 'N', 'E', 'NAR']


def _describe_datafolder(data_path, to_file, display):
	print(f"[INFO] Describing data folder: {data_path} ...")

	stats_file = csv.writer(open("data_description.csv", "w"))
	stats_file.writerow(["slide_name", "roi_name", "filepath", "label"])

	for slide_name in sorted(os.listdir(data_path)):
		slide_path = os.path.join(data_path, slide_name)
		# exclude system files.
		if slide_name.startswith('.'):
			continue

		for label in label_names:

			label_path = os.path.join(slide_path, label)
			for roi_name in sorted(os.listdir(label_path)):
				# exclude system files.
				if roi_name.startswith('.'):
					continue

				roi_path = os.path.join(label_path, roi_name)
				stats_file.writerow([slide_name, roi_name, roi_path, label])

	if display or not to_file:
		data_df = pd.read_csv("data_description.csv")

		if display:
			print(data_df.groupby(['label']).size())

		if not to_file:
			os.remove("data_description.csv")
			return data_df


def describe_datafolder(data_path=None, to_file=True, display=True):
    """
    Use data_path=None to set data source in as env-var or global-var
    """

    if data_path is None:
        return _describe_datafolder(SRC_PATH, to_file, display)
    else:
        return _describe_datafolder(data_path, to_file, display)



