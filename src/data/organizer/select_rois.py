"""
Moves a list of ROI files from supplies source directory of the form `ds_phase_*_raw/` (the file hierarchy is assumed)
into a target directory, simply, i.e., not hierarchy in target.

Typical use-case: moving manual choices of rois for evaluation, annotation, viz, etc.
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
    "ds_phase_4",
	"viz",
	"gradcam_inputs"
))

#--enter
data_classes = [ 'Y', 'N' ]

#--enter
ROI_NAMES = [
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_13.tiff",
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_13.tiff",
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_21.tiff",
	"13829$2000-050-04$US$SCAN$OR$001 -region_11.tiff",
	"13829$2000-050-04$US$SCAN$OR$001 -region_2.tiff",
	"13829$2000-050-08$US$SCAN$OR$001 -region_3.tiff",
	"13829$2002-022-1$US$SCAN$OR$001 -region_11.tiff",
	"13829$2017-044-6$US$SCAN$OR$001 -region_1.tiff",   # neg sample.
	"13829$2000-050-11$US$SCAN$OR$001 -region_4.tiff"   # neg sample.
]


# TODO: (MAJOR) save data files without hierarchy; use annotation files to classify and select.

def select_rois():

	# create destination if it doesn't exist; reproduce the classification hierarchy.
	for class_ in data_classes:
		Path(os.path.join(DESTN_PATH, class_)).mkdir(
			parents=True, exist_ok=True
		)

	# parse rois and pick them from hierarchy.
	ctr = 0
	for roi_name in ROI_NAMES:
		
		# search recursively in path.
		roi_filepath = glob.glob(os.path.join(SRC_PATH, "*/*", roi_name))
		if not roi_filepath:
			print("[ERROR] f{roi_name} not found.")
			continue
		elif len(roi_filepath)>1:
			print(f"[ERROR] ambiguous matches for {roi_name}.")
			continue 
		else:
			roi_filepath = roi_filepath[0]
			
		# copy the roi.
		shutil.copyfile(
			src = roi_filepath,
			dst = os.path.join(DESTN_PATH, *Path(roi_filepath).parts[-2:])
		)
		ctr += 1
		

	print(f"[INFO] copied {ctr} roi files to {DESTN_PATH}.")
	return ctr