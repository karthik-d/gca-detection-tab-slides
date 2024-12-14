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
ROI_NAMES = [
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_13.tiff",
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_13.tiff",
	"13829$2000-005-5$US$SCAN$OR$001 -003-region_21.tiff",
	"13829$2000-050-04$US$SCAN$OR$001 -region_11.tiff",
	"13829$2000-050-04$US$SCAN$OR$001 -region_2.tiff",
	"13829$2000-050-08$US$SCAN$OR$001 -region_3.tiff",
	"13829$2002-022-1$US$SCAN$OR$001 -region_11.tiff",
]


# TODO: (MAJOR) save data files without hierarchy; use annotation files to classify and select.

def select_rois():

	# create destination if it doesn't exist.abs
	Path(DESTN_PATH).mkdir(
		parents=True,
		exist_ok=True
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
			
		# copy the roi.
		shutil.copy2(
			src = roi_filepath[0],
			dst = os.path.join(DESTN_PATH, roi_name)
		)
		ctr += 1
		

	print(f"[INFO] copied {ctr} roi files to {DESTN_PATH}.")
	return ctr