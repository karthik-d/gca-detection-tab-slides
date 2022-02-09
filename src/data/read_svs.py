import os
import numpy as np
import tempfile
from pathlib import Path

import openslide
from PIL import Image
#from config import config

"""
PARAMETERES TO BE SET ---------------------------------------------
"""
# 1. Name of the directory containing the .svs files to be converted
CONVERSION_DIR = 'check'

# 2. Names of files to be excluded from the data-path (if any)
EXCLUDE_FILES = [
    # "mixed_13829$2000-050-10$US$SCAN$OR$001 -001.tiff",
    # "sample.tiff"
]
"""
--------------------------------------------------------------------
"""


BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, 'dataset', 'data'))
CONVERSION_PATH = os.path.join(BASE_PATH, CONVERSION_DIR)
EXTRACTS_PATH = os.path.join(BASE_PATH, CONVERSION_PATH+'-extracts')

REQD_REL_IMGS = [
	'thumbnail',
	'macro',
	'label'
]
REL_IMG_FORMAT = 'png'


if not os.path.isdir(CONVERSION_PATH):
	raise ValueError(f"Could not find {CONVERSION_PATH}")


def make_temp_arrfile(slide, mode='w+'):
	"""
	Create or truncate an existing file
	with required shape buffer
	Return its file pointer object
	"""
	# Extract configuration
	test_part = np.asarray(slide.read_region((0,0), level=0, size=(1,1)))
	dtype = test_part.dtype
	shape=(*slide.dimensions, *test_part.shape[2:])
	# Prepare file
	tempdir = tempfile.TemporaryDirectory()
	file_destn = os.path.join(tempdir.name, 'temp_')
	return np.memmap(file_destn, shape=shape, dtype=dtype, mode=mode)


def get_in_parts(slide, filename, part_size):
	range_x, range_y = part_size
	# Extract till image ends
	start_x = 0
	extent_x = range_x
	last_x = False
	while extent_x!=0 and (start_x+extent_x)<=slide.dimensions[0]:
		start_y = 0
		extent_y = range_y
		last_y = False
		while extent_y!=0 and (start_y+extent_y)<=slide.dimensions[1]:
			part_data = np.asarray(slide.read_region(
				(start_x, start_y), 
				level=0,
				size=(extent_x, extent_y)
			))
			yield (
				np.transpose(part_data, (1, 0, 2)),
				start_x,
				start_y
			)
			start_y += extent_y
			# Include remainder patch
			if not last_y and (start_y+extent_y)>slide.dimensions[1]:
				extent_y = slide.dimensions[1] - start_y
				last_y = True
		# Next x-level
		start_x += extent_x
		# Include remainder patch
		if not last_x and (start_x+extent_x)>slide.dimensions[0]:
			extent_x = slide.dimensions[0] - start_x
			last_x = True
	

def extract_slide0(slide, filename, part_size=(2048, 2048)):    
	# Open accumulator file
	img_acc = make_temp_arrfile(slide)
	for part, x, y in get_in_parts(slide, filename, part_size):
		img_acc[x:x+part.shape[0], y:y+part.shape[1], :] = part
	# Retranspose the array
	img_acc = np.transpose(img_acc, (1, 0, 2))
	return img_acc


if __name__=='__main__':

	"""
	- Creates a directory at same level as the conversion files directory (CONVERSION_DIR)
	- Named as [CONVERSION_DIR]-extracts
	- Contains 1 subdirectory per file, each with:
		- main.tiff: Level-0 slide (Highest Resolution)
		- thumbnail.tiff: Thumbnail image
		- macro.tiff: Macro of the slide
		- label.tiff: Label of the slide
	"""

	# Create extraction target directory
	Path(EXTRACTS_PATH).mkdir(
		parents=False,
		exist_ok=True
	)

	skipped_files = []
	cnt_extracted = 0
	for filename in os.listdir(CONVERSION_PATH):

		if filename in EXCLUDE_FILES:
			continue

		filepath = os.path.join(CONVERSION_PATH, filename)
		if not os.path.isfile(filepath):
			print(f"Not a valid file: {filepath}\nSkipped")
			skipped_files.append(filepath)
			continue

		slide = openslide.OpenSlide(filepath)

		# Extract related images
		for map_key in slide.associated_images:
			if map_key in REQD_REL_IMGS and isinstance(slide.associated_images.get(map_key), Image.Image):
				save_path = ".".join([
					os.path.join(EXTRACTS_PATH, map_key),
					REL_IMG_FORMAT
				])
				slide.associated_images.get(map_key).save(
					fp=save_path,
					format=REL_IMG_FORMAT
				)

		cnt_extracted += 1
		#extract_representation(slide, filename)

	print(f"Extracted {cnt_extracted} file(s)")
	print("\nThe following file(s) could not be processed:" + "\n".join(skipped_files)) if skipped_files else None

# TODO: Generate a .csv of all metadata (slide.properties)
# TODO: Infer 'FILES' from another .csv/.txt instead ?
