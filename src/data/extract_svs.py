import os
import numpy as np
import tempfile
from pathlib import Path
import time

import openslide
from PIL import Image
#from config import config

DOWNSCALE_FACTORS = [1, 4, 16, 32]
FACTOR_LEVEL_MAP = dict(zip(DOWNSCALE_FACTORS, range(len(DOWNSCALE_FACTORS))))

"""
PARAMETERES TO BE SET ---------------------------------------------
"""
# 1. Name of the directory containing the .svs files to be converted
CONVERSION_DIR = 'final'

# 2. Names of files to be excluded from the data-path (if any)
EXCLUDE_FILES = [
    # "mixed_13829$2000-050-10$US$SCAN$OR$001 -001.tiff",
	# "negative.tiff",
	# "Postivie_13829$2000-005-5$US$SCAN$OR$001 -003.tiff",
	"test.tiff",
	"14276.svs",
    "sample.tiff"
]

# 3. Downscaling level - set to one of the values in `DOWNSCALE_FACTORS` list above
DOWNSCALE_FACTOR = 32

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


def make_temp_arrfile(slide, level=0, mode='w+'):
	"""
	Create or truncate an existing file
	with required shape buffer
	Return its file pointer object
	"""
	# Extract configuration
	test_part = np.asarray(slide.read_region((0,0), level, size=(1,1)))
	dtype = test_part.dtype
	shape=(*slide.level_dimensions[level], *test_part.shape[2:])
	# Prepare file
	tempdir = tempfile.TemporaryDirectory()
	file_destn = os.path.join(tempdir.name, 'temp_')
	return np.memmap(file_destn, shape=shape, dtype=dtype, mode=mode)


def get_prescale_value(slide, level):
	return int(slide.level_downsamples[level]) if level!=0 else 1


def get_in_parts(slide, level, part_size):
	range_x, range_y = part_size
	prescale = get_prescale_value(slide, level)

	# Extract till image ends
	start_x = 0
	extent_x = prescale*range_x
	start_x_downscaled = 0
	extent_x_downscaled = range_x
	last_x = False
	while (not last_x) or (extent_x!=0 and (start_x+extent_x)<=slide.level_dimensions[0][0]):
		
		# Include remainder patch
		if not last_x and (start_x+extent_x)>slide.level_dimensions[0][0]:
			extent_x = slide.level_dimensions[0][0] - start_x
			extent_x_downscaled = slide.level_dimensions[level][0] - start_x_downscaled
			last_x = True

		# Iterate vertically
		start_y = 0
		extent_y = prescale*range_y
		start_y_downscaled = 0
		extent_y_downscaled = range_y
		last_y = False

		while (not last_y) or (extent_y!=0 and (start_y+extent_y)<=slide.level_dimensions[0][1]):
			# Include remainder patch
			if not last_y and (start_y+extent_y)>slide.level_dimensions[0][1]:
				extent_y = slide.level_dimensions[0][1] - start_y
				extent_y_downscaled = slide.level_dimensions[level][1] - start_y_downscaled
				last_y = True
			# Extract part
			part_data = np.asarray(slide.read_region(
				(start_x, start_y), 
				level=level,
				size=(extent_x_downscaled, extent_y_downscaled)
			))
			yield (
				np.transpose(part_data, (1, 0, 2)),
				start_x_downscaled,
				start_y_downscaled
			)
			start_y += extent_y
			start_y_downscaled += extent_y_downscaled
			
		# Next x-row
		start_x += extent_x
		start_x_downscaled += extent_x_downscaled
	

def extract_level(slide, level=0, part_size=(2048, 2048)):    
	# Open accumulator file
	img_acc = make_temp_arrfile(slide, level)
	for part, x, y in get_in_parts(slide, level, part_size):
		img_acc[x:x+part.shape[0], y:y+part.shape[1], :] = part
		print(x, y)
	# Retranspose the array
	img_acc = np.transpose(img_acc, (1, 0, 2))
	return img_acc


def infer_scaling_levels(slide):
	zero_w, zero_h = slide.dimensions
	scales = [(1, 1)]
	for resolution in slide.level_dimensions[1:]:
		w, h = resolution 
		scales.append((zero_w/w, zero_h/h))
	return scales


if __name__=='__main__':

	"""
	- Creates a directory at same level as the conversion files directory (CONVERSION_DIR)
	- Named as [CONVERSION_DIR]-extracts
	- Contains 1 subdirectory per file, each with:
		- main.tiff: Level-0 slide (Highest Resolution)
		- thumbnail.[REL_IMG_FORMAT]: Thumbnail image
		- macro.[REL_IMG_FORMAT]: Macro of the slide
		- label.[REL_IMG_FORMAT]: Label of the slide
	"""

	if DOWNSCALE_FACTOR is None or DOWNSCALE_FACTOR not in DOWNSCALE_FACTORS:
		print("\nSet DOWNSCALE_FACTOR")
		print(f"Available Downscale Factors: {DOWNSCALE_FACTORS}")
		exit()

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

		src_path = os.path.join(CONVERSION_PATH, filename)
		if not os.path.isfile(src_path):
			print(f"Not a valid file: {src_path}\nSkipped")
			skipped_files.append(src_path)
			continue

		print(f"\nProcessing {src_path}...")
		destn_path = os.path.join(EXTRACTS_PATH, filename.split('.')[0])
		Path(destn_path).mkdir(
			parents=False,
			exist_ok=True
		)

		slide = openslide.OpenSlide(src_path)
		print(slide.level_dimensions)

		"""
		# TEST to check if using the 'get_best_level_for_downsample' is suitable
		print(slide.get_best_level_for_downsample(32))
		print("Level-wise scaling")
		for level, scale in enumerate(infer_scaling_levels(slide)):
			print(f"Level {level} - Width: {scale[0]}, Height: {scale[1]}")
		print(slide.level_downsamples)
		"""		

		start_ = time.time()
		img = extract_level(
			slide, 
			level=FACTOR_LEVEL_MAP[DOWNSCALE_FACTOR], 
			part_size=((1024, 1024))
		)
		save_path = os.path.join(destn_path, 'main.tiff')
		Image.fromarray(img).save(save_path, compression='tiff_lzw')
		print(save_path)
		print(f"Converted and stored in {time.time()-start_} s")

		# Extract related images
		for map_key in slide.associated_images:
			if map_key in REQD_REL_IMGS and isinstance(slide.associated_images.get(map_key), Image.Image):
				save_path = ".".join([
					os.path.join(destn_path, map_key),
					REL_IMG_FORMAT
				])
				slide.associated_images.get(map_key).save(
					fp=save_path,
					format=REL_IMG_FORMAT
				)

		cnt_extracted += 1
		#extract_representation(slide, filename)

	print(f"\nExtracted {cnt_extracted} file(s)")
	print("\nThe following file(s) could not be processed:" + "\n".join(skipped_files)) if skipped_files else None

# TODO: Generate a .csv of all metadata (slide.properties)
