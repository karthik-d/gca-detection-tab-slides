import os
import numpy as np
import tempfile
import openslide
from PIL import Image
#from config import config

# Store path-to directory containing .svs files to be converted 
BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, 'dataset', 'data'))
# filenames to be converted
FILES = [
    "mixed_13829$2000-050-10$US$SCAN$OR$001 -001.tiff",
    # "sample.svs"
]

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
	print("DIM", slide.dimensions)
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
	

def extract_representation(slide, filename, part_size=(2048, 2048)):    
	# Open accumulator file
	img_acc = make_temp_arrfile(slide)
	for part, x, y in get_in_parts(slide, filename, part_size):
		img_acc[x:x+part.shape[0], y:y+part.shape[1], :] = part
	# Retranspose the array
	img_acc = np.transpose(img_acc, (1, 0, 2))
	# Save to disk
	Image.fromarray(img_acc).save(os.path.join(BASE_PATH, 'check.tiff'))



for filename in FILES:
    print(os.path.join(BASE_PATH, filename))
    slide = openslide.OpenSlide(os.path.join(BASE_PATH, filename))
    extract_representation(slide, filename)
    #level_0_img = slide.read_region((0,0), level=0, size=slide.level_dimensions[0])

# TODO: Generate a .csv of all metadata (slide.properties)
# TODO: Infer 'FILES' from another .csv/.txt instead ?
