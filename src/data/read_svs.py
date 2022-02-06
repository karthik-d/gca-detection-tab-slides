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
    #"mixed_13829$2000-050-10$US$SCAN$OR$001 -001.tiff",
    "sample.svs"
]

def make_temp_arrfile(slide, mode='w+'):
    """
    Create or truncate an existing file
    with required shape buffer
    Return its file pointer object
    """
    tempdir = tempfile.TemporaryDirectory()
	dtype = slide.read_region((0,0), level=0, size=(0,1)).dtype
	shape=(*slide.dimensions, *test_part.shape[2:])
    # Prepare file
    file_destn = os.path.join(tempdir.name, 'temp_')
    return np.memmap(file_destn, shape=shape, dtype=dtype, mode=mode)


def get_in_parts(slide, filename, part_size):
	range_x, range_y = part_size
	start_x = 0
	start_y = 0
	# Extract till image ends
	while (start_x+range_x)<slide.dimensions[0]:
		while (start_y+range_y)<slide.dimensions[1]:
			yield (
				slide.read_region(
					(start_x, start_y), 
					level=0,
					size=(range_x, range_y)
				),
				start_x,
				start_y,
				range_x,
				range_y 
			)
			start_x += range_x 
			start_y += range_y


def extract_representation(slide, filename, part_size=(4096, 4096)):    
	# Open accumulator file
	img_acc = make_temp_arrfile(slide)
	for part_data, x, y, _x, _y in get_in_parts(slide, filename, part_size):
		img_acc[x:x+_x, y:y+_y, :] = part_data
	# Save to disk
	Image.fromarray(img_acc).save(os.path.join(BASE_PATH, 'check.tiff'))


for filename in FILES:
    print(os.path.join(BASE_PATH, filename))
    slide = openslide.OpenSlide(os.path.join(BASE_PATH, filename))
    extract_representation(slide, filename)
    #level_0_img = slide.read_region((0,0), level=0, size=slide.level_dimensions[0])

# TODO: Generate a .csv of all metadata (slide.properties)
# TODO: Infer 'FILES' from another .csv/.txt instead ?
