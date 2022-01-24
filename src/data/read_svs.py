import os
import numpy as np
import openslide
from PIL import Image
#from config import config

# Store path-to directory containing .svs files to be converted 
BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, 'dataset', 'data'))
# filenames to be converted
FILES = [
    "mixed_13829$2000-050-10$US$SCAN$OR$001 -001.tiff",
]

def get_in_parts(slide, part_size=(4096, 4096)):
    part_range_x, part_range_y = part_size
    test_part = np.asarray(slide.read_region((0,0), level=0, size=(1,1)))
    img_acc = np.empty((*slide.dimensions, *test_part.shape[2:]), dtype=test_part.dtype)


for file in FILES:
    slide = openslide.OpenSlide(os.path.join(BASE_PATH, file))
    get_in_parts(slide)
    #level_0_img = slide.read_region((0,0), level=0, size=slide.level_dimensions[0])


# TODO: Generate a .csv of all metadata (slide.properties)
# TODO: Infer 'FILES' from another .csv/.txt instead ?
