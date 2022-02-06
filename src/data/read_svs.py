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

# TODO: Move temp directory to data dir
def make_temp_arrfile(filename, shape, dtype, mode='w+'):
    """
    Create or truncate an existing file
    with required shape buffer
    Return its file pointer object
    """
    tempdir = tempfile.TemporaryDirectory()
    # Prepare file
    file_destn = os.path.join(tempdir.name, 'temp_'+filename)
    return np.memmap(file_destn, shape=shape, dtype=dtype, mode=mode)


def get_in_parts(slide, part_size=(4096, 4096), filename='unkown'):
    part_range_x, part_range_y = part_size
    print(slide.dimensions)
    test_part = np.asarray(slide.read_region((800, 800), level=0, size=(200,200)))
    print(test_part.shape)
    # Open accumulator file
    img_acc = make_temp_arrfile(
        filename, 
        shape=(*slide.dimensions, *test_part.shape[2:]), 
        dtype=test_part.dtype
    )
    img_acc[0:200, 0:200, :] = test_part
    Image.fromarray(img_acc).save(os.path.join(BASE_PATH, 'check.tiff'))
    print("HERE")

for filename in FILES:
    print(os.path.join(BASE_PATH, filename))
    slide = openslide.OpenSlide(os.path.join(BASE_PATH, filename))
    get_in_parts(slide, filename=filename)
    #level_0_img = slide.read_region((0,0), level=0, size=slide.level_dimensions[0])

# TODO: Generate a .csv of all metadata (slide.properties)
# TODO: Infer 'FILES' from another .csv/.txt instead ?
