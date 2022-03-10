import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image
import re
import sys
import ntpath

from wsi import utils
from wsi.utils import Time

# Used constants: SRC_TRAIN_DIR, SRC_TRAIN_EXT

BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'dataset', 'data'))
TRAIN_PREFIX = "TUPAC-TR-"   # Not in use
SRC_TRAIN_DIR = os.path.join(BASE_DIR, "final")
SRC_TRAIN_EXT = "svs"
DEST_TRAIN_SUFFIX = ""  # Example: "train-"
DEST_TRAIN_EXT = "png"
DEST_ROI_EXT = "tiff"
SCALE_FACTOR = 32
DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"

DEST_TRAIN_THUMBNAIL_DIR = os.path.join(BASE_DIR, "training_thumbnail_" + THUMBNAIL_EXT)

FILTER_SUFFIX = ""  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"
FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)
FILTER_THUMBNAIL_DIR = os.path.join(BASE_DIR, "filter_thumbnail_" + THUMBNAIL_EXT)
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
FILTER_HTML_DIR = BASE_DIR

ROI_SUFFIX = ""
ROI_DIR = os.path.join(BASE_DIR, "roi")

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_HTML_DIR = BASE_DIR

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"

TOP_TILES_SUFFIX = "top_tile_summary"
TOP_TILES_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
TOP_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR,
                                                   TOP_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)
TILE_SUFFIX = "tile"

STATS_DIR = os.path.join(BASE_DIR, "svs_stats")


def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).
  Args:
    filename: Name of the slide file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.OpenSlide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


def open_image(filename):
  """
  Open an image (*.jpg, *.png, etc).
  Args:
    filename: Name of the image file.
  returns:
    A PIL.Image.Image object representing an image.
  """
  image = Image.open(filename)
  return image


def open_image_np(filename):
  """
  Open an image (*.jpg, *.png, etc) as an RGB NumPy array.
  Args:
    filename: Name of the image file.
  returns:
    A NumPy representing an RGB image.
  """
  pil_img = open_image(filename)
  np_img = utils.pil_to_np_rgb(pil_img)
  return np_img

# Expunged "get_training_image_path" - Using filenames based on the TAB Dataset's naming convention
# Added - Replaces 'get_training_image_path' to generate/retrieve filename for downscaled training image
def get_downscaled_training_image_path(slide_filepath, large_w=None, large_h=None, small_w=None, small_h=None):
	slide_filename = ntpath.basename(slide_filepath).split('.')[0]
	if large_w is None and large_h is None and small_w is None and small_h is None:
		wildcard_path = os.path.join(DEST_TRAIN_DIR, slide_filename + "*." + DEST_TRAIN_EXT)
		img_path = glob.glob(wildcard_path)[0]
	else:
		img_path = os.path.join(
			DEST_TRAIN_DIR, 
			slide_filename + "-" + str(SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
			large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT
		)
	return img_path


# Expunged "get_training_thumbnail_path" - Using filenames based on the TAB Dataset's naming convention
# Added - Replaces 'get_training_thumbnail_path' to generate filename for downscaled training image
def make_downscaled_training_thumbnail_path(slide_filepath, large_w, large_h, small_w, small_h):
  slide_filename = ntpath.basename(slide_filepath).split('.')[0]
  img_path = os.path.join(
    DEST_TRAIN_THUMBNAIL_DIR, 
    slide_filename + "-" + str(SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + THUMBNAIL_EXT)
  return img_path


# Modified: Made slide_path based instead of slide_num
def get_filter_image_path(slide_filename, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter image file.
  Example:
    5, 1, "rgb" -> ../data/filter_png/TUPAC-TR-005-001-rgb.png
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
  Returns:
    Path to the filter image file.
  """
  dir = FILTER_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_filename, filter_number, filter_name_info))
  return img_path


def get_filter_thumbnail_path(slide_filename, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter thumbnail file.
  Example:
    5, 1, "rgb" -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-001-rgb.jpg
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
  Returns:
    Path to the filter thumbnail file.
  """
  dir = FILTER_THUMBNAIL_DIR
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_filename, filter_number, filter_name_info, thumbnail=True))
  return img_path


# Modified
def get_filter_image_filename(slide_filepath, filter_number, filter_name_info, thumbnail=False):
  """
  Convert slide number, filter number, and text to a filter file name.
  Example:
    5, 1, "rgb", False -> TUPAC-TR-005-001-rgb.png
    5, 1, "rgb", True -> TUPAC-TR-005-001-rgb.jpg
  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
    thumbnail: If True, produce thumbnail filename.
  Returns:
    The filter image or thumbnail file name.
  """
  slide_filename = ntpath.basename(slide_filepath).split('.')[0]
  if thumbnail:
    ext = THUMBNAIL_EXT
  else:
    ext = DEST_TRAIN_EXT
  padded_fi_num = str(filter_number).zfill(3)
  img_filename = slide_filename + "-" + padded_fi_num + "-" + FILTER_SUFFIX + filter_name_info + "." + ext
  return img_filename


# Modified
def get_filter_image_result_path(slide_filepath):
  """
  Convert slide number to the path to the file that is the final result of filtering.
  Example:
  5 -> ../data/filter_png/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.png
  Args:
  slide_number: The slide number.
  Returns:
  Path to the filter image file.
  """
  slide_filename = ntpath.basename(slide_filepath).split('.')[0]
  training_img_path = get_downscaled_training_image_path(slide_filepath)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(FILTER_DIR, slide_filename + "-" + str(
  SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
  small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
  return img_path


# Modified
def get_filter_thumbnail_result_path(slide_filepath):
	"""
	Convert slide filepath to the path to the file that is the final thumbnail result of filtering.
	Example:
	5 -> ../data/filter_thumbnail_jpg/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.jpg
	Args:
	slide_number: The slide number.
	Returns:
	Path to the filter thumbnail file.
	"""
	slide_filename = ntpath.basename(slide_filepath).split('.')[0]
	training_img_path = get_downscaled_training_image_path(slide_filepath)
	large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
	img_path = os.path.join(
		FILTER_THUMBNAIL_DIR, 
		slide_filename + "-" + str(
			SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
			small_h) + "-" + FILTER_RESULT_TEXT + "." + THUMBNAIL_EXT
	)
	return img_path


def parse_dimensions_from_image_filename(filename):
  """
  Parse an image filename to extract the original width and height and the converted width and height.
  Example:
    "TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)
  Args:
    filename: The image filename.
  Returns:
    Tuple consisting of the original width, original height, the converted width, and the converted height.
  """
  m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
  large_w = int(m.group(1))
  large_h = int(m.group(2))
  small_w = int(m.group(3))
  small_h = int(m.group(4))
  return large_w, large_h, small_w, small_h


# Modified (using slide_path in place of slide_num)
def training_slide_to_image(slide_filepath):
	"""
	Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
	Args:
	slide_number: The slide number.
	"""

	# Scale down the WSI by SCALE_FACTOR
	img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_filepath)
	img_path = get_downscaled_training_image_path(slide_filepath, large_w, large_h, new_w, new_h)
	print("Saving image to: " + img_path)
	if not os.path.exists(DEST_TRAIN_DIR):
		os.makedirs(DEST_TRAIN_DIR)
	img.save(img_path)

	thumbnail_path = make_downscaled_training_thumbnail_path(slide_filepath, large_w, large_h, new_w, new_h)
	save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)

# Modified: changes from slide_num based to slide_path based
def slide_to_scaled_pil_image(slide_filepath):
  """
  Convert a WSI training slide to a scaled-down PIL image.
  Args:
    slide_number: The slide number.
  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """

  print(f"Opening Slide: {slide_filepath}")
  slide = open_slide(slide_filepath)

  large_w, large_h = slide.dimensions
  new_w = math.floor(large_w / SCALE_FACTOR)
  new_h = math.floor(large_h / SCALE_FACTOR)
  level = slide.get_best_level_for_downsample(SCALE_FACTOR)
  whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
  whole_slide_image = whole_slide_image.convert("RGB")
  img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
  return img, large_w, large_h, new_w, new_h


def save_thumbnail(pil_img, size, path, display_path=False):
  """
  Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.
  Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
    display_path: If True, display thumbnail path in console.
  """
  max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
  img = pil_img.resize(max_size, PIL.Image.BILINEAR)
  if display_path:
    print("Saving thumbnail to: " + path)
  dir = os.path.dirname(path)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  img.save(path)


# Modified
def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.
  Returns:
    The total number of WSI training slide images.
  """
  return len(get_training_slide_paths())


# Added
def get_training_slide_paths():
  return glob.glob(os.path.join(SRC_TRAIN_DIR, "*."+SRC_TRAIN_EXT))


# Expunged
def training_slide_range_to_images(start_ind, end_ind):
  """
  Convert a range of WSI training slides to smaller images (in a format such as jpg or png).
  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
  Returns:
    The starting index and the ending index of the slides that were converted.
  """
  for slide_num in range(start_ind, end_ind + 1):
    training_slide_to_image(slide_num)
  return (start_ind, end_ind)


# Addition - To replace `training_slide_range_to_images`
def training_slide_paths_to_images(paths_l):
	for path in paths_l:
		training_slide_to_image(path)


# Modified: Changed to slide_path based from slide_num based
def singleprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using a single process.
  """
  t = Time()

  training_paths = get_training_slide_paths()
  training_slide_paths_to_images(training_paths)

  t.elapsed_display()


# Modified
def multiprocess_training_slides_to_images():
	"""
	Convert all WSI training slides to smaller images using multiple processes (one process per core).
	Each process will process a range of slide numbers.
	"""
	timer = Time()

	# how many processes to use
	num_processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_processes)

	num_train_images = get_num_training_slides()
	if num_processes > num_train_images:
		num_processes = num_train_images
	images_per_process = (num_train_images // num_processes)+1

	print("Number of processes: " + str(num_processes))
	print("Number of training images: " + str(num_train_images))

	# Split training instances across processes
	training_paths = get_training_slide_paths()
	tasks = []
	for num_process in range(num_processes):
		start_idx = num_process*images_per_process
		end_idx = (num_process+1)*images_per_process
		tasks.append((training_paths[start_idx:end_idx],))

	# start tasks
	results = []
	for t in tasks:
		results.append(pool.apply_async(training_slide_paths_to_images, t))
		results[-1].get()

	timer.elapsed_display()


# ADDED Functions

def get_roi_image_result_filepath(slide_filepath):
	"""
	Convert slide number to the path to the file that is the final result of ROI extraction.
	Args:
	slide_number: The slide number.
	Returns:
	Path to the filter image file.
	"""
	slide_filename = ntpath.basename(slide_filepath).split('.')[0]
	img_path = os.path.join(
    ROI_DIR, 
    slide_filename,
    slide_filename + "-" + "region_{region_num}." + DEST_ROI_EXT
  )
	return img_path


def get_roi_image_result_dirpath(slide_filepath):
	"""
	Convert slide number to the path to the directory containing the final result of ROI extraction.
	Args:
	slide_number: The slide number.
	Returns:
	Path to the filter image file.
	"""
	slide_filename = ntpath.basename(slide_filepath).split('.')[0]
	img_path = os.path.join(
    ROI_DIR, 
    slide_filename
  )
	return img_path