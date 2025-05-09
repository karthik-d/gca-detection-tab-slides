import multiprocessing
import numpy as np
import os 
import ntpath
from pathlib import Path
from skimage import measure, morphology, transform, draw, color

import matplotlib.pyplot as plot

from PIL import Image 
import tempfile

from wsi import slide, filters, utils
from wsi.utils import Time

REL_IMGS_DIR_NAME = "related-imgs"
REL_IMG_FORMAT = 'png'

THUMBNAIL_METANAME = 'thumbnail'
LABEL_METANAME = 'label'

ROI_BOUND_PAD_TOP = 20
ROI_BOUND_PAD_BOTTOM = 20
ROI_BOUND_PAD_LEFT = 20
ROI_BOUND_PAD_RIGHT = 20
ROI_BUFFER = 10


def make_temp_memarr_file(slide, level=0, mode='w+', dimensions=(None, None, None)):
	"""
	Create or truncate an existing memory-mapped array file	with required shape buffer
	Return its file pointer object
	"""
	# Extract configuration
	test_part = np.asarray(slide.read_region((0,0), level, size=(1,1)))
	dtype = test_part.dtype
	# Infer required dimensions
	x_dim, y_dim, op_channels = dimensions 
	if x_dim is None:
		x_dim = slide.level_dimensions[level][0]
	if y_dim is None:
		y_dim = slide.level_dimensions[level][1]
	if op_channels is None:
		op_channels = test_part.shape[2]
	shape = (x_dim, y_dim, op_channels)
	# Prepare file
	tempdir = tempfile.TemporaryDirectory()
	file_destn = os.path.join(tempdir.name, 'temp_')
	return np.memmap(file_destn, shape=shape, dtype=dtype, mode=mode)


def get_prescale_value_for_level(slide, level):
	return int(slide.level_downsamples[level]) if level!=0 else 1


def read_slide_level_in_parts(slide, level, part_size, channels=None, start_xy=None, end_xy=None):
	"""
	Generator to read the slide level in parts
	start_xy and end_xy are coordinates on level-0, conforming to OpenSlide conventions
	"""
	
	range_x, range_y = part_size
	prescale = get_prescale_value_for_level(slide, level)

	# If start and end xy are not specified, extract whole image
	if start_xy is None:
		start_x = 0
		start_x_downscaled = 0
	else:
		start_x = start_xy[0]
		start_x_downscaled = start_x//prescale

	if end_xy is None:
		end_x, end_y = slide.level_dimensions[0]
		end_x_downscaled, end_y_downscaled = slide.level_dimensions[level]
	else:
		end_x, end_y = end_xy 
		end_x_downscaled, end_y_downscaled = end_x//prescale, end_y//prescale

	extent_x = prescale*range_x
	extent_x_downscaled = range_x
	last_x = False
	# if start_xy is not None:
		# print(f"Limits: ({start_x_downscaled}, {start_xy[1]//prescale}) to ({end_x_downscaled}, {end_y_downscaled})")
	while (not last_x) or (extent_x>0 and (start_x+extent_x)<=end_x):
		
		# Include remainder patch
		if not last_x and ((start_x+extent_x)>=end_x or (start_x_downscaled+extent_x_downscaled)>=end_x_downscaled):
			extent_x = end_x - start_x
			extent_x_downscaled = end_x_downscaled - start_x_downscaled
			if extent_x<=0 or extent_x_downscaled<=0:
				break
			last_x = True

		# Iterate vertically
		if start_xy is None:
			start_y = 0
			start_y_downscaled = 0
		else:
			start_y = start_xy[1]
			start_y_downscaled = start_y//prescale

		extent_y = prescale*range_y
		extent_y_downscaled = range_y
		last_y = False

		while (not last_y) or (extent_y>0 and (start_y+extent_y)<=end_y):
			# Include remainder patch
			# print("DEBUG: ", (start_y+extent_y), end_y )
			if not last_y and ((start_y+extent_y)>=end_y or (start_y_downscaled+extent_y_downscaled)>=end_y_downscaled):
				extent_y = end_y - start_y
				extent_y_downscaled = end_y_downscaled - start_y_downscaled
				if extent_y<=0 or extent_y_downscaled<=0:
					break
				last_y = True
			# Extract part
			# print(f"Reading ({start_x_downscaled}, {start_y_downscaled}) to ({start_x_downscaled+extent_x_downscaled}, {start_y_downscaled+extent_y_downscaled}) ==> {last_x}, {last_y}")
			print(f"Reading ({start_x_downscaled}, {start_y_downscaled}) to ({start_x_downscaled+extent_x_downscaled}, {start_y_downscaled+extent_y_downscaled})")
			part_data = np.asarray(slide.read_region(
				(start_x, start_y), 
				level=level,
				size=(extent_x_downscaled, extent_y_downscaled)
			))
			# Covert to RGB if required
			if channels==3:
				part_data = utils.rgba_to_rgb(part_data, channel_axis=-1)
			# Run generator one step
			yield (
				np.transpose(part_data, (1, 0, 2)),
				last_x,
				last_y
			)
			start_y += extent_y
			start_y_downscaled += extent_y_downscaled
			
		# Next x-row
		start_x += extent_x
		start_x_downscaled += extent_x_downscaled
	

def extract_level_from_slide(slide, level=0, part_size=(2048, 2048), start_xy=None, end_xy=None):
	"""
	Extracts a specific magnification level from the slide object
	part_size is used to specify the chunks in which data is copied from the slide
	start_xy and end_xy are specified wrt slide-0, conforming to OpenSlide conventions
	NOTE: The RGBA image in .svs is converted down to 3-channel RGB during conversion - using alpha blending
	"""    
	
	prescale = get_prescale_value_for_level(slide, level)
	# Set x-dimension of result
	x_dim = None if (start_xy is None) or (end_xy is None) else (end_xy[0]-start_xy[0])
	x_dim = None if x_dim is None else (round(x_dim/prescale)+ROI_BUFFER)
	# Set y-dimension of result
	y_dim = None if (start_xy is None) or (end_xy is None) else (end_xy[1]-start_xy[1])
	y_dim = None if y_dim is None else (round(y_dim/prescale)+ROI_BUFFER)
	# Open accumulator file. Make memory-mapped array
	img_acc = make_temp_memarr_file(slide, level, dimensions=(x_dim, y_dim, 3))
	# Iterate y, for each iteration of x -- x and y are transposed w.r.t the read_region config
	store_x, store_y = (0, 0) 
	for part, last_x, last_y in read_slide_level_in_parts(slide, level, part_size, channels=3, start_xy=start_xy, end_xy=end_xy):
		img_acc[store_x:(store_x+part.shape[0]), store_y:(store_y+part.shape[1]), :] = part
		if last_y:
			store_y = 0
			store_x += part.shape[0]
		else:
			store_y += part.shape[1]
	# Retranspose the array
	img_acc = np.transpose(img_acc, (1, 0, 2))
	return img_acc


def get_roi_contours_from_image(np_img, close_neighborhood=(30,30), open_neighborhood=(16,16)):
	np_gray = filters.filter_grays(np_img, output_type='uint8')
	contour_level = (np.max(np_gray)-np.min(np_gray))/(2*255)
	# "close" to club nearby speckles, "open" to remove islands of speckles
	# Neighborhood can be large - hence, approximate - only extracting bounding boxes
	np_gray = filters.apply_binary_closing(np_gray, close_neighborhood)
	np_gray = filters.apply_binary_opening(np_gray, open_neighborhood)
	contours = measure.find_contours(np_gray, level=contour_level)
	return contours


def get_roi_boxes_from_image(np_img):
	""" 
	Return the ROI boxes from the image
	sorted as per the labelling order
	"""
	contours = get_roi_contours_from_image(np_img)
	# Make bounding boxes
	roi_boxes = []
	for contour in contours:
		X_min = int(np.min(contour[:,0]))
		X_max = int(np.max(contour[:,0]))
		Y_min = int(np.min(contour[:,1]))
		Y_max = int(np.max(contour[:,1]))
		roi_boxes.append([X_min, X_max, Y_min, Y_max])

	# Sort in labelling order
	sorted_idx = np.argsort(list(map(utils.roi_labelling_order, roi_boxes)), order=['vertical', 'horizontal'])
	return np.array(roi_boxes)[sorted_idx]


def save_roi_portions(slide_filepath, slide_obj, downscale_level, np_img, roi_boxes, padding=True):
	# Make result path skeleton
	base_img_path = slide.get_roi_image_result_filepath(slide_filepath)
	Path(ntpath.split(base_img_path)[0]).mkdir(
		parents=True,
		exist_ok=True
	)
	# box-coords are 90-deg clockwise rotated wrt np_img and slide
	level_0_x, level_0_y = slide_obj.level_dimensions[0]
	level_3_x, level_3_y = slide_obj.level_dimensions[3]
	# Extract each region from level-0 and save
	for serial, box in enumerate(roi_boxes, start=1):
		# Rotate bounding box - counter-clockwise 90-deg
		box = utils.rotate_bounding_box_anticlockwise_90(box, np_img.shape[:2])
		# Formatted as [X_min, X_max, Y_min, Y_max]
		# Apply padding and scale to level-0
		start_xy=(box[0], box[2])
		end_xy=(box[1], box[3])
		if padding:
			box[0] = utils.scale_value_between_dimensions(
				max(box[0]-ROI_BOUND_PAD_LEFT, 0),
				level_3_x,
				level_0_x
			)
			box[1] = utils.scale_value_between_dimensions(
				min(box[1]+ROI_BOUND_PAD_RIGHT, np_img.shape[1]-1),
				level_3_x,
				level_0_x
			)
			box[2] = utils.scale_value_between_dimensions(
				max(box[2]-ROI_BOUND_PAD_TOP, 0),
				level_3_y,
				level_0_y
			)
			box[3] = utils.scale_value_between_dimensions(
				min(box[3]+ROI_BOUND_PAD_BOTTOM, np_img.shape[0]-1),
				level_3_y,
				level_0_y 
			)
		else:
			box[0] = utils.scale_value_between_dimensions(
				box[0],
				level_3_x,
				level_0_x
			)
			box[1] = utils.scale_value_between_dimensions(
				box[1],
				level_3_x,
				level_0_x
			)
			box[2] = utils.scale_value_between_dimensions(
				box[2],
				level_3_y,
				level_0_y
			)
			box[3] = utils.scale_value_between_dimensions(
				box[3],
				level_3_y,
				level_0_y 
			)
		# Make PIL img and save
		start_xy=(box[0], box[2])
		end_xy=(box[1], box[3])
		print(f"\nExtracting ROI #{serial}...")
		np_result = extract_level_from_slide(slide_obj, level=downscale_level, start_xy=start_xy, end_xy=end_xy)
		Image.fromarray(np_result).save(base_img_path.format(region_num=serial), compression="tiff_lzw")
	print(f"\nExtracted and saved {len(roi_boxes)} ROI(s)")


def extract_roi_from_image(slide_filepath, downscale_level, save=False, display=False):
	# Load slide object
	slide_orig = slide.open_slide(slide_filepath)
	if slide_orig is None:
		print(f"Could not find {slide_filepath}")
		return None
	# Extract the 32x level i.e. level 3, locate the ROIs from it
	np_downscaled = extract_level_from_slide(slide_orig, level=3)
	np_downscaled_rot90 = utils.rotate_clockwise_90(np_downscaled)
	roi_boxes = get_roi_boxes_from_image(np_downscaled_rot90)
	# Extract ROI from full-resolution slide and save
	save_roi_portions(slide_filepath, slide_orig, downscale_level, np_downscaled, roi_boxes)
	save_wholeside_related_images(slide_filepath)


	# Display the image and plot all the contours found	
	# NOT FUNCTIONAL!
	if display:
		fig, ax = plot.subplots()
		ax.imshow(np_downscaled)	
		for box in roi_boxes:
			ax.plot(box[:, 1], contour[:, 0], linewidth=2)
		plot.show()

	return slide_filepath


def extract_roi_image_path_list(path_l, downscale_level, save=False, display=False):
	for slide_path in path_l:
		print("\nProcessing", slide_path)
		_ = extract_roi_from_image(slide_path, downscale_level, save, display)
	return path_l


def save_wholeside_related_images(slide_filepath, include_thumbnail=True, include_label=True):
	# Load slide object
	slide_orig = slide.open_slide(slide_filepath)
	if slide_orig is None:
		print(f"Could not find {slide_filepath}")
		return None
	# Locate ROI directory and make required subdirs
	roi_dir = slide.get_roi_image_result_dirpath(slide_filepath)
	related_imgs_dir = os.path.join(roi_dir, REL_IMGS_DIR_NAME)
	Path(related_imgs_dir).mkdir(
		parents=True,
		exist_ok=True
	)
	# Save reqd images, if they exist
	reqd_rel_imgs = []
	reqd_rel_imgs.append(THUMBNAIL_METANAME) if include_thumbnail else None
	reqd_rel_imgs.append(LABEL_METANAME) if include_label else None
	for img_metaname in reqd_rel_imgs:
		if isinstance(slide_orig.associated_images.get(img_metaname), Image.Image):
			save_path = ".".join([
				os.path.join(related_imgs_dir, img_metaname),
				REL_IMG_FORMAT
			])
			slide_orig.associated_images.get(img_metaname).save(
				fp=save_path,
				format=REL_IMG_FORMAT
			)
	print("Saved related images")


def singleprocess_extract_roi_from_filtered(downscale_level=1, save=False, display=False):

  t = Time()

  training_paths = slide.get_training_slide_paths()
  num_training_slides = len(training_paths)
  path_l = extract_roi_image_path_list(training_paths, downscale_level, save, display)

  print("\nTime taken to extract ROIs: %s\n" % str(t.elapsed()))


def multiprocess_extract_roi_from_filtered(downscale_level=1, save=False, display=False):

	timer = Time()

	if save and not os.path.exists(slide.FILTER_DIR):
		os.makedirs(slide.FILTER_DIR)

	# how many processes to use
	num_processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(num_processes)

	num_train_images = slide.get_num_training_slides()
	if num_processes > num_train_images:
		num_processes = num_train_images
	images_per_process = (num_train_images // num_processes)+1

	print("Number of processes: " + str(num_processes))
	print("Number of training images: " + str(num_train_images))

	training_paths = slide.get_training_slide_paths()
	tasks = []
	for num_process in range(num_processes):
		start_idx = num_process*images_per_process
		end_idx = (num_process+1)*images_per_process
		tasks.append((training_paths[start_idx:end_idx], downscale_level, save, display))

	# start tasks
	results = []
	for t in tasks:
		results.append(pool.apply_async(extract_roi_image_path_list, t))
		results[-1].get()

	print("\nTime taken to extract ROIs (multiprocess): %s\n" % str(timer.elapsed()))


# TIME-STATS
# morphology neighborhood Vs. time
# 50, 30					25s/img (x32)
# 40, 24					14s/img (x32)
# 30, 16					9s /img (x32)

# TODO: Display and Save intermediate results