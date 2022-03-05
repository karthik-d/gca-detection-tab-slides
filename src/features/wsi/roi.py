import multiprocessing
import numpy as np
import os 
import ntpath
from skimage import measure, morphology, transform
from skimage import draw

import matplotlib.pyplot as plot

from wsi import slide, filters, utils
from wsi.utils import Time

ROI_BOUND_PAD_TOP = 20
ROI_BOUND_PAD_BOTTOM = 20
ROI_BOUND_PAD_LEFT = 20
ROI_BOUND_PAD_RIGHT = 20


def get_roi_contours_from_image(np_img, close_neighborhood=(30,30), open_neighborhood=(16,16)):
	np_gray_rot90 = filters.filter_rgb_to_grayscale(np_img).T
	# "close" to club nearby speckles, "open" to remove islands of speckles
	# Neighborhood can be large - hence, approximate - only extracting bounding boxes
	np_gray_rot90 = filters.apply_binary_closing(np_gray_rot90, close_neighborhood)
	np_gray_rot90 = filters.apply_binary_opening(np_gray_rot90, open_neighborhood)
	contours = measure.find_contours(np_gray_rot90)
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


def extract_roi_from_image(slide_filepath, display=False):
	img_path = slide.get_filter_image_result_path(slide_filepath)
	np_orig = slide.open_image_np(img_path)
	# Find ROI boxes
	np_gray_rot90 = np.zeros(np_orig.shape[:2]).T
	roi_boxes = get_roi_boxes_from_image(np_orig)
	print(len(roi_boxes))
	roi_regions = []
	i = 1
	for box in roi_boxes:
		# Formatted as [X_min, X_max, Y_min, Y_max]
		# Apply padding
		box[0] = max(box[0]-ROI_BOUND_PAD_LEFT, 0)
		box[1] = min(box[1]+ROI_BOUND_PAD_RIGHT, np_orig.shape[1]-1)
		box[2] = max(box[2]-ROI_BOUND_PAD_TOP, 0)
		box[3] = min(box[3]+ROI_BOUND_PAD_BOTTOM, np_orig.shape[0]-1)
		roi_regions.append(np_orig[box[2]:box[3]+1, box[0]:box[1]+1, :])
		i += 1
		print("done")
	# save_roi_portions(np_orig)
	
	"""
	# Draw Bounding Boxes
	roi_boxes = []
	for contour in contours:
		X_min = max(np.min(contour[:,0])-ROI_BOUND_PAD_LEFT, 0)
		X_max = min(np.max(contour[:,0])+ROI_BOUND_PAD_RIGHT, np_gray_rot90.shape[0]-1)
		Y_min = max(np.min(contour[:,1])-ROI_BOUND_PAD_TOP, 0)
		Y_max = min(np.max(contour[:,1])+ROI_BOUND_PAD_BOTTOM, np_gray_rot90.shape[1]-1)
		roi_boxes.append([X_min, X_max, Y_min, Y_max])
		
	contours_sorted_idx = np.argsort(list(map(roi_labelling_order, roi_boxes)), order=['vertical', 'horizontal'])

	roi_boxes_sorted = np.array(roi_boxes)[contours_sorted_idx]
	for roi in roi_boxes_sorted:
		print(roi)
	"""

	"""
	with_boxes  = np.copy(np_gray_rot90)
	for box in roi_boxes:
		# Formatted as [X_min, X_max, Y_min, Y_max]
		r = [ box[0], box[1], box[1], box[0], box[0] ]
		c = [ box[3], box[3], box[2], box[2], box[3] ]
		rr, cc = polygon_perimeter(r, c, with_boxes.shape)
		with_boxes[rr, cc] = 1 
	"""

	plot.imshow(roi_regions[-1])
	plot.show()

	# Display the image and plot all contours found
	if display:
		fig, ax = plot.subplots()
		ax.imshow(np_gray_rot90, cmap=plot.cm.gray)	
		for contour in contours:
			ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
		plot.show()

	return slide_filepath


def extract_roi_image_path_list(path_l):
	for slide_path in path_l:
		_ = extract_roi_from_image(slide_path)
	return path_l


def singleprocess_extract_roi_from_filtered(save=True, display=False):

  t = Time()

  training_paths = slide.get_training_slide_paths()
  num_training_slides = len(training_paths)
  path_l = extract_roi_image_path_list(training_paths)

  print("Time taken to extract ROIs: %s\n" % str(t.elapsed()))


def multiprocess_extract_roi_from_filtered(save=True, display=False):

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
    tasks.append((training_paths[start_idx:end_idx], save, display))

  # start tasks
  results = []
  for t in tasks:
    results.append(pool.apply_async(apply_filters_to_image_path_list, t))

  html_page_info = dict()
  for result in results:
    (path_l, html_page_info_res) = result.get()
    html_page_info.update(html_page_info_res)

  print("Time taken to extract ROIs (multiprocess): %s\n" % str(timer.elapsed()))