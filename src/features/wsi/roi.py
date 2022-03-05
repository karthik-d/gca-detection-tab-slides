import multiprocessing
import numpy as np
import os 
import ntpath
from skimage import measure, morphology, transform
from skimage.draw import polygon_perimeter

import matplotlib.pyplot as plot

from wsi import slide, filters, utils
from wsi.utils import Time

ROI_BOUND_PAD_TOP = 20
ROI_BOUND_PAD_BOTTOM = 20
ROI_BOUND_PAD_LEFT = 20
ROI_BOUND_PAD_RIGHT = 20


def roi_labelling_order(box_extents):
	"""
	Returns a sort key for boxes, with the descending precedence [vertical_posn, horizontal_posn]
	For use with np.argsort() - named fields
	Leverages Python's tuple sort logic
	- box_extents : [X_min, X_max, Y_min, Y_max]
	"""
	sort_key = np.array(
		(box_extents[2], box_extents[0]),
		dtype=[('vertical', 'i2'),('horizontal', 'i2')]
	)
	return sort_key



def extract_roi_from_image(slide_filepath, display=False, save=False):
	img_path = slide.get_filter_image_result_path(slide_filepath)
	np_orig = slide.open_image_np(img_path)
	np_gray_rot90 = filters.filter_rgb_to_grayscale(np_orig).T
	# "close" to club nearby speckles, "open" to remove islands of speckles
	# Neighborhood can be large - hence, approximate - only extracting bounding boxes
	np_gray_rot90 = filters.apply_binary_closing(np_gray_rot90, (30,30))
	np_gray_rot90 = filters.apply_binary_opening(np_gray_rot90, (16,16))
	contours = measure.find_contours(np_gray_rot90)
	# Draw Bounding Boxes
	bounding_boxes = []
	for contour in contours:
		X_min = max(np.min(contour[:,0])-ROI_BOUND_PAD_LEFT, 0)
		X_max = min(np.max(contour[:,0])+ROI_BOUND_PAD_RIGHT, np_gray_rot90.shape[0]-1)
		Y_min = max(np.min(contour[:,1])-ROI_BOUND_PAD_TOP, 0)
		Y_max = min(np.max(contour[:,1])+ROI_BOUND_PAD_BOTTOM, np_gray_rot90.shape[1]-1)
		bounding_boxes.append([X_min, X_max, Y_min, Y_max])
		
	contours_sorted_idx = np.argsort(list(map(roi_labelling_order, bounding_boxes)), order=['vertical', 'horizontal'])
	print(contours_sorted_idx)
	contours_sorted = contours[contours_sorted_idx]
	for contour in contours_sorted:
		pass

	with_boxes  = np.copy(np_gray_rot90)
	for box in bounding_boxes:
		# Formatted as [X_min, X_max, Y_min, Y_max]
		r = [ box[0], box[1], box[1], box[0], box[0] ]
		c = [ box[3], box[3], box[2], box[2], box[3] ]
		rr, cc = polygon_perimeter(r, c, with_boxes.shape)
		with_boxes[rr, cc] = 1 

	plot.imshow(with_boxes, cmap=plot.cm.gray)
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