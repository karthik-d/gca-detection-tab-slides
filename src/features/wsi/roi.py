import multiprocessing
import numpy as np
import os 
import ntpath

import matplotlib.pyplot as plot

from wsi import slide, utils
from wsi.utils import Time


def extract_roi_from_image(slide_filepath):
	img_path = slide.get_filter_image_result_path(slide_filepath)
	np_orig = slide.open_image_np(img_path)
	plot.imshow(np_orig)
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