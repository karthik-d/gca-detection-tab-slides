from wsi import slide, utils, filters, roi

# slide.multiprocess_training_slides_to_images()
# slide.singleprocess_training_slides_to_images()
# filters.singleprocess_apply_filters_to_images()
# filters.multiprocess_apply_filters_to_images(display=False)
roi.singleprocess_extract_roi_from_filtered()
# roi.multiprocess_extract_roi_from_filtered()

# TODO: Use .tiff+lzw in place of .png
# TODO: Move global constants and config variables to config.py
# TODO: Change directory prefix and suffix
# TODO: Split boolean arg `save` into save_intermediate and save_final
# TODO: Add 'level' arg to find_contours() method