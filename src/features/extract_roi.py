from wsi import slide, utils, filters, roi

# slide.multiprocess_training_slides_to_images()
# slide.singleprocess_training_slides_to_images()
# filters.singleprocess_apply_filters_to_images()
# filters.multiprocess_apply_filters_to_images(display=False)

""" 
Set downscale_level to:
- 0, if ROIs must be from - TOP SLIDE (Highest Resolution) 
- 1, if ROIs must be from - x4 DOWNSCALED SLIDE
- 2, if ROIs must be from - x16 DOWNSCALED SLIDE
- 3, if ROIs must be from - x64 DOWNSCALED SLIDE
# NOTE: Only one of 0, 1, 2 or 3 must be specified. 
Other downscaling levels have been pruned to improve execution time.
"""
roi.multiprocess_extract_roi_from_filtered(downscale_level=3)
# roi.singleprocess_extract_roi_from_filtered(downscale_level=1)

# TODO: Fix extract_roi multiprocess
# TODO: Move global constants and config variables to config.py