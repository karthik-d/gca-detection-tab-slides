# from data.extract_svs import *
from metadata.generate_mappings import *
from metadata.describe import *
from metadata.utils import *
# from metadata.split_samples import *
from data.driver import *
from train.driver import *
from test.heldout_test import heldout_test
from test.inference_time import estimate_inference_time
from test.heldout_test_classwise import heldout_test_classwise
from test.make_eval_figures import eval_metrics
from visualize.driver import *



if __name__=='__main__':
	# generate_mappings_fsc()
	# split_samples()
	# preprocess()
	# train()
	# heldout_test()
	# estimate_inference_time(n_batches=16)
	# heldout_test_classwise()
	# eval_metrics()
	# fold_classnames()

	# print("========================== ALL ANALYSIS RELEVANT ==============")
	# describe_all_analysis_slides()
	# get_slides_list_minus('/home/miruna/.dumps/BAT-ACG/repo/dataset/slides-list-on-feb-7-2023.csv', '/home/miruna/.dumps/BAT-ACG/repo/dataset/required-slides-on-feb-7-2023.csv')
	# print("========================== ANNOTATED ==============")
	# describe_any_slide_list('../dataset/slides-list-on-feb-7-2023.csv')
	# print("========================== PENDING ==============")
	# describe_any_slide_list('/home/miruna/.dumps/BAT-ACG/repo/dataset/required-slides-on-feb-7-2023.csv')

	# assort_classwise()
	# describe_datafolder(to_file=False)
	# split_data()
	# apply_split(dry_run=True)
	# filter_data()

	# viz: (1) select rois to analyze; (2) run gradcam.
	# (1) specify required ROIs in `organizer/select_rois.py` -- they'll be copied to the required input directory.
	# (2) compute gradcam gradients and overlay on original ROI.
	# select_rois()
	visualize()

	print("ran!")