# from data.extract_svs import *
from metadata.generate_mappings import *
from metadata.describe import *
from metadata.utils import *
# from metadata.split_samples import *
from data.driver import *
from train.driver import *
from test.heldout_test import heldout_test
from test.heldout_test_classwise import heldout_test_classwise
from test.make_eval_figures import eval_metrics
from visualize.driver import *

# generate_mappings_fsc()
# split_samples()
# preprocess()
# train()
# heldout_test()
# heldout_test_classwise()
# eval_metrics()
# fold_classnames()

# print("========================== ALL ANALYSIS RELEVANT ==============")
# describe_all_analysis_slides()
# get_slides_list_minus('/home/miruna/.dumps/BAT-ACG/repo/dataset/slides-list-on-feb-7-2023.csv', '/home/miruna/.dumps/BAT-ACG/repo/dataset/required-slides-on-feb-7-2023.csv')
# print("========================== ANNOTATED ==============")
# describe_any_slide_list('/home/miruna/.dumps/BAT-ACG/repo/dataset/slides-list-on-feb-7-2023.csv')
# print("========================== PENDING ==============")
# describe_any_slide_list('/home/miruna/.dumps/BAT-ACG/repo/dataset/required-slides-on-feb-7-2023.csv')

# assort_classwise()
# describe_datafolder(to_file=False)
# split_data()
# apply_split()
# filter_data()
visualize()

print("ran!")