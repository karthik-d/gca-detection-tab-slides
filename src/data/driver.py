# Pipeline the data procesing steps here, for each phase

"""
Data split sequence: assort -> split -> train
"""

from .preprocessor.augment import augment 
from .organizer.describe import describe_datafolder as _describe_datafolder
from .organizer.fold_classnames import fold_classnames as _fold_classnames
from .organizer.assort_classwise import assort_classwise as _assort_classwise
from .organizer.split_pooled import split_pooled
from .organizer.split_chronological import split_chronological
from .organizer.filters import filter_by_roiname

def organize():
    pass

def preprocess():
    augment()

def describe_datafolder(data_path=None, to_file=True, display=True):
    _describe_datafolder(data_path, to_file, display)

def fold_classnames():
    _fold_classnames()

def assort_classwise():
    assort_classwise()

def split_data():
    # split_pooled()
    split_chronological()

def filter_data(roi_names_file=None):
    if roi_names_file is None:
        roi_names_file = "/home/miruna/.dumps/BAT-ACG/repo/src/data/organizer/filenames.txt"
    filter_by_roiname(roi_names_file, False)