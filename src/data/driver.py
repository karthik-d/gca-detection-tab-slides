# Pipeline the data procesing steps here, for each phase

"""
Data split sequence: 
    - (optional) assort: produces DESTN from SRC
    - split: generates CSV with filepaths references to SRC
    - apply_split: generates split directories in DESTN using CSV in SRC
"""

from .preprocessor.augment import augment 
from .organizer.describe import describe_datafolder as _describe_datafolder
from .organizer.fold_classnames import fold_classnames as _fold_classnames
from .organizer.assort_classwise import assort_classwise as _assort_classwise
from .organizer.split_pooled import split_pooled
from .organizer.split_chronological import split_chronological
from .organizer.assort_splitwise import assort_splitwise
from .organizer.filters import filter_by_roiname
from .organizer.select_rois import select_rois as select_rois_

def organize():
    pass

def preprocess():
    augment()

def describe_datafolder(data_path=None, to_file=True, display=True):
    _describe_datafolder(data_path, to_file, display)

def fold_classnames():
    _fold_classnames()

def assort_classwise():
    _assort_classwise()

def split_data():
    # split_pooled()
    split_chronological()

def apply_split(dry_run=False):
    assort_splitwise(dry_run)

def filter_data(roi_names_file=None):
    if roi_names_file is None:
        roi_names_file = "/home/miruna/.dumps/BAT-ACG/repo/src/data/organizer/filenames.txt"
    filter_by_roiname(roi_names_file, False)


def select_rois():
	select_rois_()