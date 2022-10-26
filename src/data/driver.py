# Pipeline the data procesing steps here, for each phase

from .preprocessor.augment import augment 
from .organizer.describe import describe_datafolder
from .organizer.split_for_experiment import split_for_experiment
from .organizer.filters import filter_by_roiname

def organize():
    pass

def preprocess():
    augment()

def describe():
    describe_datafolder()

def split_data():
    split_for_experiment()
    # split_for_training()

def filter_data():
    roi_names_file = "/home/miruna/.dumps/BAT-ACG/repo/src/data/organizer/filenames.txt"
    filter_by_roiname(roi_names_file, False)