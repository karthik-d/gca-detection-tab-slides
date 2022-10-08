# Pipeline the data procesing steps here, for each phase

from .preprocessor.augment import augment 
from .organizer.describe import describe_datafolder
from .organizer.split_for_experiment import split_for_experiment

def organize():
    pass

def preprocess():
    augment()

def describe():
    describe_datafolder()

def split_data():
    split_for_experiment()
    # split_for_training()