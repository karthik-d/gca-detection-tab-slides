# Pipeline the data procesing steps here, for each phase

from .preprocessor.augment import augment 
from .organizer.describe import describe_datafolder

def organize():
    pass

def preprocess():
    augment()

def describe():
    describe_datafolder()