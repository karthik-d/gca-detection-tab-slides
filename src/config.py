import os

"""
Dictionary of configuration parameters
"""

config = {
    'DATA_PATH': os.path.abspath(os.path.join(__file__,
        os.path.pardir, 
        os.path.pardir, 
        'dataset', 
        'data'
    )),
    'METADATA_PATH': os.path.abspath(os.path.join(__file__,
        os.path.pardir, 
        os.path.pardir, 
        'dataset'
    )),
}