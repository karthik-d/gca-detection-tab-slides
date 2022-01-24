import os

config = {
    'DATA_PATH': os.path.abspath(os.path.join(__file__,
        os.path.pardir, 
        os.path.pardir, 
        'dataset', 
        'data'
    )),
}