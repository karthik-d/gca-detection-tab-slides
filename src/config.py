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

# Logger configurations

config.update(dict(
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*1)))
))

config.update(dict(
    LOGS_PATH = os.path.join(
        config.get('ROOT_PATH'),
        'logs'
    )
))