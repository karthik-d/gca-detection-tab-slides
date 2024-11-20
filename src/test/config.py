import os

from config import config as root_config


config = dict()
config.update(root_config)

config.update(dict(
    INPUT_XY = (512, 512),
    INPUT_CHANNELS = 3
))

config.update(dict(
    INPUT_SHAPE = (*config.get('INPUT_XY'), config.get('INPUT_CHANNELS'))
))

config.update(dict(
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*2)))
))

config.update(dict(
    SRC_PATH = os.path.join(
        config.get('ROOT_PATH'), 
        'dataset', 
        'data', 
        'roi',
        'ds_phase_4',
        'splits',
        'valid'
    ),
    # Set as `None` to use imagenet weights
    CHECKPOINT_FILEPATH = os.path.join(
        config.get('LOGS_PATH'),
        'train',
        'experiment_3',
        'run_1',
        'epoch#3_val_acc#0-9395.ckpt'
    )
))

config.update(dict(
    TRANSFORMS = [
        'img_to_tensor',
        'resize_to_input_shape'
    ]
))

# allowed values: (18, 34, 50, 101, 152)
config.update(dict(
    RESNET_NLAYERS = 34
))
