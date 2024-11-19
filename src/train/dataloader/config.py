import os

config = dict()

split_key_map = {
    'train': 'TRAIN',
    'valid': 'VALID'
}

config.update(dict(
    INPUT_XY = (512, 512),
    INPUT_CHANNELS = 3
))

config.update(dict(
    INPUT_SHAPE = (*config.get('INPUT_XY'), config.get('INPUT_CHANNELS'))
))

config.update(dict(
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3)))
))

config.update(dict(
    DATA_PATH = os.path.join(config.get('ROOT_PATH'), 'dataset', 'data', 'roi', 'ds_phase_2', 'splits')
))

config.update(dict(
    TRAIN_PATH = os.path.join(config.get('DATA_PATH'), 'train'),
    VALID_PATH = os.path.join(config.get('DATA_PATH'), 'valid')
))

config.update(dict(
    TRAIN_TRANSFORMS = [
        'img_to_tensor',
        'resize_to_input_shape'
    ],
    VALID_TRANSFORMS = [
        'img_to_tensor',
        'resize_to_input_shape'
    ]
))
