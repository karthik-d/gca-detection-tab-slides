import os

config = dict()

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
    TRANSFORMS = [
        'img_to_tensor',
        'resize_to_input_shape'
    ]
))
