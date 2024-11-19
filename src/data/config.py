import os

config = dict()

config.update(dict(
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*2)))
))

config.update(dict(
    DATA_PATH = os.path.join(config.get('ROOT_PATH'), 'dataset', 'data')
))

config.update(dict(
    ROI_PATH = os.path.join(config.get('ROOT_PATH'), 'dataset', 'data', 'roi')
))

# Set pre-train augmentations here
# For a list of available transforms, see `name_augmentation_map` in ./preprocessor/augment.py

# All augmentations correspond to PyTorch's random augmentations - EXCEPT `rotate_90` which is defined in this codebase
config.update(dict(
    AUGMENTATIONS = [
        'vertical_flip',
        'horizontal_flip',
        'rotate_90',
        'rotate_acute',
        'random_hgram_equalize',
        'color_jitter'
    ],
    AUGMENTATION_PARAMS = [
        dict(),
        dict(),
        dict(
            allow_clockwise=True,
            allow_counter_clockwise=True
        ),
        dict(
            degrees=(-10, 10)
        ),
        dict(),
        dict(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5
        )
    ]
))