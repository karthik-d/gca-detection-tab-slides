"""
Applies augmentation to increase the dataset size 
for specified `class`es within the 
given `SRC_PATH` directory, and stores the increased set (including the original)
into `DESTN_PATH` 
"""

#TODO: Use global config for setting paths and other parameters

import os
import glob
import ntpath
from pathlib import Path
from PIL import Image
from torchvision import transforms

from .lib import custom_transforms
from ..config import config


probabilistic_augs = [
    'vertical_flip',
    'horizontal_flip',
    'random_hgram_equalize'
]

name_augmentation_map = dict(
    vertical_flip = transforms.RandomVerticalFlip,
    horizontal_flip = transforms.RandomHorizontalFlip,
    rotate_90 = custom_transforms.RandomRotate90,
    rotate_acute = transforms.RandomRotation,
    random_hgram_equalize = transforms.RandomEqualize,
    color_jitter = transforms.ColorJitter
)

"""
# PARAMETERS TO SET in [ROOT]/src/data/config.py: 
# - AUGMENTATIONS
# - AUGMENTATION_PARAMS <for corresponding augmentation methods> as a dict(<param>=<value>, ...)

# SET THESE PARAMETERS HERE BELOW

# SET PATHS 
# - Relative to "[ROOT]/dataset/data/roi"  using `os.path.join(config.get('ROI_PATH'), <rel_path>)`
# - Absolute path from '/' (for *nix) OR '<Drive:>' (for windows)
"""

#--enter
SRC_PATH = os.path.abspath(os.path.join(
    config.get('ROI_PATH'),
    'ds_phase_1',
    'splits',
    'train'
))

#--enter
DESTN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    config.get('ROI_PATH'),
    'ds_phase_2',
    'splits',
    'train'
))

#--enter
classes_to_extract = ['Y', 'N']


# ------------------------------

def augment():

    # Ensure valid augmentation config
    augmentation_names = config.get('AUGMENTATIONS')
    assert len(augmentation_names)==len(config.get('AUGMENTATION_PARAMS')), "Parameters missing for one/more augmentations"

    # Enforce all augmentations - setting application probability to 1.0
    augmentation_params = config.get('AUGMENTATION_PARAMS')
    for idx, kwargs in enumerate(augmentation_params):
        if augmentation_names[idx] in probabilistic_augs:
            kwargs.update(dict(p=1.0))
            augmentation_params[idx] = kwargs

    # Instantiate required augmentations
    augmentation_functors = [
        name_augmentation_map.get(aug_key)(**aug_params)
        for aug_key, aug_params in zip(
            augmentation_names,
            augmentation_params
        )
    ]


    class_path_map = {
        class_: os.path.join(DESTN_PATH, class_) 
        for class_ in classes_to_extract
    }

    # Create destination template
    for class_, path_ in class_path_map.items():
        Path(path_).mkdir(
            parents=True,
            exist_ok=False
        )

    print("\nDestination template created")


    # Augment and store results
    for class_name in os.listdir(SRC_PATH):
        
        # check if valid class name
        if class_name in classes_to_extract:

            class_path = os.path.join(SRC_PATH, class_name)
            #  Pass all .tiff files through the aug pipeline
            for filepath in glob.glob(os.path.join(class_path, "*.tiff")):
                filename = ntpath.basename(filepath).split('.')[0]

                orig_img = Image.open(os.path.join(class_path, f"{filename}.tiff"))
                img_ctr = 0
                # Apply and save all images
                for idx, augmentor in enumerate(augmentation_functors):
                    
                    # Apply augmentation
                    img = augmentor(orig_img)

                    # Save transformed image
                    save_name = f"{filename}-{augmentation_names[idx]}.tiff"
                    img.save(
                        fp=os.path.join(
                            class_path_map.get(class_name),
                            save_name
                        ),
                        format='TIFF'
                    )
                    img_ctr += 1
            
                # Save original
                orig_img.save(
                    fp=os.path.join(
                        class_path_map.get(class_name),
                        f"{filename}.tiff"
                    ),
                    format='TIFF'
                )
                img_ctr += 1

                print(f"\nSaved {img_ctr} files for {class_name}/{filename}")






