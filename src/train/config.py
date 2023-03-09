import os
import glob
from pathlib import Path

from config import config as root_config

"""
HELPER Functions
"""

def infer_last_experiment_num(train_logs_path):
    max_exp_num = 0
    for file_ in glob.glob(os.path.join(train_logs_path, "*experiment_*")):
        if os.path.isdir(file_):
            try:
                exp_num = int(file_.split("_")[-1])
            except Exception as e:
                print("Warning:", e)
                exp_num = 0
            if exp_num > max_exp_num:
                max_exp_num = exp_num 
    return max_exp_num

def infer_last_run_num(exp_logs_path):
    max_run_num = 0
    # Empty loop, if experiment folder doesn't exist
    for file_ in glob.glob(os.path.join(exp_logs_path, "*run_*")):
        if os.path.isdir(file_):
            try:
                run_num = int(file_.split("_")[-1])
            except Exception as e:
                print("Warning:", e)
                run_num = 0
            if run_num > max_run_num:
                max_run_num = run_num 
    return max_run_num


config = dict()
config.update(root_config)

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
    DATA_PATH = os.path.join(
        root_config.get('ROOT_PATH'), 
        'dataset', 
        'data', 
        'roi', 
        'ds_phase_3', 
        'splits'
    )
))

config.update(dict(
    TRAIN_PATH = os.path.join(config.get('DATA_PATH'), 'train'),
    VALID_PATH = os.path.join(config.get('DATA_PATH'), 'valid')
))


# Set run-time transforms here
# For a list of available transforms, see `name_transform_map` in ./dataloader/transforms.py
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


# Logger configuration

config.update(dict(
    TRAIN_LOGS_PATH = os.path.join(
        root_config.get('LOGS_PATH'),
        'train'
    )
))

# Set manually to override the default experiment number
config.update(dict(
    #EXPERIMENT_NUM = infer_last_experiment_num(config.get('TRAIN_LOGS_PATH')) + 1
    EXPERIMENT_NUM = 3
))
config.update(dict(
    EXPERIMENT_LOGS_PATH = os.path.join(
        config.get('TRAIN_LOGS_PATH'),
        f"experiment_{config.get('EXPERIMENT_NUM')}"
    )
))


# Set manually to override the default run number
config.update(dict(
    # RUN_NUM = infer_last_run_num(config.get('EXPERIMENT_LOGS_PATH')) + 1
    RUN_NUM = 1
))
config.update(dict(
    RUN_CHECKPOINT_PATH = os.path.join(
        config.get('EXPERIMENT_LOGS_PATH'),
        f"run_{config.get('RUN_NUM')}"
    )
))

# Ensure that experiment directories exist
Path(config.get('RUN_CHECKPOINT_PATH')).mkdir(
    parents=True,
    exist_ok=True
)