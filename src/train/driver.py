# Configure and pipline the trainining process, here

import os

from .architectures.resnet import train_driver
from .config import config

def train():

    train_driver(
        classes=['P', 'N'],
        resnet_layers=18,
        checkpoint_resumepath=None
    )

    # train_driver(
    #     classes=['P', 'N'],
    #     resnet_layers=18,
    #     checkpoint_resumepath=os.path.join(
    #         config.get('TRAIN_LOGS_PATH'),
    #         'experiment_2',
    #         'run_9',
    #         'epoch#1_val_acc#0-5147.ckpt'
    #     )
    # )