from torchvision import datasets
from collections import Counter

from .transforms import get_transforms
from ..config import config


def get_dataset(transforms={}):

    folder = datasets.ImageFolder(
        root=config.get('SRC_PATH'),
        transform=get_transforms(transforms)
    )

    print("--->", folder.find_classes(config.get('SRC_PATH')))
    print("Data composition:", dict(Counter(folder.targets)))

    return datasets.ImageFolder(
        root=config.get('SRC_PATH'),
        transform=get_transforms(transforms)
    )