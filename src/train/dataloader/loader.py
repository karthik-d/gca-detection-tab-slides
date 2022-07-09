from torchvision import datasets
from collections import Counter

from .transforms import get_transforms_for
from ..config import config, split_key_map

def get_dataset_for(split="train", transforms={}):

    path_key = "_".join([
        split_key_map.get(split),
        'PATH'
    ]) 

    folder = datasets.ImageFolder(
        root=config.get(path_key),
        transform=get_transforms_for(split, transforms)
    )

    print("--->", folder.find_classes(config.get(path_key)))
    print(split, "composition:", dict(Counter(folder.targets)))

    return datasets.ImageFolder(
        root=config.get(path_key),
        transform=get_transforms_for(split, transforms)
    )