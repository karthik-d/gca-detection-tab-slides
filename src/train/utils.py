import operator
import random
import time
from pathlib import Path
from typing import (Dict, IO, List, Tuple)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import (datasets, transforms)

def get_printable_confusion_matrix(
    all_labels,
    all_predictions,
    classes
):
    """
    Renders a `printable` confusion matrix
    Uses the pandas display utility
    """

    map_classes = {
        x: class_ 
        for x, class_ in enumerate(classes)
    }

    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 0

    truth = pd.Series(
        pd.Categorical(
            pd.Series(all_labels).replace(map_classes), 
            categories=classes
        ),
        name="Truth"
    )

    prediction = pd.Series(
        pd.Categorical(
            pd.Series(all_predictions).replace(map_classes), 
            categories=classes
        ),
        name="Prediction"
    )

    confusion_matrix = pd.crosstab(
        index=truth, 
        columns=prediction, 
        normalize="index", 
        dropna=False
    )
    confusion_matrix.style.hide(axis='index')
    return f'\n{confusion_matrix}\n'

'''
class Random90Rotation:
    def __init__(self, degrees: Tuple[int] = None) -> None:
        """
        Randomly rotate the image for training. Credits to Naofumi Tomita.

        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.

        Args:
            im: The image to rotate.

        Returns:    
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])


# def create_model(num_layers: int, num_classes: int,
#                  pretrain: bool) -> torchvision.models.resnet.ResNet:
#     """
#     Instantiate the ResNet model.

#     Args:
#         num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
#         num_classes: Number of classes in the dataset.
#         pretrain: Use pretrained ResNet weights.

#     Returns:
#         The instantiated ResNet model with the requested parameters.
#     """
#     assert num_layers in (
#         18, 34, 50, 101, 152
#     ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
#     model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
#     model = model_constructor(num_classes=num_classes)

#     if pretrain:
#         pretrained = model_constructor(pretrained=True).state_dict()
#         if num_classes != pretrained["fc.weight"].size(0):
#             del pretrained["fc.weight"], pretrained["fc.bias"]
#         model.load_state_dict(state_dict=pretrained, strict=False)
#     return model


def get_data_transforms(color_jitter_brightness: float,
                        color_jitter_contrast: float,
                        color_jitter_saturation: float,
                        color_jitter_hue: float, path_mean: List[float],
                        path_std: List[float]
                        ) -> Dict[str, torchvision.transforms.Compose]:
    """
    Sets up the dataset transforms for training and validation.

    Args:
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.

    Returns:
        A dictionary mapping training and validation strings to data transforms.
    """
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.ColorJitter(brightness=color_jitter_brightness,
                                   contrast=color_jitter_contrast,
                                   saturation=color_jitter_saturation,
                                   hue=color_jitter_hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ])
    }


## PRUNE THIS
def print_params(train_folder: Path, num_epochs: int, num_layers: int,
                 learning_rate: float, batch_size: int, weight_decay: float,
                 learning_rate_decay: float, resume_checkpoint: bool,
                 resume_checkpoint_path: Path, save_interval: int,
                 checkpoints_folder: Path, pretrain: bool,
                 log_csv: Path) -> None:
    """
    Print the configuration of the model.

    Args:
        train_folder: Location of the automatically built training input folder.
        num_epochs: Number of epochs for training.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        learning_rate: Learning rate to use for gradient descent.
        batch_size: Mini-batch size to use for training.
        weight_decay: Weight decay (L2 penalty) to use in optimizer.
        learning_rate_decay: Learning rate decay amount per epoch.
        resume_checkpoint: Resume model from checkpoint file.
        resume_checkpoint_path: Path to the checkpoint file for resuming training.
        save_interval: Number of epochs between saving checkpoints.
        checkpoints_folder: Directory to save model checkpoints to.
        pretrain: Use pretrained ResNet weights.
        log_csv: Name of the CSV file containing the logs.
    """
    print(f"train_folder: {train_folder}\n"
          f"num_epochs: {num_epochs}\n"
          f"num_layers: {num_layers}\n"
          f"learning_rate: {learning_rate}\n"
          f"batch_size: {batch_size}\n"
          f"weight_decay: {weight_decay}\n"
          f"learning_rate_decay: {learning_rate_decay}\n"
          f"resume_checkpoint: {resume_checkpoint}\n"
          f"resume_checkpoint_path (only if resume_checkpoint is true): "
          f"{resume_checkpoint_path}\n"
          f"save_interval: {save_interval}\n"
          f"output in checkpoints_folder: {checkpoints_folder}\n"
          f"pretrain: {pretrain}\n"
          f"log_csv: {log_csv}\n\n")
'''

def render_verbose_props(title, *args, **kwargs):
    print("\n--------------------------")
    print(title)
    print("--------------------------")

    # Render direct properties
    for value in args:
        print(value)

    print("--------------------------")

    # Render key-value based properties
    for key, value in kwargs.items():
        print(f"{key}: {value}")


###########################################
#          MAIN TRAIN FUNCTION            #
###########################################

'''
def parse_val_acc(model_path: Path) -> float:
    """
    Parse the validation accuracy from the filename.

    Args:
        model_path: The model path to parse for the validation accuracy.

    Returns:
        The parsed validation accuracy.
    """
    return float(
        f"{('.'.join(model_path.name.split('.')[:-1])).split('_')[-1][2:]}")


def get_best_model(checkpoints_folder: Path) -> str:
    """
    Finds the model with the best validation accuracy.

    Args:
        checkpoints_folder: Folder containing the models to test.

    Returns:
        The location of the model with the best validation accuracy.
    """
    return max({
        model: parse_val_acc(model_path=model)
        for model in [m for m in checkpoints_folder.rglob("*.pt") if ".DS_Store" not in str(m)]
    }.items(),
               key=operator.itemgetter(1))[0]


def get_predictions(patches_eval_folder: Path, output_folder: Path,
                    checkpoints_folder: Path, auto_select: bool,
                    eval_model: Path, device: torch.device, classes: List[str],
                    num_classes: int, path_mean: List[float],
                    path_std: List[float], num_layers: int, pretrain: bool,
                    batch_size: int, num_workers: int) -> None:
    """
    Main function for running the model on all of the generated patches.

    Args:
        patches_eval_folder: Folder containing patches to evaluate on.
        output_folder: Folder to save the model results to.
        checkpoints_folder: Directory to save model checkpoints to.
        auto_select: Automatically select the model with the highest validation accuracy,
        eval_model: Path to the model with the highest validation accuracy.
        device: Device to use for running model.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        pretrain: Use pretrained ResNet weights.
        batch_size: Mini-batch size to use for training.
        num_workers: Number of workers to use for IO.
    """
    # Initialize the model.
    model_path = get_best_model(
        checkpoints_folder=checkpoints_folder) if auto_select else eval_model

    model = create_model(num_classes=num_classes,
                         num_layers=num_layers,
                         pretrain=pretrain)
    ckpt = torch.load(f=model_path)
    model.load_state_dict(state_dict=ckpt["model_state_dict"])
    model = model.to(device=device)

    model.train(mode=False)
    print(f"model loaded from {model_path}")

    # For outputting the predictions.
    class_num_to_class = {i: classes[i] for i in range(num_classes)}

    start = time.time()
    # Load the data for each folder.
    image_folders = get_subfolder_paths(folder=patches_eval_folder)

    # Where we want to write out the predictions.
    # Confirm the output directory exists.
    output_folder.mkdir(parents=True, exist_ok=True)

    # For each WSI.
    for image_folder in image_folders:

        # Temporary fix. Need not to make folders with no crops.
        try:
            # Load the image dataset.
            dataloader = torch.utils.data.DataLoader(
                dataset=datasets.ImageFolder(
                    root=str(image_folder),
                    transform=transforms.Compose(transforms=[
                        transforms.ToTensor(),
                        transforms.Normalize(mean=path_mean, std=path_std)
                    ])),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)
        except RuntimeError:
            print(
                "WARNING: One of the image directories is empty. Skipping this directory."
            )
            continue

        num_test_image_windows = len(dataloader) * batch_size

        # Load the image names so we know the coordinates of the patches we are predicting.
        image_folder = image_folder.joinpath(image_folder.name)
        window_names = get_image_paths(folder=image_folder)

        print(f"testing on {num_test_image_windows} crops from {image_folder}")

        with output_folder.joinpath(f"{image_folder.name}.csv").open(
                mode="w") as writer:

            writer.write("x,y,prediction,confidence\n")

            # Loop through all of the patches.
            for batch_num, (test_inputs, test_labels) in enumerate(dataloader):
                batch_window_names = window_names[batch_num *
                                                  batch_size:batch_num *
                                                  batch_size + batch_size]

                confidences, test_preds = torch.max(nn.Softmax(dim=1)(model(
                    test_inputs.to(device=device))),
                                                    dim=1)
                for i in range(test_preds.shape[0]):
                    # Find coordinates and predicted class.
                    xy = batch_window_names[i].name.split(".")[0].split(";")

                    writer.write(
                        f"{','.join([xy[0], xy[1], f'{class_num_to_class[test_preds[i].data.item()]}', f'{confidences[i].data.item():.5f}'])}\n"
                    )

    print(f"time for {patches_eval_folder}: {time.time() - start:.2f} seconds")
'''