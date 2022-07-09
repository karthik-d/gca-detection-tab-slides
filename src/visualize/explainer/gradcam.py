"""
Applies GradCAM inference to all images 
in the given `SRC_PATH` directory formatted as, 

    [SRC_DIR]/
        - Y/
            *.tiff
            .
            .
        - N/
            *.tiff
            .
            .

and stores the resulting image-overlayed heatmaps
into `DESTN_PATH` 
based on the ResNet-18 model weights in the `CHECKPOINT_FILEPATH`
"""

## SET THE 3 PARAMETERS in [ROOT]/src/visualize/config.py: 

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
from PIL import Image
import numpy as np
import os

from train.architectures import resnet
from ..dataloader import loader
from .. import utils
from ..config import config


def img_of_tensor(img_tensor):
    """
    Tensor contains: batchsize, channels, width, height 
    Returns in CV2 format: width, height, channels
    """

    img = img_tensor.numpy()[0]
    img = np.transpose(img, [1, 2, 0])
    return img


def prepare_inputs(num_workers=8):

    image_dataset = loader.get_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset=image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    classes = image_dataset.find_classes(image_dataset.root)[0]

    return dataloader, classes


def render_inputs(dataloader):
    for (vis_x, vis_y), (img_path, _) in zip(iter(dataloader), dataloader.dataset.samples):
        yield vis_x, vis_y, img_path



def save_visualization():
    """
    # Driver function to generate and save visualizations
    """

    model = resnet.prepare_load_model(
        num_layers=18,
        num_classes=2,
        pretrain=True
    )

    # Load checkpoint file
    checkpoint_resumepath = config.get('CHECKPOINT_FILEPATH')
    if checkpoint_resumepath is not None:
        checkpoint = torch.load(f=checkpoint_resumepath)
        model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Using weights from {checkpoint_resumepath} - Epoch: {start_epoch}")
    else:
        print(f"Using IMAGENET weights")

    # target_layers = [model.layer4[-1]]
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    dataloader, classes = prepare_inputs()
    # Classes to activate the visualization for
    targets_0 = [ ClassifierOutputTarget(0) ]
    targets_1 = [ ClassifierOutputTarget(1) ]

    for ip_tensor, op_tensor, img_path in render_inputs(dataloader):

        if(op_tensor.item()!=1):
            continue

        img_name = os.path.basename(img_path)
        # aug_smooth=True and eigen_smooth=True
        grayscale_cam_0 = cam(input_tensor=ip_tensor, targets=targets_0)[0, :]
        grayscale_cam_1 = cam(input_tensor=ip_tensor, targets=targets_1)[0, :]

        # rgb_img = Image.open(img_path)
        rgb_img = img_of_tensor(ip_tensor)

        overlayed_img = utils.make_detailed_overlay_img(
            rgb_img, 
            grayscale_cam_0,
            grayscale_cam_1,
            label=f"Actual Class: {classes[op_tensor.item()]}",
            classes=classes, 
            use_rgb=False,
            save_path=os.path.join(
                config.get('DESTN_PATH'),
                f"{img_name}_actualclass-{classes[op_tensor.item()]}-gradcam.png"
            ),
            show_preview=True
        )
        # utils.plot_img(overlayed_img)