"""
Applies inference to all images 
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

and displays evaluation metrics,
based on the ResNet-18 model weights in the `CHECKPOINT_FILEPATH`
"""

## SET THE 3 PARAMETERS in [ROOT]/src/test/config.py: 

import torch
import numpy as np
import os

from train.architectures import resnet
from train.utils import render_verbose_props, get_printable_confusion_matrix
from .dataloader import loader
from .config import config


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

    return image_dataset, dataloader, classes


def render_inputs(dataloader):
    for (vis_x, vis_y), (img_path, _) in zip(iter(dataloader), dataloader.dataset.samples):
        yield vis_x, vis_y, img_path



def save_visualization():
    """
    # Driver function to generate and save visualizations
    """

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = resnet.prepare_load_model(
        num_layers=18,
        num_classes=2,
        pretrain=True
    )
	model.to(device=device)
	model.eval()

    # Load checkpoint file
    checkpoint_resumepath = config.get('CHECKPOINT_FILEPATH')
    if checkpoint_resumepath is not None:
        checkpoint = torch.load(f=checkpoint_resumepath)
        model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Using weights from {checkpoint_resumepath} - Epoch: {start_epoch}")
    else:
        print(f"Using IMAGENET weights")

    image_dataset, dataloader, classes = prepare_inputs()

	# Init accumulators
	running_loss = 0.0
	running_corrects = 0

	# Store start time
	inference_start_time = time.time()

	# Feed forward over all the data.
	for idx, (inputs, labels) in enumerate(dataloader):
		inputs = inputs.to(device=device)
		labels = labels.to(device=device)

		# Feed-Forward ONLY!
		with torch.set_grad_enabled(mode=False):
			outputs = model(inputs)
			__dispose, preds = torch.max(outputs, dim=1)
			loss = loss_function(
				input=outputs, 
				target=labels
			)

		# Update validation stats
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(
			preds == labels.data,
			dtype=torch.double
		)

		start_idx = idx * batch_size
		end_idx = start_idx + batch_size
		# Detach from graph and store result tensor values
		all_labels[start_idx:end_idx] = labels.detach().cpu()
		all_predicts[start_idx:end_idx] = preds.detach().cpu()

	render_verbose_props(
		"Confusion Matrix - Validation Data",
		get_printable_confusion_matrix(
			all_labels=all_labels.numpy(),
			all_predictions=all_predicts.numpy(),
			classes=classes
		)
	)

	# Store validation stats
	loss_stat = running_loss / len(image_dataset)
	acc_stat = running_corrects / len(image_dataset)

	# CUDA cleanup
	torch.cuda.empty_cache() if torch.cuda.is_available() else pass	

	# Display Epoch Summary
	render_verbose_props(
		title="Held-out Testing Summary",
		loss=loss_stat,
		acc=acc_stat,
		inference_time=f"{(time.time()-inference_start_time)} seconds"
	)