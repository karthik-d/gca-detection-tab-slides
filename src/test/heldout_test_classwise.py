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

## SET THE 4 PARAMETERS in [ROOT]/src/test/config.py.

from sklearn import metrics
from matplotlib import pyplot as plot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time

from train.architectures import resnet
from train.utils import render_verbose_props, get_printable_confusion_matrix
from .dataloader import loader
from .config import config


def prepare_inputs(batch_size=16, num_workers=1):

	image_dataset = loader.get_dataset()
	dataloader = torch.utils.data.DataLoader(
		dataset=image_dataset,
		batch_size=batch_size,
		shuffle=False
	)
	classes = image_dataset.find_classes(image_dataset.root)[0]

	return image_dataset, dataloader, classes


def render_confusion_matrix(predictions, truths, save_path=None):

	conf_matrix = metrics.confusion_matrix(truths, predictions)
	display = metrics.ConfusionMatrixDisplay.from_predictions(truths, predictions)

	plot.title('Held Out Test - Confusion Matrix')
	if save_path is not None:
		plot.gcf().savefig(save_path, dpi=1000)
	else:
		plot.show()
	
	return conf_matrix

	


def heldout_test_driver(model_filepath=None):
	"""
	# Driver function to generate and save visualizations
	"""
	
	device = 'cuda:0' if torch.cuda.is_available() else 'mps'
	model = resnet.prepare_load_model(
		num_layers=config.get('RESNET_NLAYERS'),
		num_classes=2,
		pretrain=True
	)
	model.to(device=device)
	model.eval()

	# Load checkpoint file
	if model_filepath is not None:
		checkpoint = torch.load(f=model_filepath, map_location=device)
		model.load_state_dict(state_dict=checkpoint["model_state_dict"])
		start_epoch = checkpoint["epoch"]
		print(f"Using weights from {model_filepath} - Epoch: {start_epoch}")
	else:
		print(f"Using IMAGENET weights")

	loss_function = nn.CrossEntropyLoss()
	batch_size = 8
	image_dataset, dataloader, classes = prepare_inputs(batch_size=batch_size)

	# initialize inference stores
	all_labels = torch.empty(
		size=(len(image_dataset), ),
		dtype=torch.long
	).cpu()
	all_predicts = torch.empty(
		size=(len(image_dataset), ),
		dtype=torch.long
	).cpu()

	# Init accumulators
	running_loss = 0.0
	running_corrects = 0

	fnames, labels_l = tuple(zip(*dataloader.dataset.samples))
	fnames = [os.path.basename(x).rstrip('.tiff') for x in fnames]
	pos_probs = []
	neg_probs = []

	# Store start time
	eval_start_time = time.time()
	num_eval_steps = len(dataloader)
	# Feed forward over all the data.
	for idx, (inputs, labels) in enumerate(dataloader):
		print(f"Evaluation Step: {idx} of {num_eval_steps}", end='\r')

		inputs = inputs.to(device=device)
		labels = labels.to(device=device)

		# Feed-Forward ONLY!
		with torch.set_grad_enabled(mode=False):
			outputs = model(inputs)

			# save probabilities.
			out_probs = F.softmax(outputs).T.detach().cpu()
			neg_probs.extend([float(x) for x in out_probs[0]])
			pos_probs.extend([float(x) for x in out_probs[1]])
			
			# save predictions.
			__dispose, preds = torch.max(outputs, dim=1)
			loss = loss_function(
				input=outputs, 
				target=labels
			)

		start_idx = idx * batch_size
		end_idx = start_idx + batch_size
		# Detach from graph and store result tensor values
		all_labels[start_idx:end_idx] = labels.detach().cpu()
		all_predicts[start_idx:end_idx] = preds.detach().cpu()

	
	# create dataframe to store probabilities.
	predictions_df = pd.DataFrame(dict(fname=fnames, label=labels_l, pos_prob=pos_probs, neg_prob=neg_probs)).set_index('fname')
	print(predictions_df)
	predictions_df.to_csv(f"{os.path.basename(model_filepath).rstrip('.ckpt')}_valid.csv")
	
	print(model_filepath)
	render_verbose_props(
		"Confusion Matrix - Validation Data",
		get_printable_confusion_matrix(
			all_labels=all_labels.numpy(),
			all_predictions=all_predicts.numpy(),
			classes=classes
		)
	)
	

def heldout_test_classwise():

	models_to_test = [
		# 'experiment_4/run_1/epoch#6_val_acc#0-9564.ckpt',
		# 'experiment_3/run_1/epoch#6_val_acc#0-6076.ckpt',
		'experiment_3/run_1/epoch#0_val_acc#0-9297.ckpt',
	]
	model_path_base = os.path.join(
        config.get('LOGS_PATH'),
        'train'
    )
	for model_name in models_to_test:
		heldout_test_driver(os.path.join(model_path_base, model_name))