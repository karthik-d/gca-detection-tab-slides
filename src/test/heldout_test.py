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
import numpy as np
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


def render_roc_curve(predictions, truths, save_path=None):

	conf_matrix = metrics.roc_curve(truths, predictions)
	auc_score = metrics.auc(fpr, tpr)

	plot.title('ROC Curve')
	plot.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
	plot.legend(loc = 'lower right')
	plot.plot([0, 1], [0, 1],'r--')
	plot.xlim([0, 1])
	plot.ylim([0, 1])
	plot.ylabel('True Positive Rate')
	plot.xlabel('False Positive Rate')

	if save_path is not None:
		plot.gcf().savefig(save_path, dpi=50)
	else:
		plot.show()
	
	return auc_score


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
	batch_size = 16
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
			__dispose, preds = torch.max(outputs, dim=1)
			loss = loss_function(
				input=outputs, 
				target=labels
			)

		# Update validation stats
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(
			preds == labels.data,
			dtype=torch.float32
		)

		start_idx = idx * batch_size
		end_idx = start_idx + batch_size
		# Detach from graph and store result tensor values
		all_labels[start_idx:end_idx] = labels.detach().cpu()
		all_predicts[start_idx:end_idx] = preds.detach().cpu()

		# print(all_labels[start_idx:end_idx])
		# print(all_labels[start_idx:end_idx])


	print(model_filepath)
	render_verbose_props(
		"Confusion Matrix - Validation Data",
		get_printable_confusion_matrix(
			all_labels=all_labels.numpy(),
			all_predictions=all_predicts.numpy(),
			classes=classes
		)
	)

	# # Store validation stats
	# loss_stat = running_loss / len(image_dataset)
	# acc_stat = running_corrects / len(image_dataset)

	# # CUDA cleanup
	# torch.cuda.empty_cache() if torch.cuda.is_available() else None	

	# # Save ROC-AUC
	# auc_score = render_roc_curve(
	# 	all_predicts, 
	# 	all_labels, 
	# 	save_path=os.path.join(config.get('ARBIT_STORE_PATH', 'held-out_roc-curve.png'))
	# )

	# # Display Epoch Summary
	# render_verbose_props(
	# 	title="Held-out Testing Summary",
	# 	loss=loss_stat,
	# 	acc=acc_stat,
	# 	auc_score=auc_score,
	# 	inference_time=f"{(time.time()-eval_start_time)} seconds"
	# )

	# # plot and save confusion matrix.
	# conf_matrix = render_confusion_matrix(
	# 	all_predicts, 
	# 	all_labels, 
	# 	save_path=os.path.join(config.get('ARBIT_STORE_PATH', 'held-out_confusion-matrix.png'))
	# )
	

def heldout_test():

	models_to_test = [
		# 'experiment_2/run_3/epoch#4_val_acc#0-9706.ckpt',
		# 'experiment_3/run_1/epoch#3_val_acc#0-9395.ckpt',
		'experiment_4/run_1/epoch#6_val_acc#0-9564.ckpt',
		# 'experiment_5/run_1/epoch#4_val_acc#0-9775.ckpt',
		# 'experiment_6/run_1/epoch#4_val_acc#0-9789.ckpt',
		# 'experiment_7/run_1/epoch#1_val_acc#0-9564.ckpt',
	]
	model_path_base = os.path.join(
        config.get('LOGS_PATH'),
        'train'
    )
	for model_name in models_to_test:
		heldout_test_driver(os.path.join(model_path_base, model_name))