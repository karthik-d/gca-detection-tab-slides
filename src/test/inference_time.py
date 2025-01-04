from sklearn import metrics
from matplotlib import pyplot as plot
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time

from train.architectures import resnet
from train.utils import render_verbose_props, get_printable_confusion_matrix
from .dataloader import loader
from .config import config


model_filepath = os.path.join(
	config.get('LOGS_PATH'),
	'train',
	'experiment_4/run_1/epoch#6_val_acc#0-9564.ckpt'
)

def estimate_inference_time(n_batches=16):

	# device config.
	if torch.cuda.is_available():
		device = 'cuda:0' 
		device_backend = torch.cuda
	else:
		device = 'mps'
		device_backend = torch.mps
	
	# set up timers and sync devices.
	starter, ender = device_backend.Event(enable_timing=True), device_backend.Event(enable_timing=True)
	model = resnet.prepare_load_model(
		num_layers=config.get('RESNET_NLAYERS'),
		num_classes=2,
		pretrain=True
	)

	# load weights.
	if model_filepath is not None:
		checkpoint = torch.load(f=model_filepath, map_location=device)
		model.load_state_dict(state_dict=checkpoint["model_state_dict"])
		start_epoch = checkpoint["epoch"]
		print(f"Using weights from {model_filepath} - Epoch: {start_epoch}")
	else:
		print(f"Using IMAGENET weights")

	model.to(device=device)
	model.eval()

	dummy_input = torch.randn(n_batches, 3, 512, 512, dtype=torch.float).to(device)
	
	# number of batches.
	repetitions = 1000
	timings = np.zeros((repetitions, ))
	
	# gpu warm up.
	for _ in range(10):
		_ = model(dummy_input)
	
	# measure performance.
	with torch.no_grad():
		for rep in range(repetitions):
			starter.record()
			_ = model(dummy_input)
			ender.record()

			# wait for gpu sync.
			device_backend.synchronize()

			curr_time = starter.elapsed_time(ender)
			timings[rep] = curr_time

			# verbosity.
			_ = print(rep) if rep%5==0 else None
	
	pd.DataFrame(dict(time=timings), index=False).to_csv(f"inference-times_batch-{n_batches}.csv")
	return timings 
