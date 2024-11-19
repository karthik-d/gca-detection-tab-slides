import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

from ..utils import render_verbose_props, get_printable_confusion_matrix
from ..dataloader.loader import get_dataset_for
from ..config import config

# TODO: Perform logging to a file
# Possible format: (epoch,train_loss,train_acc,valid_loss,valid_acc)

# TODO: Improve training-step display -- split to functors


def prepare_load_model(num_layers=18, num_classes=2, pretrain=True):    
    """
    Renders a ResNet model of specified layer-count
    """

    # Validate layer count
    if num_layers not in (18, 34, 50, 101, 152):
        raise ValueError("Invalid ResNet Layer count. Need one of [18, 34, 50, 101, 152]")

    model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
    model = model_constructor(num_classes=num_classes)

    # Use torch's pretrainer config
    if pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)

    return model


def run_concrete_training_loop(
    classes,
    model,
    dataloaders,
    dataset_sizes,
    start_epoch,
    n_epochs,
    batch_size,
    checkpointing_frequency,
    loss_function,
    optimizer,
    scheduler,
    device
):

    """
    Runs the complete train-val loop for `n` epochs
    """

    # Initialize tensors --- get CPU memory image ref
    train_all_labels = torch.empty(
        size=(dataset_sizes["train"], ),
        dtype=torch.long
    ).cpu()
    train_all_predicts = torch.empty(
        size=(dataset_sizes["train"], ),
        dtype=torch.long
    ).cpu()
    valid_all_labels = torch.empty(
        size=(dataset_sizes["valid"], ),
        dtype=torch.long
    ).cpu()
    valid_all_predicts = torch.empty(
        size=(dataset_sizes["valid"], ),
        dtype=torch.long
    ).cpu()


    # Begin Train-Validate Loop
    for epoch in range(start_epoch, n_epochs):

        epoch_start_time = time.time()

        # TRAINING PHASE --------
        model.train(mode=True)

        # Init accumulators
        train_running_loss = 0.0
        train_running_corrects = 0

        # Train over all training data -- batch-wise
        num_train_steps = len(dataloaders["train"])
        print()
        for idx, (inputs, labels) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
            optimizer.zero_grad()

            # Propagate forward and back
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __dispose, train_preds = torch.max(train_outputs, dim=1)
                train_loss = loss_function(
                    input=train_outputs,
                    target=train_labels
                )
                train_loss.backward()
                optimizer.step()

            # Update training stats
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(
                train_preds == train_labels.data, 
                dtype=torch.double
            )

            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            # Detach from graph and store result tensor values
            train_all_labels[start_idx:end_idx] = train_labels.detach().cpu()
            train_all_predicts[start_idx:end_idx] = train_preds.detach().cpu()

            print(f"Training Step: {idx} of {num_train_steps}", end='\r')

        print()
        render_verbose_props(
            "Confusion Matrix - Train Data",
            get_printable_confusion_matrix(
                all_labels=train_all_labels.numpy(),
                all_predictions=train_all_predicts.numpy(),
                classes=classes
            )
        )

        # Store training stats
        train_loss_stat = train_running_loss / dataset_sizes["train"]
        train_acc_stat = train_running_corrects / dataset_sizes["train"]

        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        # VALIDATION PHASE --------
        model.train(mode=False)

        # Init accumulators
        valid_running_loss = 0.0
        valid_running_corrects = 0

        # Feed forward over all the validation data.
        for idx, (valid_inputs, valid_labels) in enumerate(dataloaders["valid"]):
            valid_inputs = valid_inputs.to(device=device)
            valid_labels = valid_labels.to(device=device)

            # Feed-Forward ONLY!
            with torch.set_grad_enabled(mode=False):
                valid_outputs = model(valid_inputs)
                __dispose, valid_preds = torch.max(valid_outputs, dim=1)
                valid_loss = loss_function(
                    input=valid_outputs, 
                    target=valid_labels
                )

            # Update validation stats
            valid_running_loss += valid_loss.item() * valid_inputs.size(0)
            valid_running_corrects += torch.sum(
                valid_preds == valid_labels.data,
                dtype=torch.double
            )

            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            # Detach from graph and store result tensor values
            valid_all_labels[start_idx:end_idx] = valid_labels.detach().cpu()
            valid_all_predicts[start_idx:end_idx] = valid_preds.detach().cpu()

        render_verbose_props(
            "Confusion Matrix - Validation Data",
            get_printable_confusion_matrix(
                all_labels=valid_all_labels.numpy(),
                all_predictions=valid_all_predicts.numpy(),
                classes=classes
            )
        )

        # Store validation stats
        valid_loss_stat = valid_running_loss / dataset_sizes["valid"]
        valid_acc_stat = valid_running_corrects / dataset_sizes["valid"]

        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        # UPDATE TRAINING DRIVERS and META ------
        scheduler.step()

        current_lr = []
        for group in optimizer.param_groups:
            current_lr.append(group["lr"])


        # Save Checkpoint (based on frequency)    
        if epoch % checkpointing_frequency == 0:

            # Make checkpoint filename
            epoch_output_path = os.path.join(
                config.get('RUN_CHECKPOINT_PATH'),
                "epoch#{epoch_num}_val_acc#{valid_acc}.ckpt".format(
                   epoch_num = epoch,
                   valid_acc = str(round(valid_acc_stat.item(), 4)).replace('.', '-')
                )
            )

            # Save the model as a state dictionary.
            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1
                },
                f=str(epoch_output_path)
            )

            print(f"Checkpoint saved as: {os.path.basename(epoch_output_path)}")
        

        # Display Epoch Summary
        render_verbose_props(
            title="Training Epoch Summary",
            epoch=epoch,
            learning_rate=current_lr,
            train_loss=train_loss_stat,
            train_acc=train_acc_stat,
            valid_loss=valid_loss_stat,
            valid_acc=valid_acc_stat,
            epoch_duration=f"{(time.time()-epoch_start_time)} seconds"
        )


def train_driver(
    classes,
    resnet_layers=18,
    pretrain=True,
    batch_size=16,
    n_epochs=10,
    learning_rate=1e-03,
    weight_decay=1e-04,
    learning_rate_decay=0.85,
    device='cuda:0',
    checkpoint_resumepath=None,
    checkpointing_frequency=1,
    num_workers=8
):

    """
    Loads training configuration and data
    Drives the train-val loop
    """

    data_transforms = {}
    """
    data_transforms = get_data_transforms(
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_hue=color_jitter_hue,
        color_jitter_saturation=color_jitter_saturation,
        path_mean=path_mean,
        path_std=path_std)
    """

    # TEMP: Load without transforms (not used at run-time)
    image_datasets = {
        x: get_dataset_for(split=x)
        for x in ("train", "valid")
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            dataset=image_datasets[x],
            batch_size=batch_size,
            shuffle=(x=="train"),
            num_workers=num_workers
        )
        for x in ("train", "valid")
    }

    dataset_sizes = {
        x: len(image_datasets[x]) 
        for x in ("train", "valid")
    }

    render_verbose_props(
        title="Dataset Configuration",
        classes_count=len(classes),
        class_names=classes,
        train_set_size=(len(dataloaders['train']) * batch_size),
        valid_set_size=(len(dataloaders['valid']) * batch_size)
    )

    model = prepare_load_model(
        num_classes=len(classes),
        num_layers=resnet_layers,
        pretrain=pretrain
    ).to(device=device)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=learning_rate_decay
    )

    # Start/Resume Training
    if checkpoint_resumepath is not None:
        checkpoint = torch.load(f=checkpoint_resumepath)
        model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Continuing training from {checkpoint_resumepath} - Epoch: {start_epoch}")
    else:
        start_epoch = 0

    # Print the model hyperparameters.
    render_verbose_props(
        title="Training Configuration",
        model="ResNet",
        num_layers=resnet_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        pretrain=pretrain,
        resume_checkpoint=(checkpoint_resumepath if checkpoint_resumepath is not None else False),
        checkpointing_frequency=checkpointing_frequency,
        using_CUDA=torch.cuda.is_available()
    )

    # Run the trainer
    # -- Use cross-entropy loss
    run_concrete_training_loop(
        classes=classes,
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        start_epoch=start_epoch,
        n_epochs=n_epochs,
        batch_size=batch_size,
        checkpointing_frequency=checkpointing_frequency,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )