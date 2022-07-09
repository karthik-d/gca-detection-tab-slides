from utils import render_verbose_params

# TODO: Perform logging to a file
# Possible format: (epoch,train_loss,train_acc,val_loss,val_acc)

# TODO: Add checkpointing to training loop


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
    model,
    dataloaders,
    dataset_sizes,
    start_epoch,
    n_epochs,
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
    val_all_labels = torch.empty(
        size=(dataset_sizes["val"], ),
        dtype=torch.long
    ).cpu()
    val_all_predicts = torch.empty(
        size=(dataset_sizes["val"], ),
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

        print(get_printable_confusion_matrix(
            all_labels=val_all_labels.numpy(),
            all_predicts=val_all_predicts.numpy(),
            classes=classes
        ))

        # Store training stats
        train_loss_stat = train_running_loss / dataset_sizes["train"]
        train_acc_stat = train_running_corrects / dataset_sizes["train"]

        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        # VALIDATION PHASE --------
        model.train(mode=False)

        # Init accumulators
        val_running_loss = 0.0
        val_running_corrects = 0

        # Feed forward over all the validation data.
        for idx, (val_inputs, val_labels) in enumerate(dataloaders["val"]):
            val_inputs = val_inputs.to(device=device)
            val_labels = val_labels.to(device=device)

            # Feed-Forward ONLY!
            with torch.set_grad_enabled(mode=False):
                val_outputs = model(val_inputs)
                __dispose, val_preds = torch.max(val_outputs, dim=1)
                val_loss = loss_function(
                    input=val_outputs, 
                    target=val_labels
                )

            # Update validation stats
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(
                val_preds == val_labels.data,
                dtype=torch.double
            )

            start_idx = idx * batch_size
            end_idx = start_idx + batch_size
            # Detach from graph and store result tensor values
            val_all_labels[start_idx:end_idx] = val_labels.detach().cpu()
            val_all_predicts[start_idx:end_idx] = val_preds.detach().cpu()

        print(get_printable_confusion_matrix(
            all_labels=val_all_labels.numpy(),
            all_predicts=val_all_predicts.numpy(),
            classes=classes
        ))

        # Store validation stats
        val_loss_stat = val_running_loss / dataset_sizes["val"]
        val_acc_stat = val_running_corrects / dataset_sizes["val"]

        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        # UPDATE TRAINING DRIVERS and META ------
        scheduler.step()

        current_lr = []
        for group in optimizer.param_groups:
            current_lr.append(group["lr"])


        # TODO: Add checkpointing (template written)
        """
        if epoch % checkpointing_frequency == 0:
            epoch_output_path = checkpoints_folder.joinpath(
                f"resnet{num_layers}_run1_epoch{epoch}_valacc{val_acc:.5f}.ckpt")

            # Confirm the output directory exists.
            epoch_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the model as a state dictionary.
            torch.save(obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1
            },
                       f=str(epoch_output_path))
        """

        # Display Epoch Summary
        render_verbose_params(
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
    batch_size=16,
    learning_rate=1e-03,
    weight_decay=1e-04,
    learning_rate_decay=0.85,
    device='cuda:0',
    checkpoint_filepath=None,
    checkpointing_frequency=1
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

    image_datasets = {
        x: datasets.ImageFolder(root=str(train_folder.joinpath(x)),
                                transform=data_transforms[x])
        for x in ("train", "val")
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=(x is "train"),
                                       num_workers=num_workers)
        for x in ("train", "val")
    }

    dataset_sizes = {
        x: len(image_datasets[x]) 
        for x in ("train", "val")
    }

    render_verbose_params(
        title="Dataset Configuration",
        classes_count=len(classes),
        class_names=classes,
        train_set_size=(len(dataloaders['train']) * batch_size),
        valid_set_size=(len(dataloaders['val']) * batch_size)
    )

    model = prepare_load_model(
        num_classes=num_classes,
        num_layers=num_layers,
        pretrain=pretrain
    ).to(device=device)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=learning_rate_decay
    )

    # Start/Resume Training
    if checkpoint_filepath is not None:
        checkpoint = torch.load(f=checkpoint_filepath)
        model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Continuing training from {checkpoint_filepath} - Epoch: {start_epoch}")
    else:
        start_epoch = 0

    # Print the model hyperparameters.
    render_verbose_params(
        title="Training Configuration",
        batch_size=batch_size,
        checkpoints_folder=checkpoints_folder,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        log_csv=log_csv,
        num_epochs=num_epochs,
        num_layers=num_layers,
        pretrain=pretrain,
        resume_checkpoint=resume_checkpoint,
        resume_checkpoint_path=resume_checkpoint_path,
        save_interval=save_interval,
        train_folder=train_folder,
        weight_decay=weight_decay,
        using_CUDA=torch.cuda.is_available()
    )

    # Run the trainer
    # -- Use cross-entropy loss
    run_concrete_training_loop(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        start_epoch=start_epoch,
        n_epochs=n_epochs,
        checkpointing_frequency=checkpointing_frequency,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=opimizer,
        scheduler=scheduler,
        device=device
    )