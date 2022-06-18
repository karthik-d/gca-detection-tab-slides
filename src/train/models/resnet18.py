

def train_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, num_epochs: int,
                 writer: IO, device: torch.device, start_epoch: int,
                 batch_size: int, save_interval: int, checkpoints_folder: Path,
                 num_layers: int, classes: List[str],
                 num_classes: int) -> None:
    """
    Function for training ResNet.

    Args:
        model: ResNet model for training.
        dataloaders: Dataloaders for IO pipeline.
        dataset_sizes: Sizes of the training and validation dataset.
        criterion: Metric used for calculating loss.
        optimizer: Optimizer to use for gradient descent.
        scheduler: Scheduler to use for learning rate decay.
        start_epoch: Starting epoch for training.
        writer: Writer to write logging information.
        device: Device to use for running model.
        num_epochs: Total number of epochs to train for.
        batch_size: Mini-batch size to use for training.
        save_interval: Number of epochs between saving checkpoints.
        checkpoints_folder: Directory to save model checkpoints to.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    since = time.time()

    # Initialize all the tensors to be used in training and validation.
    # Do this outside the loop since it will be written over entirely at each
    # epoch and doesn't need to be reallocated each time.
    train_all_labels = torch.empty(size=(dataset_sizes["train"], ),
                                   dtype=torch.long).cpu()
    train_all_predicts = torch.empty(size=(dataset_sizes["train"], ),
                                     dtype=torch.long).cpu()
    val_all_labels = torch.empty(size=(dataset_sizes["val"], ),
                                 dtype=torch.long).cpu()
    val_all_predicts = torch.empty(size=(dataset_sizes["val"], ),
                                   dtype=torch.long).cpu()

    # Train for specified number of epochs.
    for epoch in range(start_epoch, num_epochs):

        # Training phase.
        model.train(mode=True)

        train_running_loss = 0.0
        train_running_corrects = 0

        # Train over all training data.
        for idx, (inputs, labels) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs,
                                       target=train_labels)
                train_loss.backward()
                optimizer.step()

            # Update training diagnostics.
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(
                train_preds == train_labels.data, dtype=torch.double)

            start = idx * batch_size
            end = start + batch_size

            train_all_labels[start:end] = train_labels.detach().cpu()
            train_all_predicts[start:end] = train_preds.detach().cpu()

        calculate_confusion_matrix(all_labels=train_all_labels.numpy(),
                                   all_predicts=train_all_predicts.numpy(),
                                   classes=classes,
                                   num_classes=num_classes)

        # Store training diagnostics.
        train_loss = train_running_loss / dataset_sizes["train"]
        train_acc = train_running_corrects / dataset_sizes["train"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation phase.
        model.train(mode=False)

        val_running_loss = 0.0
        val_running_corrects = 0

        # Feed forward over all the validation data.
        for idx, (val_inputs, val_labels) in enumerate(dataloaders["val"]):
            val_inputs = val_inputs.to(device=device)
            val_labels = val_labels.to(device=device)

            # Feed forward.
            with torch.set_grad_enabled(mode=False):
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, dim=1)
                val_loss = criterion(input=val_outputs, target=val_labels)

            # Update validation diagnostics.
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data,
                                              dtype=torch.double)

            start = idx * batch_size
            end = start + batch_size

            val_all_labels[start:end] = val_labels.detach().cpu()
            val_all_predicts[start:end] = val_preds.detach().cpu()

        calculate_confusion_matrix(all_labels=val_all_labels.numpy(),
                                   all_predicts=val_all_predicts.numpy(),
                                   classes=classes,
                                   num_classes=num_classes)

        # Store validation diagnostics.
        val_loss = val_running_loss / dataset_sizes["val"]
        val_acc = val_running_corrects / dataset_sizes["val"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()

        current_lr = None
        for group in optimizer.param_groups:
            current_lr = group["lr"]

        # Remaining things related to training.
        if epoch % save_interval == 0:
            epoch_output_path = checkpoints_folder.joinpath(
                f"resnet{num_layers}_e{epoch}_va{val_acc:.5f}.pt")

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

        writer.write(f"{epoch},{train_loss:.4f},"
                     f"{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # Print the diagnostics for each epoch.
        print(f"Epoch {epoch} with lr "
              f"{current_lr:.15f}: "
              f"t_loss: {train_loss:.4f} "
              f"t_acc: {train_acc:.4f} "
              f"v_loss: {val_loss:.4f} "
              f"v_acc: {val_acc:.4f}\n")

    # Print training information at the end.
    print(f"\ntraining complete in "
          f"{(time.time() - since) // 60:.2f} minutes")


def train_resnet(
        train_folder: Path, batch_size: int, num_workers: int,
        device: torch.device, classes: List[str], learning_rate: float,
        weight_decay: float, learning_rate_decay: float,
        resume_checkpoint: bool, resume_checkpoint_path: Path, log_csv: Path,
        color_jitter_brightness: float, color_jitter_contrast: float,
        color_jitter_hue: float, color_jitter_saturation: float,
        path_mean: List[float], path_std: List[float], num_classes: int,
        num_layers: int, pretrain: bool, checkpoints_folder: Path,
        num_epochs: int, save_interval: int) -> None:
    """
    Main function for training ResNet.

    Args:
        train_folder: Location of the automatically built training input folder.
        batch_size: Mini-batch size to use for training.
        num_workers: Number of workers to use for IO.
        device: Device to use for running model.
        classes: Names of the classes in the dataset.
        learning_rate: Learning rate to use for gradient descent.
        weight_decay: Weight decay (L2 penalty) to use in optimizer.
        learning_rate_decay: Learning rate decay amount per epoch.
        resume_checkpoint: Resume model from checkpoint file.
        resume_checkpoint_path: Path to the checkpoint file for resuming training.
        log_csv: Name of the CSV file containing the logs.
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.
        num_classes: Number of classes in the dataset.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        pretrain: Use pretrained ResNet weights.
        checkpoints_folder: Directory to save model checkpoints to.
        num_epochs: Number of epochs for training.
        save_interval: Number of epochs between saving checkpoints.
    """
    # Loading in the data.
    data_transforms = get_data_transforms(
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_hue=color_jitter_hue,
        color_jitter_saturation=color_jitter_saturation,
        path_mean=path_mean,
        path_std=path_std)

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
    dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}

    print(f"{num_classes} classes: {classes}\n"
          f"num train images {len(dataloaders['train']) * batch_size}\n"
          f"num val images {len(dataloaders['val']) * batch_size}\n"
          f"CUDA is_available: {torch.cuda.is_available()}")

    model = create_model(num_classes=num_classes,
                         num_layers=num_layers,
                         pretrain=pretrain)
    model = model.to(device=device)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,
                                           gamma=learning_rate_decay)

    # Initialize the model.
    if resume_checkpoint:
        ckpt = torch.load(f=resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"model loaded from {resume_checkpoint_path}")
    else:
        start_epoch = 0

    # Print the model hyperparameters.
    print_params(batch_size=batch_size,
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
                 weight_decay=weight_decay)

    # Logging the model after every epoch.
    # Confirm the output directory exists.
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    with log_csv.open(mode="w") as writer:
        writer.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        # Train the model.
        train_helper(model=model,
                     dataloaders=dataloaders,
                     dataset_sizes=dataset_sizes,
                     criterion=nn.CrossEntropyLoss(),
                     optimizer=optimizer,
                     scheduler=scheduler,
                     start_epoch=start_epoch,
                     writer=writer,
                     batch_size=batch_size,
                     checkpoints_folder=checkpoints_folder,
                     device=device,
                     num_layers=num_layers,
                     save_interval=save_interval,
                     num_epochs=num_epochs,
                     classes=classes,
                     num_classes=num_classes)