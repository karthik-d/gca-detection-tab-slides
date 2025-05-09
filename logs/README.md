# Model Configurations and Dataset Phases

**NOTE**: The training parameters for each phase, configurations, and run are listed in their respective training logs

## Dataset Phases

### Phase 1 (ds_phase_1)

- 348 ROIs
    - 124 Positive
    - 224 Negative
- 80:20 Train-Val Split
- Universal resize to (512, 512, 3)
- No additional transformaions or augmentations

### Phase 2 (ds_phase_2)

- 348 ROIs
    - 124 Positive
    - 224 Negative
- 80:20 Train-Val Split
- Universal resize to (512, 512, 3)
- Train set augmented using ***6 transformations***. `config` setting as below,
    ```
    AUGMENTATIONS = [
        'vertical_flip',
        'horizontal_flip',
        'rotate_90',
        'rotate_acute',
        'random_hgram_equalize',
        'color_jitter'
    ],
    AUGMENTATION_PARAMS = [
        dict(),
        dict(),
        dict(
            allow_clockwise=True,
            allow_counter_clockwise=True
        ),
        dict(
            degrees=(-10, 10)
        ),
        dict(),
        dict(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5
        )
    ]
    ```
- Training set composition after augmentation:
    - 693 Positive
    - 1253 Negative

### Phase 3 (ds_phase_3)

- 4912 ROIs
    - 1178 Positive
    - 3634 Negative
- 70:30 (approx.) Train-Val Split - Chronological with 
- Held-out Test Years: [2020, 2018, 2017, 2016, 2015, 2013]
- Universal resize to (512, 512, 3)
- No additional transformaions or augmentations


## Model Configurations

### Configuration 1 (model_config_1)

- Architecture: ResNet18
- No runtime transformations or augmentations


### Configuration 2 (model_config_2)

- Architecture: ResNet34
- No runtime transformations or augmentations


## Experiments

| Experiment# | Dataset | Model | Runs | Logs |
|:-----------:|:-------:|:-----:|:----:|:----:|
| 01 | ds_phase_1 | model_config_1 | 2 | [Here](/logs/train/experiment_1) |
| 02 | ds_phase_2 | model_config_1 | 2 | [Here](/logs/train/experiment_2) |
| 03 | ds_phase_3 | model_config_1 | 1 | [Here](/logs/train/experiment_3) |
| 04 | ds_phase_3 | model_config_2 | 1 | [Here](/logs/train/experiment_4) |




