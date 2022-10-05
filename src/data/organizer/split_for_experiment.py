"""
Splits entire data into `training` and `testing`

ds_phase_N_raw/
|- training/
|- testing/
|- experiment_splits.json

ds_phase_N/
|- splits/       (copy of ds_phase_N_raw/training)
    |- train/
    |- valid/
|- train_splits.json

"""

"""
## TODO

= Unzip files into `ds_phase_N_raw`
= Generate `experiment_splits.json` into `ds_phase_N_raw` with 'training' and 'testing'
= Parse json to prepare directories `training` and `testing`
"""