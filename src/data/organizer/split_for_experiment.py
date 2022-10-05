"""
Splits entire data into `training` and `testing`

ds_phase_N_raw/
|- labeled
|- experiment_splits.json
    |- training
    |- testing

ds_phase_N/
|- splits/       (copy of ds_phase_N_raw/training)
    |- train/
    |- valid/
|- train_splits.json
    |- train
    |- valid

"""

"""
## TODO

= Unzip files into `ds_phase_N_raw`
= Generate `experiment_splits.json` into `ds_phase_N_raw` with 'training' and 'testing'
= Parse json to prepare directories `training` and `testing`
"""