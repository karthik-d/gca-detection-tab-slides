"""
Splits training data into `train` and `valid`

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

= Generate `train_splits.json` into `ds_phase_N` with 'train' and 'valid'
= Parse json to prepare data hierarchy in `splits` -- 'train' and 'valid'
"""