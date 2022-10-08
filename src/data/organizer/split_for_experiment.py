"""
Splits entire data into `training` and `testing` at the sample level

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


import os

from .describe import describe_datafolder

#--enter
DS_RAW_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*3), 
    "dataset",
    "data",
    "roi",
    "ds_phase_3_raw",
    "labeled"
))

#--enter
split_fractions = [0.8, 0.2]
# Approximate

#--enter
classes_to_split = ['P', 'N']


def extract_year(data_df):

    def extraction_applicator(val):
        return val.split('$')[1].split('-')[0]

    data_df['year'] = data_df['slide_name'].apply(extraction_applicator)
    return data_df


def gather_sample_weighting(data_df):

    data_df.sort_values(by=['year', 'slide_name', 'label'])
    print(data_df)
    print(data_df.groupby(['slide_name', 'label']).count())


def split_for_experiment():

    data_df = describe_datafolder(
        DS_RAW_PATH,
        to_file=False,
        display=False
    )
    data_df = extract_year(data_df)
    print(data_df)

    gather_sample_weighting(data_df)
