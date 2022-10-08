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
import pandas as pd

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
classes_to_split = ['Y', 'N']


def extract_year(data_df):

    def extraction_applicator(val):
        return int(val.split('$')[1].split('-')[0])

    data_df['year'] = data_df['slide_name'].apply(extraction_applicator)
    return data_df


def split_weighted_chronological(data_df):

    def var_delta(row):
        return abs(row[0]-row[1])
    
    sample_weights = data_df.groupby(['slide_name', 'label']).count().unstack(fill_value=0).iloc[:, 0:len(classes_to_split)]
    sample_weights.columns = sample_weights.columns.droplevel(level=0)    
    sample_weights['var_delta'] = sample_weights.apply(var_delta, axis=1)
    sample_weights.reset_index()
    print(sample_weights.columns)
    
    data_df_samples = data_df.loc[:, ['slide_name', 'year']]

    data_df_aug = data_df_samples.merge(sample_weights, on='slide_name').drop_duplicates(keep='first')
    data_df_aug = data_df_aug.sort_values(by=['year', 'var_delta'], ascending=[True, True])
    # data_df_aug.to_csv("temp.csv")
    print(data_df_aug)
    print(data_df_aug[classes_to_split].cumsum())
    print(data_df_aug.join(data_df_aug[classes_to_split].cumsum(), rsuffix='_cumul').to_csv("temp.csv", index=False))

    return sample_weights


def split_for_experiment():

    data_df_raw = describe_datafolder(
        DS_RAW_PATH,
        to_file=False,
        display=False
    )
    data_df = extract_year(
        data_df_raw.loc[data_df_raw['label'].isin(classes_to_split)]
    )

    sample_weights = split_weighted_chronological(data_df)
