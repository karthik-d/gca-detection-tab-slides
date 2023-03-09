"""
Splits entire data into `training` and `testing` at the sample/slide level

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
# TODO: process split years through cumulative weighting
# TODO: Unzip files into `ds_phase_N_raw`
# TODO: Generate `experiment_splits.json` into `ds_phase_N_raw` with 'training' and 'testing'
# TODO: Parse json to prepare directories `training` and `testing`
"""


import os
import pandas as pd
from functools import partial
import numpy as np

from .describe import describe_datafolder

#--enter
DS_RAW_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), *((os.path.pardir,)*4), 
    "dataset",
    "annotations",
    "phase-on-07Feb23"
))

#--enter (higher fraction)
split_fraction = 0.8
# approximate fractions

#--enter
classes_to_split = ['Y', 'N']

# (to change)
#--enter
test_years = []


def extract_year(data_df):

    def extraction_applicator(val):
        return int(val.split('$')[1].split('-')[0])

    data_df['year'] = data_df['slide_name'].apply(extraction_applicator)
    return data_df


def get_splitting_desciptors(data_df, to_file=False):

    def var(row):
        return abs(row[0]-row[1])

    def slide_inference(pos_name, row):
        """returns 'Y' for a positive slide-level inference, 'N' otherwise"""
        return 'Y' if row[pos_name]!=0 else 'N'
        
   
    data_df_samples = data_df.loc[:, ['slide_name', 'year']]
    
    # overall divisive weighting
    primitive_counts = data_df.groupby(['slide_name', 'label']).count().unstack(fill_value=0).iloc[:, 0:len(classes_to_split)]
    primitive_counts.columns = primitive_counts.columns.droplevel(level=0)
    primitive_counts['var'] = primitive_counts.apply(var, axis=1)
    primitive_counts = primitive_counts.reset_index()
    
    # slide-level (ROI): cumulative sum of ROI class counts for each slide
    slide_roi_cumul = data_df_samples.merge(primitive_counts, on='slide_name').drop_duplicates(keep='first')
    slide_roi_cumul = slide_roi_cumul.sort_values(by=['year', 'var'], ascending=[True, True])
    if to_file:
        slide_roi_cumul.join(slide_roi_cumul[classes_to_split].cumsum(), rsuffix='_cumul').to_csv("slide-roi-cumulation.csv", index=False)
    
    # year-level (ROI): cumulative sum of ROI class counts for each year
    year_roi_cumul = data_df.groupby(['year', 'label']).count().unstack(fill_value=0).iloc[:, 0:len(classes_to_split)]
    year_roi_cumul.columns = year_roi_cumul.columns.droplevel(level=0)
    year_roi_cumul = year_roi_cumul.reset_index()
    if to_file:
        year_roi_cumul.join(year_roi_cumul[classes_to_split].cumsum(), rsuffix='_cumul').to_csv("year-roi-cumulation.csv", index=False)

    # year-level (slide): cumulative sum of slide class counts for each year
    year_slide_cumul = primitive_counts.loc[:, ['slide_name'] + classes_to_split]
    year_slide_cumul['slide_inference'] = year_slide_cumul.apply(
        partial(slide_inference, ('P' if 'P' in classes_to_split else 'Y')), 
        axis=1
    )
    year_slide_cumul = year_slide_cumul.reset_index()
    year_slide_cumul = year_slide_cumul[['slide_name', 'slide_inference']].merge(
        data_df_samples, 
        on='slide_name'
    ).drop_duplicates(keep='first')
    year_slide_cumul = year_slide_cumul.groupby(['year', 'slide_inference']).count().unstack(fill_value=0).iloc[:, 0:len(classes_to_split)]
    year_slide_cumul.columns = year_slide_cumul.columns.droplevel(level=0)
    year_slide_cumul = year_slide_cumul.sort_values(by='year', ascending=True)
    year_slide_cumul = year_slide_cumul.join(year_slide_cumul[classes_to_split].cumsum(), rsuffix='_cumul')
    year_slide_cumul = year_slide_cumul.reset_index(level=0)
   
    if to_file:
        year_slide_cumul.to_csv("year-slide-cumulation.csv", index=False)

    return (slide_roi_cumul, year_roi_cumul, year_slide_cumul)


def get_test_years_weighted_chronological(data_df):
    """ 
    get the years to split year-weighted class-counts by choosing the last few chronological years
    to warrant the closest match to split_fractions
    """

    return None


def get_test_years_weighted_best_subset(data_df):
    """ 
    get the years to split year-weighted class-counts by choosing the best set of years
    to warrant the closest match to split_fractions
    """

    _, _, year_slide_cumul = get_splitting_desciptors(data_df)
    year_slide_cumul = year_slide_cumul[['year'] + classes_to_split]
    
    ## Incremental cumulation in descending variance of sample sizes.

    # order rows by variance
    year_cumul_ordered = year_slide_cumul.sort_values(
        by=year_slide_cumul[classes_to_split].var().sort_values(ascending=False).keys().tolist(),
        ascending=[False for _ in classes_to_split]
    )

    # sum and select
    total_cnt = 472
    pos_name = ('P' if 'P' in classes_to_split else 'Y')
    n_sum, y_sum = 0, 0
    best_fraction = 0
    best_ratio = np.inf
    best_index = None
    first_pass = True
    for index, row in year_cumul_ordered.iterrows():
        print(index)

        n_sum = row['N']
        y_sum = row[pos_name]

        if first_pass:
            best_ratio = n_sum/y_sum
            best_fraction = (n_sum+y_sum)/total_cnt
            first_pass = False
        
        else: 
            curr_ratio = n_sum/y_sum
            curr_ratio = np.inf if curr_ratio==0 else curr_ratio
            if abs(1-curr_ratio) < best_ratio:
                best_ratio = curr_ratio
                print("A", index)
            
            curr_fraction = (n_sum+y_sum)/total_cnt
            if abs(split_fraction-curr_fraction) < best_fraction:
                best_fraction = curr_fraction
                print(index)

    print(best_ratio)
    print(best_fraction)
    print(year_cumul_ordered)


def split_chronological():

    data_df_raw = describe_datafolder(
        DS_RAW_PATH,
        to_file=False,
        display=False
    )

    # add `year` as an attribute to the data
    data_df = extract_year(
        data_df_raw.loc[data_df_raw['label'].isin(classes_to_split)]
    )

    # TODO: Insert cumulative-count based processing lines here
    primitive_counts = split_year_weighted_best_subset(data_df)
    print(primitive_counts)