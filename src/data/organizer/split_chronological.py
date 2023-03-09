"""
Splits entire data into `training` and `testing` at the sample/slide level to generate csv
- `test_year` is placed into the `test` subset
- the remainder is randomly shuffled between `train` and `valid`

BEFORE:
    ds_phase_N/
    |- assorted

AFTER:
    ds_phase_N/
    |- assorted/
    |- experiment_split.csv (with `train`, `valid`, and `test` as members under split_category column)
"""

# TODO: process split years through cumulative weighting


import os
import csv
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

#--enter 
test_split_fraction = 0.2
# approximate fractions

#--enter 
valid_split_fraction = 0.2
# approximate fractions

#--enter
classes_to_split = ['Y', 'N']

# (to change)
#--enter
test_years = [
    int(year)
    for year in [2020, 2018, 2017, 2016, 2015, 2013]
]


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
            if abs(test_split_fraction-curr_fraction) < best_fraction:
                best_fraction = curr_fraction
                print(index)

    print(best_ratio)
    print(best_fraction)
    print(year_cumul_ordered)


def split_data(image_path, prefix=""):
    paths = glob.glob(f"{image_path}/*.*")
    base_names = list(map(os.path.basename, paths))
    base_names = np.array(base_names)

    # Set data length for valid splits
    total_len = len(base_names)
    print(f"Total {total_len} data")
    valid_len = int(total_len * 0.1)

    # Create data splits
    indices = np.random.permutation(total_len)
    train_indices = indices[valid_len:]
    valid_indices = indices[:valid_len]

    train_names = base_names[train_indices].tolist()
    valid_names = base_names[valid_indices].tolist()
    train_names = list(map(lambda x: f"{prefix}{x}", train_names))
    valid_names = list(map(lambda x: f"{prefix}{x}", valid_names))
    print(f"Train has {len(train_names)} data from {image_path}")
    print(f"Valid has {len(valid_names)} data from {image_path}")
    return {"train": train_names, "valid": valid_names}


def split_chronological(pos_name='Y'):
    
    """
    perform the split based on inferred years for test
    (temporarily) splits by supplied value of `test_years`
    """

    data_df_raw = describe_datafolder(
        DS_RAW_PATH,
        to_file=False,
        display=False
    )

    # add `year` as an attribute to the data
    data_df = extract_year(
        data_df_raw.loc[data_df_raw['label'].isin(classes_to_split)]
    )

    """
    # DEBUG
    (slide_roi_cumul, year_roi_cumul, year_slide_cumul) = get_splitting_desciptors(data_df)
    print(slide_roi_cumul)
    print(year_roi_cumul)
    print(year_slide_cumul)
    """

    # TODO: Insert cumulative-count based processing lines here

    # Perform test split
    data_df = data_df.assign(split_category=[None,]*len(data_df))
    for year in test_years:
        data_df.loc[data_df['year']==year, ['split_category']] = 'test'
    
    # Stratified random split at ROI-level to assign `train` and `valid` labels
    non_test_neg_indices = data_df.loc[
        (data_df['split_category']!='test') & (data_df['label']=='N'), 
        :
    ].index
    non_test_pos_indices = data_df.loc[
        (data_df['split_category']!='test') & (data_df['label']==pos_name), 
        :
    ].index

    neg_total_count = len(non_test_neg_indices)
    neg_val_count = int(neg_total_count*valid_split_fraction)
    neg_permuatation = np.random.permutation(neg_total_count)
    train_neg_indices = non_test_neg_indices[neg_permuatation[neg_val_count:]]
    val_neg_indices = non_test_neg_indices[neg_permuatation[:neg_val_count]]

    pos_total_count = len(non_test_pos_indices)
    pos_val_count = int(pos_total_count*valid_split_fraction)   
    pos_permuatation = np.random.permutation(pos_total_count)
    train_pos_indices = non_test_pos_indices[pos_permuatation[pos_val_count:]]
    val_pos_indices = non_test_pos_indices[pos_permuatation[:pos_val_count]]

    # apply train and valid splits
    data_df.loc[train_neg_indices, ['split_category']] = 'train'
    data_df.loc[train_pos_indices, ['split_category']] = 'train'

    data_df.loc[val_neg_indices, ['split_category']] = 'valid'
    data_df.loc[val_pos_indices, ['split_category']] = 'valid'

    print("[INFO] ROI Counts by Split.")
    for split in ['train', 'valid', 'test']:
        print(f"= {split}: {len(data_df.loc[data_df['split_category']==split, :])}")