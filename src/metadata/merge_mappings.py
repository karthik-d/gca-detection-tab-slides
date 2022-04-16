import pandas as pd
import csv
import os

from config import config

# File to Sample mapping
fs_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample.csv')

# Sample to Class mapping
sc_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_sample-class.csv')

# File-Sample-Class mapping
merged_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class.csv')

def merge_mappings():
    fs_mapping = pd.read_csv(fs_mapping_path)
    print(fs_mapping.head())

    sc_mapping = csv.reader(open(sc_mapping_path))
    header_row = next(sc_mapping)     # ignore row
    sample_roi_rows = []
    remarks = ''
    for row in sc_mapping:
        if not row[2].strip():
            # Sample start row
            remarks = row[-1]
            sample_roi_rows = []
        else:
            roi_num, is_analysis_relevant, is_positive, _ = tuple(row)
            sample_roi_rows.append([roi_num, is_analysis_relevant, is_positive, remarks])


    merged_mapping = pd.DataFrame(columns=[
        'slidename',
        'order',
        'sample',
        'roi_number',
        'is_analysis_relevant',
        'is_positive',
        'notes'
    ])



