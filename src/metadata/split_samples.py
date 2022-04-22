import csv 
import pandas as pd
import os 

from config import config

fileset_paths = [
    os.path.join(config.get("METADATA_PATH"), 'fileset_1.txt'),
    os.path.join(config.get("METADATA_PATH"), 'fileset_2.txt')
]

samples_path = os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class_trial.csv')

split_samples_paths = [
    os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class_trial-' + str(x) + '.csv')
    for x in range(len(fileset_paths))
]

def split_samples():
    with open(samples_path) as f_src:
        fileset_dfs = [pd.read_csv(filepath) for filepath in fileset_paths]
        split_sample_files = [open(filepath, 'w') for filepath in split_samples_paths]
        sample_ctrs = [0 for x in range(len(fileset_paths))]
        instance_ctrs = [0 for x in range(len(fileset_paths))]
        
        line = f_src.readline()
        prev_sample = None
        while line:
            sample_name = line.split(',')[1]
            written_to = None
            for idx, fileset_df in enumerate(fileset_dfs):
                if [sample_name] in fileset_df.values:
                    instance_ctrs[idx] += 1
                    split_sample_files[idx].write(line)
                    written_to = idx
                if prev_sample!=sample_name and written_to is not None:
                    instance_ctrs[written_to] += 1
            prev_sample = sample_name
            line = f_src.readline()
        
        # Close all files
        _ = map(lambda x: close(x), split_sample_files)
        print(f"Saved {sample_ctrs} into 2 sets with {instance_ctrs} rows")
