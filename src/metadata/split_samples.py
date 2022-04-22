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
    os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class_trial-' + str(x+1) + '.csv')
    for x in range(len(fileset_paths))
]

def split_samples():
    with open(samples_path) as f_src:
        fileset_dfs = [pd.read_csv(filepath) for filepath in fileset_paths]
        split_sample_files = [open(filepath, 'w') for filepath in split_samples_paths]
        sample_ctrs = [0 for x in range(len(fileset_paths))]
        instance_ctrs = [0 for x in range(len(fileset_paths))]

        header_line = f_src.readline()
        for file_out in split_sample_files:
            file_out.write(header_line)
        
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
                if prev_sample!=sample_name and (written_to is not None):
                    sample_ctrs[written_to] += 1
                    written_to = None
            prev_sample = sample_name
            line = f_src.readline()
        
        # Close all files
        _ = map(lambda x: close(x), split_sample_files)
        print(f"Saved {sample_ctrs} into {len(sample_ctrs)} sets with {instance_ctrs} rows")

    # Cross-Check
    for f_path in fileset_paths:
        print("\nChecking", f_path)
        with open(os.path.join(config.get("METADATA_PATH"), f_path)) as f_in:
            _ = f_in.readline()
            check_fd = pd.read_csv(os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class_trial.csv'))
            sample_name = f_in.readline().strip()
            ctr = 0
            while sample_name:
                # print(sample_name)
                ctr += len(check_fd.loc[check_fd["slidename"]==sample_name, :]) 
                if len(check_fd.loc[check_fd["slidename"]==sample_name, :]) < 1:
                    print("- No match for:", sample_name)
                # print(len(check_fd.loc[check_fd["slidename"]==sample_name, :]))
                sample_name = f_in.readline().strip()
            print(f"Total for '{f_path}':", ctr)

