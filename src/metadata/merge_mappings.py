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

    sc_mapping = csv.reader(open(sc_mapping_path))
    header_row = next(sc_mapping)     # ignore row

    merged_mapping = pd.DataFrame(columns=[
        'slidename',
        'order',
        'sample',
        'roi_number',
        'is_analysis_relevant',
        'is_positive',
        'notes'
    ])

    sample = None
    sample_has_rows = False
    notes = ''
    for row in sc_mapping:

        sample = row[0]
        is_valid_sample = ( len(fs_mapping.loc[fs_mapping['Sample']==sample, :])>0 )
        if is_valid_sample:
            # Merge and Store collected rows
            if sample_has_rows:
                
                if(len(fs_mapping.loc[fs_mapping['Sample']==sample, :])!=1):
                    """
                    print("\n--------------------------------------------------")
                    print("Sample:", sample, row)
                    print(fs_mapping.loc[fs_mapping['Sample']==sample, :])
                    print("\n--------------------------------------------------\n")
                    # print("\n--------------------------------------------------\n")
                    """
                    pass
                else:
                    print("Storing sample ", sample)
                    # print("\n--------------------------------------------------")
                    fs_rows = fs_mapping.loc[fs_mapping['Sample']==sample, ['Slide Name', 'Order', 'Sample']].to_dict(orient='records')[0]
                    num_rois = len(merged_rows['roi_number'])
                    merged_rows['slidename'] = [ str(fs_rows['Slide Name']) for x in range(num_rois) ]
                    merged_rows['order'] = [ str(fs_rows['Order']) for x in range(num_rois) ]
                    merged_rows['sample'] = [ str(fs_rows['Sample']) for x in range(num_rois) ]
                    # Append to merged dataframe
                    merged_mapping = pd.concat([merged_mapping, pd.DataFrame(merged_rows)])
            
            # Sample starting row - reset dictionary
            sample, is_analysis_relevant, _, notes = tuple(row) 
            sample_roi_rows = []
            merged_rows = {
                'slidename': [],
                'order': [],
                'sample': [],
                'roi_number': [],
                'is_analysis_relevant': [],
                'is_positive': [],
                'notes': []
            }

        else:
            roi_num, is_analysis_relevant, is_positive, notes = tuple(row)
            merged_rows['roi_number'].append(roi_num)
            merged_rows['is_analysis_relevant'].append(is_analysis_relevant)
            merged_rows['is_positive'].append(is_positive)
            merged_rows['notes'].append(notes)
            sample_has_rows = True
        
    print(merged_mapping)
    merged_mapping.to_csv(merged_mapping_path)



