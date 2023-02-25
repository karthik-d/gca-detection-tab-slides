import pandas as pd
import csv
import os

from config import config

# File to Sample mapping
fs_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample.csv')

# Sample to Class mapping
sc_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_sample-class.csv')

# FSample to Class cleaned mapping
sc_mapping_cleaned_path = os.path.join(config.get("METADATA_PATH"), 'mapping_sample-class_cleaned.csv')

# File-Sample-Class mapping
merged_mapping_path = os.path.join(config.get("METADATA_PATH"), 'mapping_file-sample-class.csv')

# NOT-Relevant-For-Analysis rows
no_analysis_slidenames_path = os.path.join((config.get("METADATA_PATH")), 'not_analysis_relevant.csv')

# Relevant-For-Analysis rows
analysis_slidenames_path = os.path.join((config.get("METADATA_PATH")), 'analysis_relevant.csv')


def _describe_analysis_slides(slides_df):
	print(slides_df.describe())
	print(slides_df.groupby(['slide_inference']).count())


def describe_all_analysis_slides():
	_describe_analysis_slides(pd.read_csv(analysis_slidenames_path))


def describe_any_slide_list(slides_list_path):

	reqd_slides = pd.read_csv(slides_list_path)
	analysis_slides = pd.read_csv(analysis_slidenames_path)

	print(reqd_slides)
	for_describe = analysis_slides.merge(reqd_slides, on='slidename', how='inner')
	print(for_describe)
	# _describe_analysis_slides(for_describe)