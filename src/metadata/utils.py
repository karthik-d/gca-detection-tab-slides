import pandas as pd
import csv
import os

from config import config

# Relevant-For-Analysis rows
analysis_slidenames_path = os.path.join((config.get("METADATA_PATH")), 'analysis_relevant.csv')

def get_slides_list_minus(slides_list_path, minus_list_path):

	slides_list = pd.read_csv(slides_list_path)
	analysis_slides = pd.read_csv(analysis_slidenames_path)

	minus_list = analysis_slides[analysis_slides.slidename.isin(slides_list.slidename)==False]
	minus_list['slidename'].to_csv(minus_list_path, index=False)