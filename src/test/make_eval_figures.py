from sklearn import metrics
from matplotlib import pyplot as plot

import pandas as pd 
import os 


def merge_results(pos_df, neg_df):
	pos_df_part = pos_df.loc[pos_df['label']==1, :]
	neg_df_part = neg_df.loc[neg_df['label']==0, :]
	return pd.concat([pos_df_part, neg_df_part])


def save_roc_curve(df, save_path, dataset):

	# if pos>neg, take pos; else take 1-neg i.e. pos ==> so pos either way.
	predictions = [pos for pos, neg in zip(df['pos_prob'], df['neg_prob'])]
	truths = df['label']

	fpr, tpr, thresholds = metrics.roc_curve(truths, predictions)
	auc_score = metrics.auc(fpr, tpr)

	# make plot.
	plot.clf()
	plot.title(f'{dataset} - ROC Curve')
	plot.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
	plot.legend(loc = 'lower right')
	plot.plot([0, 1], [0, 1],'r--')
	plot.xlim([0, 1])
	plot.ylim([0, 1])
	plot.ylabel('True Positive Rate')
	plot.xlabel('False Positive Rate')

	if save_path is not None:
		plot.gcf().savefig(save_path, dpi=100)
	else:
		plot.show()
	return auc_score



def save_confusion_matrix(df, save_path, dataset):

	# if pos>neg, take pos; else take 1-neg i.e. pos ==> so pos either way.
	predictions = [int(pos>neg) for pos, neg in zip(df['pos_prob'], df['neg_prob'])]
	truths = df['label']

	conf_matrix = metrics.confusion_matrix(truths, predictions)
	display = metrics.ConfusionMatrixDisplay.from_predictions(truths, predictions)

	plot.clf()
	plot.title(f'{dataset} - Confusion Matrix')
	if save_path is not None:
		plot.gcf().savefig(save_path, dpi=100)
	else:
		plot.show()
	return conf_matrix



def eval_metrics():

	# held-out test.
	result_df = merge_results(
		pos_df = pd.read_csv("epoch#6_val_acc#0-6076_test.csv", index_col=0),
		neg_df = pd.read_csv("epoch#0_val_acc#0-9297_test.csv", index_col=0)
	)
	save_roc_curve(result_df, "roc-auc_test.png", dataset="Held-out Test")
	save_confusion_matrix(result_df, "conf-matrix_test.png", dataset="Held-out Test")

	# validation test.
	result_df = pd.read_csv("epoch#6_val_acc#0-9564_valid.csv", index_col=0)
	save_roc_curve(result_df, "roc-auc_valid.png", dataset="Validation")
	save_confusion_matrix(result_df, "conf-matrix_valid.png", dataset="Validation")