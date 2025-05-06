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
	predictions = df['prob']
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
	predictions = df['prediction']
	truths = df['label']

	plot.clf()
	conf_matrix = metrics.confusion_matrix(truths, predictions)
	display = metrics.ConfusionMatrixDisplay.from_predictions(truths, predictions)
	plot.title(f'{dataset} - Confusion Matrix')
	if save_path is not None:
		plot.gcf().savefig(save_path, dpi=100)
	else:
		plot.show()
	return conf_matrix


def qualify_df(df):
	# adds slide and prediction info. 
	df_ret = df.copy()
	df_ret['prob'] = [pos for pos, neg in zip(df['pos_prob'], df['neg_prob'])]
	df_ret['prediction'] = (df['pos_prob'] > df['neg_prob']).astype(int) 
	df_ret['slide'] = [x.split(' ')[0] for x in df.index.values]
	return df_ret


def slidewise_inference(df):
	slidewise_dfs = df.groupby(by='slide')
	slides_l = []
	probs_l = []
	labels_l = []
	predictions_l = []
	
	for slide, df in slidewise_dfs:
		pos_prob = 0
		neg_prob = 0
		label = 0
		prediction = 0
		for idx, row in df.iterrows():
			# set label to 1 if any one of the ROIs is positive.
			if row['label']==1:
				label = 1
			if row['prediction']==1:
				prediction = 1
			# find the highest probability for each class.
			if row['pos_prob'] > pos_prob:
				pos_prob = row['pos_prob']
			if row['neg_prob'] > neg_prob:
				neg_prob = row['neg_prob']
		
		slides_l.append(slide)
		labels_l.append(label)
		probs_l.append((neg_prob, pos_prob)[prediction])
		predictions_l.append(prediction)
	
	return pd.DataFrame(dict(slide=slides_l, label=labels_l, prediction=predictions_l, prob=probs_l)).set_index('slide')



def eval_metrics(prefix=""):

	# held-out test.
	result_df = qualify_df(merge_results(
		pos_df = pd.read_csv("epoch#6_val_acc#0-9564_test.csv", index_col=0),
		neg_df = pd.read_csv("epoch#6_val_acc#0-9564_test.csv", index_col=0)
	))
	result_df.to_csv(f"../outputs/{prefix}_prediction-probs_test.csv")
	slidewise_df = slidewise_inference(result_df)
	save_roc_curve(result_df, f"../outputs/{prefix}_roc-auc_test.png", dataset="ROI-level Held-out Test")
	save_confusion_matrix(result_df, f"../outputs/{prefix}_conf-matrix_test.png", dataset="ROI-level Held-out Test")
	save_roc_curve(slidewise_df, f"../outputs/{prefix}_roc-auc_test_slidewise.png", dataset="Slide-level Held-out Test")
	save_confusion_matrix(slidewise_df, f"../outputs/{prefix}_conf-matrix_test_slidewise.png", dataset="Slide-level Held-out Test")

	# validation test.
	result_df = qualify_df(pd.read_csv("epoch#0_val_acc#0-9297_valid.csv", index_col=0))
	result_df.to_csv(f"../outputs/{prefix}_prediction-probs_valid.csv")
	slidewise_df = slidewise_inference(result_df)
	save_roc_curve(result_df, f"../outputs/{prefix}_roc-auc_valid.png", dataset="ROI-level Validation")
	save_confusion_matrix(result_df, f"../outputs/{prefix}_conf-matrix_valid.png", dataset="ROI-level Validation")
	save_roc_curve(slidewise_df, f"../outputs/{prefix}_roc-auc_valid_slidewise.png", dataset="Slide-level Validation")
	save_confusion_matrix(slidewise_df, f"../outputs/{prefix}_conf-matrix_valid_slidewise.png", dataset="Slide-level Validation")