from sklearn.metrics import (
	accuracy_score,
	classification_report,
	f1_score,
	precision_recall_fscore_support,
)


def evaluate_predictions(y_true, y_pred, model_name, print_report=True):
	"""
	Evaluate predictions using accuracy, macro F1, and optionally a classification report.
	"""
	accuracy = accuracy_score(y_true, y_pred)
	macro_f1 = f1_score(y_true, y_pred, average="macro")
	if print_report:
		print(model_name)
		print(f"Accuracy: {accuracy:.4f}")
		print(f"Macro F1: {macro_f1:.4f}")
		print()
		print(classification_report(y_true, y_pred, zero_division=0))
	return {
		"model": model_name,
		"accuracy": accuracy,
		"macro_f1": macro_f1,
	}


def evaluate_predictions_for_table(
	y_true,
	y_pred,
	model_name,
	feature_set=None,
	horizon=None,
):
	"""
	Evaluate predictions and return a structured row for experiment tables.
	"""
	accuracy = accuracy_score(y_true, y_pred)
	macro_f1 = f1_score(y_true, y_pred, average="macro")
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=[1, 2, 3],
		zero_division=0,
	)
	row = {
		"model": model_name,
		"accuracy": accuracy,
		"macro_f1": macro_f1,
		"precision_up": precision[0],
		"recall_up": recall[0],
		"f1_up": f1[0],
		"support_up": support[0],
		"precision_flat": precision[1],
		"recall_flat": recall[1],
		"f1_flat": f1[1],
		"support_flat": support[1],
		"precision_down": precision[2],
		"recall_down": recall[2],
		"f1_down": f1[2],
		"support_down": support[2],
	}
	if feature_set is not None:
		row["feature_set"] = feature_set
	if horizon is not None:
		row["horizon"] = horizon
	return row
