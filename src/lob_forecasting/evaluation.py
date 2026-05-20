from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_predictions(y_true, y_pred, model_name, print_report=True):
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
