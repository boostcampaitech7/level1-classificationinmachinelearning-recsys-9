import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def train_lgb_model(x_train, y_train, x_valid, y_valid, params: dict) -> lgb.Booster:
	"""
	Train a LightGBM model.

	Args:
	x_train (pd.DataFrame): Training features.
	y_train (pd.Series): Training labels.
	x_valid (pd.DataFrame): Validation features.
	y_valid (pd.Series): Validation labels.
	params (dict): LightGBM parameters.

	Returns:
	lgb.Booster: Trained LightGBM model.
	"""
	train_data = lgb.Dataset(x_train, label=y_train)
	valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

	model = lgb.train(params, train_data, valid_sets=valid_data)
	return model


def predict_lgb_model(model: lgb.Booster, x_test: pd.DataFrame) -> np.ndarray:
	"""
	Predict using a trained LightGBM model.

	Args:
	model (lgb.Booster): Trained LightGBM model.
	x_test (pd.DataFrame): Test features.

	Returns:
	np.ndarray: Predicted class labels.
	"""
	return np.argmax(model.predict(x_test), axis=1)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float]:
	"""
	Evaluate model performance.

	Args:
	y_true (pd.Series): True labels.
	y_pred (np.ndarray): Predicted labels.

	Returns:
	Tuple[float, float]: Accuracy and AUROC score.
	"""
	accuracy = accuracy_score(y_true, y_pred)
	auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
	return accuracy, auc
