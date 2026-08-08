import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def zscore_normalize(X: np.ndarray):
	"""
	Normalize data using zscore normalization

	Args:
		X (np.ndarray): Data (m_samples, n_features)

	Returns:
		X_norm (np.ndarray): Normalized data (m_samples, n_features)
		mean (np.ndarray): Mean value of every feature (n_features,)
		std (np.ndarray): Standart deviation of every feature (n_features,)
	"""

	mean = X.mean(axis=0)
	std = X.std(axis=0)

	X_norm = (X - mean) / std

	return X_norm, mean, std

def inverse_transform(data, mean, std):
	"""
	Transform values back after zscore normalization

	Args:
		data (np.ndarray): Normalized data to transform back
		mean (scalar): Mean of original data
		std (scalar): Standart deviation of original data

	Returns:
		original_data (np.ndarray): Transformed data
	"""

	original_data = data*std + mean

	return original_data

def f(X, w, b):
	"""
	Predict target using Linear Regression model

	Args:
		X_train (np.ndarray): Training data (m_samples, n_features)
		w (np.ndarray): Model weights (n_features,)
		b (scalar): Model bias

	Returns:
		y_pred (np.ndarray): Predicted target (m_samples,)
	"""

	w = w.reshape(1, -1)
	y_pred = np.sum(X*w, axis=1) + b		

	return y_pred

def J(X, y, w, b):
	"""
	Calculate loss using MSE loss function

	Args:
		X_train (np.ndarray): Training data (m_samples, n_features)
		y_true (np.ndarray): Targets (m_samples,)
		w (np.ndarray): Model weights (n_features,)
		b (scalar): Model bias

	Returns:
		loss (scalar): Loss
	"""

	m, n = X.shape

	loss = 1/(2*m) * (f(X, w, b) - y)**2

	return loss

def calculate_gradients(X_train, y_true, w, b):
	"""
	Calculates gradients for MSE Loss function

	Args:
		X_train (np.ndarray): Training data (m_samples, n_features)
		y_true (np.ndarray): Targets (m_samples,)
		w (np.ndarray): Model weights (n_features,)
		b (scalar): Model bias

	Returns:
		dj_dw (np.ndarray): gradient for weights
		dj_db (scalar): gradient for bias
	"""

	m, n = X_train.shape

	error = f(X_train, w, b) - y_true
	dj_dw = (1/m) * np.dot(X_train.T, error)
	dj_db = (1/m) * np.sum(error)

	return dj_dw, dj_db

def optimize_parameters(X_train, y_true, w_init, b_init, steps, alpha):
	"""
	Optimize parameters by computing gradient descent many steps

	Args: 
		X_train (np.ndarray): Training data (m_samples, n_features)
		y_true (np.ndarray): Targets (m_samples,)
		w_init (np.ndarray): Initial model weights (n_features,)
		b_init (scalar): Initial model bias
		steps (scalar): Number of gradient descent steps
		alpha (scalar): Learning rate

	Returns:
		w (np.ndarray): Optimized model weights (n_features,)
		b (scalar): Optimized model bias
		J_history (List): History of loss function during training
		w_history (List): History of model weights during training
		b_history (List): History of model bias during training

	"""

	J_history = []
	w_history = []
	b_history = []

	w = w_init
	b = b_init

	for step in range(steps):

		dj_dw, dj_db = calculate_gradients(X_train, y_true, w, b)

		loss = J(X_train, y_true, w, b)

		J_history.append(loss)
		w_history.append(w.copy())
		b_history.append(b)

		w -= alpha*dj_dw
		b -= alpha*dj_db

		if step % (steps//10) == 0:
			print(f"Iteration {step}: cost {loss.sum()}")

	return w, b, J_history, w_history, b_history

if __name__ == "__main__":
	pass
	#---------------------------------------------------

	# path = "./house_prices.csv"
	# data = pd.read_csv(path)

	# features = ["sqft_living", "sqft_lot", "floors", "bedrooms", "bathrooms"]

	# target = ["price"]

	# X_train = data.loc[:, features]
	# y_true = data.loc[:, target]

	# X_train = zscore_normalize(X_train.to_numpy())[0]
	# y_true = zscore_normalize(y_true.to_numpy())[0].ravel()

	# # X_train = X_train.to_numpy()
	# # y_true = y_true.to_numpy().ravel()

	# # print(f"X: std={X_train.std(axis=1)}, mean={X_train.mean(axis=0)}")
	# # print(f"y: std={y_true.std()}, mean={y_true.mean()}")

	# w_init = np.zeros(X_train.shape[1])
	# b_init = 5

	# # print("f() returned shape:", f(X_train, w_init, b_init).shape)
	# # print("J() returned shape:", J(X_train, y_true, w_init, b_init).shape)
	# # dj_dw, dj_db = calculate_gradients(X_train, y_true, w_init, b_init)
	# # print(f"{dj_dw=}\n{dj_db=}\n{dj_dw.shape=}")

	# w, b, J_his = optimize_parameters(X_train, y_true, w_init, b_init, 150_000, 3e-5)


	# fig, axes = plt.subplots(1, 5)
	# for idx in range(len(features)):
	# 	label = features[idx]

	# 	axes[idx].plot(X_train[:, idx], y_true, "bo", label=label)
	# 	axes[idx].plot(X_train[:, idx], f(X_train, w, b), "ro", label=label)
	# 	axes[idx].legend(loc='upper right')

	# plt.show()

	#---------------------------------------------------