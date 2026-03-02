# src/skipper_ai/train.py

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

def train_model(data_path, model_output_path):
	"""
	Trains a machine learning model to predict sailing performance ratios based on the processed data.
	"""
	# Load and preprocess data
	df = pd.read_csv(data_path)

	# Convert categorical variables to numeric
	df['sail_id'] = df['sail_id'].astype('category').cat.codes

	# Define features and target variable
	features = ['tws', 'twa', 'heel', 'sail_id']
	target = 'performance_ratio'

	x = df[features]
	y = df[target]

	# Split data into training and testing sets
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	model = xgb.XGBRegressor(
		objective='reg:squarederror',
		n_estimators=100,
		max_depth=5,
		learning_rate=0.1
	)
	
	model.fit(x_train, y_train)

	# Save the trained model
	joblib.dump(model, model_output_path)
	print("Model trained and saved to", model_output_path)

if __name__ == "__main__":
	train_model('data/sailing_data.csv', 'models/sailing_performance_model.pkl')