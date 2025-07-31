import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load models and scalers
url_model = tf.keras.models.load_model('models/url_model.h5')
# Load URL model
url_scaler = pickle.load(open('models/url_scaler.pkl', 'rb'))

# Load feature lists
with open('features/url_features.txt', 'r') as f:
    url_features = [line.strip() for line in f.readlines()]

print("Available URL features:", len(url_features))
print("URL features:", url_features[:10])  # Show first 10

url_data = pd.read_csv('data/fused_with_botnet_saved.csv')
print("Data shape:", url_data.shape)
print("Data columns:", url_data.columns.tolist())

# Test URL model
print("\n=== Testing URL Model ===")

# Create test data - use first sample from dataset
sample_data = url_data.iloc[0]
print("Sample data:", sample_data)

# Create feature vector
url_feature_vector = []
missing_count = 0

for feature in url_features:
    if feature in sample_data:
        value = sample_data[feature]
        if pd.isna(value):
            url_feature_vector.append(0.0)
            missing_count += 1
        else:
            url_feature_vector.append(float(value))
    else:
        url_feature_vector.append(0.0)
        missing_count += 1

print(f"Feature vector length: {len(url_feature_vector)}")
print(f"Missing features: {missing_count}")
print(f"Feature vector: {url_feature_vector[:10]}...")  # Show first 10

# Create input data
X = np.array([url_feature_vector])
print("Input shape:", X.shape)

# Prediction
print("Raw prediction:", url_model.predict(X, verbose=0))

# Test multiple samples 