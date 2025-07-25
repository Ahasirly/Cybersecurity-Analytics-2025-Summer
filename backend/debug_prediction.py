import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
import pickle
import joblib

# Load the model and scaler
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

# Load URL model and scaler
url_model = load_model(MODEL_DIR / "malicious_url_model.h5", compile=False)

try:
    with open(MODEL_DIR / "scaler_url_model.pkl", 'rb') as f:
        url_scaler = pickle.load(f)
except:
    url_scaler = joblib.load(MODEL_DIR / "scaler_url_model.pkl")

# Load URL features
with open("features/url_features.txt", 'r') as f:
    url_features = [line.strip() for line in f.readlines()]

# Load teaching data
teaching_data = pd.read_csv(DATA_DIR / "fused_with_botnet_saved.csv")

print(f"URL model expects {len(url_features)} features")
print(f"Teaching data has {len(teaching_data)} samples")

# Test with sample 0
sample_idx = 0
full_sample = teaching_data.iloc[sample_idx]

print(f"\n=== Testing with sample {sample_idx} ===")

# Create complete URL feature vector from actual data
url_feature_vector = []
for feature in url_features:
    if feature in full_sample.index:
        value = float(full_sample[feature]) if pd.notna(full_sample[feature]) else 0.0
        url_feature_vector.append(value)
    else:
        url_feature_vector.append(0.0)

print(f"URL feature vector length: {len(url_feature_vector)}")
print(f"Non-zero features: {sum(1 for x in url_feature_vector if x != 0)}/{len(url_feature_vector)}")
print(f"First 10 features: {url_feature_vector[:10]}")

# Test prediction
X = np.array(url_feature_vector).reshape(1, -1)
X_df = pd.DataFrame(X, columns=url_features)
X_scaled = url_scaler.transform(X_df)
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))

print(f"Input shape: {X.shape}")
print(f"Scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

prediction = float(url_model.predict(X_cnn, verbose=0)[0][0])
print(f"Prediction: {prediction:.4f}")

# Test with different samples
print(f"\n=== Testing multiple samples ===")
for i in range(5):
    sample_idx = i
    full_sample = teaching_data.iloc[sample_idx]
    
    url_feature_vector = []
    for feature in url_features:
        if feature in full_sample.index:
            value = float(full_sample[feature]) if pd.notna(full_sample[feature]) else 0.0
            url_feature_vector.append(value)
        else:
            url_feature_vector.append(0.0)
    
    X = np.array(url_feature_vector).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=url_features)
    X_scaled = url_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    prediction = float(url_model.predict(X_cnn, verbose=0)[0][0])
    non_zero_count = sum(1 for x in url_feature_vector if x != 0)
    
    print(f"Sample {i}: Prediction = {prediction:.4f}, Non-zero features = {non_zero_count}/{len(url_feature_vector)}")

# Check if the model is broken
print(f"\n=== Testing model with extreme values ===")
test_cases = [
    ("All zeros", [0.0] * len(url_features)),
    ("All ones", [1.0] * len(url_features)),
    ("All high values", [100.0] * len(url_features)),
    ("Mixed values", [i for i in range(len(url_features))])
]

for name, features in test_cases:
    X = np.array(features).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=url_features)
    X_scaled = url_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    prediction = float(url_model.predict(X_cnn, verbose=0)[0][0])
    print(f"{name}: Prediction = {prediction:.4f}") 