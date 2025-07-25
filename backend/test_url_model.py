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

print(f"URL model expects {len(url_features)} features")
print(f"First 10 features: {url_features[:10]}")

# Test with different inputs
test_cases = [
    {
        "name": "All zeros",
        "features": [0.0] * len(url_features)
    },
    {
        "name": "All ones", 
        "features": [1.0] * len(url_features)
    },
    {
        "name": "Random values",
        "features": np.random.random(len(url_features)).tolist()
    },
    {
        "name": "Teaching features only (others zero)",
        "features": [3.625, 2.0, 16.0, 0.0, 14.0, 0.0] + [0.0] * (len(url_features) - 6)
    }
]

for i, test_case in enumerate(test_cases):
    print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
    
    X = np.array(test_case['features']).reshape(1, -1)
    print(f"Input shape: {X.shape}")
    print(f"Input range: [{X.min():.4f}, {X.max():.4f}]")
    
    try:
        X_scaled = url_scaler.transform(X)
        print(f"Scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
        
        X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
        prediction = float(url_model.predict(X_cnn, verbose=0)[0][0])
        print(f"Prediction: {prediction:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

# Test with actual data from CSV
print(f"\n--- Testing with actual CSV data ---")
df = pd.read_csv(DATA_DIR / "fused_with_botnet_saved.csv")
sample = df.iloc[0]

# Create feature vector from actual data
feature_vector = []
for feature in url_features:
    if feature in sample.index:
        value = float(sample[feature]) if pd.notna(sample[feature]) else 0.0
        feature_vector.append(value)
    else:
        feature_vector.append(0.0)

print(f"Actual sample features: {feature_vector[:10]}...")
print(f"Non-zero features: {sum(1 for x in feature_vector if x != 0)}/{len(feature_vector)}")

X = np.array(feature_vector).reshape(1, -1)
X_scaled = url_scaler.transform(X)
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
prediction = float(url_model.predict(X_cnn, verbose=0)[0][0])
print(f"Actual sample prediction: {prediction:.4f}") 