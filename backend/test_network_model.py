import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load model and scaler
network_model = tf.keras.models.load_model('models/network_model.h5')
network_scaler = pickle.load(open('models/network_scaler.pkl', 'rb'))

# Load feature list
with open('features/network_features.txt', 'r') as f:
    network_features = [line.strip() for line in f.readlines()]

print("Available network features:", len(network_features))
print("Network features:", network_features)

# Load dataset
network_data = pd.read_csv('data/network_score_sampled_from100w.csv')
print("Data shape:", network_data.shape)

# Test first sample
sample_data = network_data.iloc[0]

print(f"\n=== Testing sample {0} ===")
print(f"Sample network_score: {sample_data['network_score']}")

# Create feature vector
network_feature_vector = []
for feature in network_features:
    if feature in sample_data.index:
        value = float(sample_data[feature]) if pd.notna(sample_data[feature]) else 0.0
        network_feature_vector.append(value)
    else:
        print(f"Feature '{feature}' not found in dataset")
        network_feature_vector.append(0.0)

print(f"Feature vector length: {len(network_feature_vector)}")
print(f"First 10 features: {network_feature_vector[:10]}")

# Create input data
X = np.array(network_feature_vector).reshape(1, -1)
X_df = pd.DataFrame(X, columns=network_features)

print(f"Input shape: {X.shape}")
print(f"Input DataFrame shape: {X_df.shape}")

# Standardize
X_scaled = network_scaler.transform(X_df)
print(f"Scaled shape: {X_scaled.shape}")
print(f"Scaled data range: {X_scaled.min():.6f} to {X_scaled.max():.6f}")

# Reshape for CNN input
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
print(f"CNN input shape: {X_cnn.shape}")

# Predict
prediction = network_model.predict(X_cnn, verbose=0)
print(f"Raw prediction: {prediction}")
print(f"Prediction shape: {prediction.shape}")
print(f"Final prediction: {float(prediction[0][0])}")

# Test multiple samples
print(f"\n=== Testing multiple samples ===")
for i in range(5):
    sample = network_data.iloc[i]
    network_feature_vector = []
    for feature in network_features:
        if feature in sample.index:
            value = float(sample[feature]) if pd.notna(sample[feature]) else 0.0
            network_feature_vector.append(value)
        else:
            network_feature_vector.append(0.0)
    
    X = np.array(network_feature_vector).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=network_features)
    X_scaled = network_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    prediction = network_model.predict(X_cnn, verbose=0)
    actual_score = sample['network_score']
    predicted_score = float(prediction[0][0])
    
    print(f"Sample {i}: Actual={actual_score:.4f}, Predicted={predicted_score:.4f}, Diff={abs(actual_score-predicted_score):.4f}") 