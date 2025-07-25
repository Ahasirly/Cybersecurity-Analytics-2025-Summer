import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

# 加载模型和scaler
MODEL_DIR = Path("models")
network_model = load_model(MODEL_DIR / "cnn_network_model.h5", compile=False)
network_scaler = joblib.load(MODEL_DIR / "scaler_network.pkl")

# 加载特征列表
def load_feature_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

network_features = load_feature_list("features/network_features.txt")
print(f"Network features count: {len(network_features)}")

# 加载数据集
df = pd.read_csv('data/network_score_sampled_from100w.csv')
print(f"Dataset shape: {df.shape}")

# 测试第一个样本
sample_idx = 0
sample = df.iloc[sample_idx]

print(f"\n=== Testing sample {sample_idx} ===")
print(f"Sample network_score: {sample['network_score']}")

# 创建特征向量
network_feature_vector = []
for feature in network_features:
    if feature in sample.index:
        value = float(sample[feature]) if pd.notna(sample[feature]) else 0.0
        network_feature_vector.append(value)
    else:
        print(f"Feature '{feature}' not found in dataset")
        network_feature_vector.append(0.0)

print(f"Feature vector length: {len(network_feature_vector)}")
print(f"First 10 features: {network_feature_vector[:10]}")

# 创建输入数据
X = np.array(network_feature_vector).reshape(1, -1)
X_df = pd.DataFrame(X, columns=network_features)

print(f"Input shape: {X.shape}")
print(f"Input DataFrame shape: {X_df.shape}")

# 标准化
X_scaled = network_scaler.transform(X_df)
print(f"Scaled shape: {X_scaled.shape}")
print(f"Scaled data range: {X_scaled.min():.6f} to {X_scaled.max():.6f}")

# 重塑为CNN输入
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
print(f"CNN input shape: {X_cnn.shape}")

# 预测
prediction = network_model.predict(X_cnn, verbose=0)
print(f"Raw prediction: {prediction}")
print(f"Prediction shape: {prediction.shape}")
print(f"Final prediction: {float(prediction[0][0])}")

# 测试多个样本
print(f"\n=== Testing multiple samples ===")
for i in range(5):
    sample = df.iloc[i]
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