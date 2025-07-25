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

# 方法1：直接使用数据集中的特征
print("\n--- Method 1: Direct from dataset ---")
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

X = np.array(network_feature_vector).reshape(1, -1)
X_df = pd.DataFrame(X, columns=network_features)
X_scaled = network_scaler.transform(X_df)
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))

prediction = network_model.predict(X_cnn, verbose=0)
print(f"Method 1 prediction: {float(prediction[0][0])}")

# 方法2：使用我们API中的方式
print("\n--- Method 2: API way ---")
# 模拟API中的数据处理
network_features_dict = {}
for feature in network_features:
    if feature in sample.index:
        try:
            value = float(sample[feature]) if pd.notna(sample[feature]) else 0.0
            network_features_dict[feature] = value
        except (ValueError, TypeError):
            print(f"⚠️  Network feature '{feature}' cannot be converted to float, using 0.0")
            network_features_dict[feature] = 0.0
    else:
        print(f"⚠️  Network feature '{feature}' not found in CSV, using 0.0")
        network_features_dict[feature] = 0.0

# 重新构建特征向量
api_feature_vector = []
for feature in network_features:
    api_feature_vector.append(network_features_dict[feature])

print(f"API feature vector length: {len(api_feature_vector)}")
print(f"First 10 features: {api_feature_vector[:10]}")

X_api = np.array(api_feature_vector).reshape(1, -1)
X_df_api = pd.DataFrame(X_api, columns=network_features)
X_scaled_api = network_scaler.transform(X_df_api)
X_cnn_api = X_scaled_api.reshape((1, X_scaled_api.shape[1], 1))

prediction_api = network_model.predict(X_cnn_api, verbose=0)
print(f"Method 2 prediction: {float(prediction_api[0][0])}")

# 检查两种方法是否一致
print(f"\n--- Comparison ---")
print(f"Method 1 vs Method 2: {float(prediction[0][0])} vs {float(prediction_api[0][0])}")
print(f"Are they equal? {np.allclose(prediction, prediction_api)}")

# 检查scaler的feature_names_in_
print(f"\n--- Scaler info ---")
print(f"Scaler feature names: {len(network_scaler.feature_names_in_)}")
print(f"First 10 scaler features: {network_scaler.feature_names_in_[:10]}")
print(f"First 10 our features: {network_features[:10]}")
print(f"Features match? {np.array_equal(network_scaler.feature_names_in_, network_features)}") 