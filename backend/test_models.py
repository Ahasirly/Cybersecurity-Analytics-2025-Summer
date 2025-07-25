import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import joblib
from pathlib import Path

# 加载模型和scaler
MODEL_DIR = Path("models")

# 加载URL模型
url_model = load_model(MODEL_DIR / "malicious_url_model.h5", compile=False)
url_scaler = joblib.load(MODEL_DIR / "scaler_url_model.pkl")

# 加载特征列表
def load_feature_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

url_features = load_feature_list("features/url_features.txt")
print(f"URL features count: {len(url_features)}")

# 测试URL模型
print("\n=== Testing URL Model ===")

# 创建测试数据 - 使用数据集中的第一个样本
df = pd.read_csv('data/URL_model_input_score_0.1_0.98.csv')
sample = df.iloc[0]

# 创建特征向量
url_feature_vector = []
for feature in url_features:
    if feature in sample.index:
        url_feature_vector.append(float(sample[feature]))
    else:
        url_feature_vector.append(0.0)

print(f"Feature vector length: {len(url_feature_vector)}")
print(f"First 10 features: {url_feature_vector[:10]}")

X = np.array(url_feature_vector).reshape(1, -1)
X_df = pd.DataFrame(X, columns=url_features)
X_scaled = url_scaler.transform(X_df)
X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))

print(f"Scaled data shape: {X_scaled.shape}")
print(f"CNN input shape: {X_cnn.shape}")

# 预测
prediction = url_model.predict(X_cnn, verbose=0)
print(f"Raw prediction: {prediction}")
print(f"Prediction shape: {prediction.shape}")
print(f"Final prediction: {float(prediction[0][0])}")

# 测试多个样本
print("\n=== Testing multiple samples ===")
for i in range(5):
    sample = df.iloc[i]
    url_feature_vector = []
    for feature in url_features:
        if feature in sample.index:
            url_feature_vector.append(float(sample[feature]))
        else:
            url_feature_vector.append(0.0)
    
    X = np.array(url_feature_vector).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=url_features)
    X_scaled = url_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    prediction = url_model.predict(X_cnn, verbose=0)
    print(f"Sample {i}: {float(prediction[0][0])}") 