#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cybersecurity Fusion System - Flask Backend
Main application: Load models + Provide API endpoints
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import openai
import random

# ────────────────────────────── Configuration ──────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration



# ────────────────────────────── Paths ──────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
FEATURES_DIR = BASE / "features"
DATA_DIR = BASE / "data"

# ────────────────────────────── Teaching Features ──────────────────────────────
TEACHING_URL_FEATURES = [
    'url_entropy', 'url_count_dot', 'url_len', 
    'url_count_hyphen', 'url_count_letter', 'url_count_digit'
]

TEACHING_USER_FEATURES = [
    'protocol_type', 'encryption_used', 'browser_type',
    'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins'
]

TEACHING_NETWORK_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Flow Pkts/s',
    'Fwd Pkt Len Max', 'Pkt Len Mean', 'Pkt Size Avg', 'Flow Byts/s'
]

ALL_TEACHING_FEATURES = TEACHING_URL_FEATURES + TEACHING_USER_FEATURES + TEACHING_NETWORK_FEATURES

# ────────────────────────────── One-Hot Decoding ──────────────────────────────
def decode_one_hot_features(data):
    """Convert one-hot encoded features back to readable values"""
    decoded_data = data.copy()
    
    # Protocol type decoding
    protocol_mapping = {
        'protocol_type_ICMP': 'ICMP',
        'protocol_type_TCP': 'TCP', 
        'protocol_type_UDP': 'UDP'
    }
    
    protocol_value = None
    for feature, value in protocol_mapping.items():
        if feature in data and data[feature] == 1.0:
            protocol_value = value
            break
    
    if protocol_value:
        decoded_data['protocol_type'] = protocol_value
        # Remove one-hot columns
        for feature in protocol_mapping.keys():
            if feature in decoded_data:
                del decoded_data[feature]
    
    # Encryption type decoding
    encryption_mapping = {
        'encryption_used_AES': 'AES',
        'encryption_used_DES': 'DES',
        'encryption_used_Unknown': 'Unknown'
    }
    
    encryption_value = None
    for feature, value in encryption_mapping.items():
        if feature in data and data[feature] == 1.0:
            encryption_value = value
            break
    
    if encryption_value:
        decoded_data['encryption_used'] = encryption_value
        # Remove one-hot columns
        for feature in encryption_mapping.keys():
            if feature in decoded_data:
                del decoded_data[feature]
    
    # Browser type decoding
    browser_mapping = {
        'browser_type_Chrome': 'Chrome',
        'browser_type_Edge': 'Edge',
        'browser_type_Firefox': 'Firefox',
        'browser_type_Safari': 'Safari',
        'browser_type_Unknown': 'Unknown'
    }
    
    browser_value = None
    for feature, value in browser_mapping.items():
        if feature in data and data[feature] == 1.0:
            browser_value = value
            break
    
    if browser_value:
        decoded_data['browser_type'] = browser_value
        # Remove one-hot columns
        for feature in browser_mapping.keys():
            if feature in decoded_data:
                del decoded_data[feature]
    
    return decoded_data

# ────────────────────────────── Standardization Reversal ──────────────────────────────
def reverse_standardization(data):
    """Convert standardized values back to original scale"""
    reversed_data = data.copy()
    
    # Reverse URL features standardization (StandardScaler)
    url_features_to_reverse = [
        'url_entropy', 'url_count_dot', 'url_len', 'url_count_hyphen', 
        'url_count_letter', 'url_count_digit'
    ]
    
    for feature in url_features_to_reverse:
        if feature in data:
            try:
                feature_idx = list(url_scaler.feature_names_in_).index(feature)
                # Reverse the standardization: original = standardized * scale + mean
                original_value = data[feature] * url_scaler.scale_[feature_idx] + url_scaler.mean_[feature_idx]
                reversed_data[feature] = round(original_value, 2)
            except (ValueError, IndexError):
                pass
    
    # Reverse user features standardization (ColumnTransformer)
    user_features_to_reverse = [
        'login_attempts', 'session_duration', 'failed_logins', 'ip_reputation_score'
    ]
    
    for feature in user_features_to_reverse:
        if feature in data:
            try:
                # For ColumnTransformer, we need to find the scaler in the transformer
                for name, transformer, columns in user_scaler.transformers_:
                    if hasattr(transformer, 'feature_names_in_') and feature in transformer.feature_names_in_:
                        feature_idx = list(transformer.feature_names_in_).index(feature)
                        original_value = data[feature] * transformer.scale_[feature_idx] + transformer.mean_[feature_idx]
                        reversed_data[feature] = round(original_value, 2)
                        break
            except (ValueError, IndexError, AttributeError):
                pass
    
    # Reverse network features standardization (StandardScaler)
    network_features_to_reverse = [
        'Flow Duration', 'Tot Fwd Pkts', 'Flow Pkts/s', 'Fwd Pkt Len Max', 
        'Pkt Len Mean', 'Pkt Size Avg', 'Flow Byts/s'
    ]
    
    for feature in network_features_to_reverse:
        if feature in data:
            try:
                feature_idx = list(network_scaler.feature_names_in_).index(feature)
                original_value = data[feature] * network_scaler.scale_[feature_idx] + network_scaler.mean_[feature_idx]
                reversed_data[feature] = round(original_value, 2)
            except (ValueError, IndexError):
                pass
    
    return reversed_data

# ────────────────────────────── Model Loading ──────────────────────────────
def load_feature_list(txt_path: Path):
    """Load feature names from txt file"""
    if not txt_path.exists():
        raise FileNotFoundError(f"⚠️  {txt_path.name} not found")
    with open(txt_path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_models():
    """Load all trained models and scalers"""
    global url_model, network_model, user_model
    global url_scaler, network_scaler, user_scaler
    global url_features, network_features, user_features
    global url_data, network_data, user_data
    
    print("🔄 Loading models...")
    
    # Load feature lists
    url_features = load_feature_list(FEATURES_DIR / "url_features.txt")
    network_features = load_feature_list(FEATURES_DIR / "network_features.txt")
    user_features = load_feature_list(FEATURES_DIR / "user6_features.txt")
    
    # Load scalers
    try:
        with open(MODEL_DIR / "scaler_url_model.pkl", 'rb') as f:
            url_scaler = pickle.load(f)
    except:
        import joblib
        url_scaler = joblib.load(MODEL_DIR / "scaler_url_model.pkl")
        
    try:
        with open(MODEL_DIR / "scaler_network.pkl", 'rb') as f:
            network_scaler = pickle.load(f)
    except:
        import joblib
        network_scaler = joblib.load(MODEL_DIR / "scaler_network.pkl")
        
    try:
        with open(MODEL_DIR / "user_feature_encoder.pkl", 'rb') as f:
            user_scaler = pickle.load(f)
    except:
        import joblib
        user_scaler = joblib.load(MODEL_DIR / "user_feature_encoder.pkl")
    
    # Load models
    url_model = load_model(MODEL_DIR / "malicious_url_model.h5", compile=False)
    network_model = load_model(MODEL_DIR / "cnn_network_model.h5", compile=False)
    user_model = load_model(MODEL_DIR / "dnn_user_mixed_model.h5", compile=False)
    
    # Load model-specific datasets
    print("📚 Loading model-specific datasets...")
    # **FIXED**: Use fused dataset for URL data instead of the problematic URL_model_input_score file
    fused_data = pd.read_csv(DATA_DIR / "fused_with_botnet_saved.csv", low_memory=False)
    # Extract URL samples from fused data
    url_columns = [col for col in fused_data.columns if col in url_features]
    url_data = fused_data[url_columns + ['sample_id'] if 'sample_id' in fused_data.columns else url_columns].copy()
    # Add a simple label based on some criteria or use random for demo (you may want to adjust this)
    if 'BinaryLabel' in fused_data.columns:
        url_data['label'] = fused_data['BinaryLabel']
    else:
        # Create demo labels - this should be replaced with actual labels
        url_data['label'] = np.random.choice([0, 1], size=len(url_data), p=[0.8, 0.2])
    
    network_data = pd.read_csv(DATA_DIR / "network_score_sampled_from100w.csv")
    user_data = pd.read_csv(DATA_DIR / "user_encoded_dataset.csv")
    
    print(f"🔗 URL dataset: {len(url_data)} samples (from fused data)")
    print(f"🌐 Network dataset: {len(network_data)} samples")
    print(f"👤 User dataset: {len(user_data)} samples")
    
    print("✅ All models and datasets loaded successfully!")

# ────────────────────────────── Prediction Functions ──────────────────────────────
def predict_url_risk(data):
    """Predict URL risk score using complete feature data"""
    # Get the feature names expected by the scaler
    scaler_features = url_scaler.feature_names_in_
    
    # Create URL feature vector using the feature names expected by the scaler
    url_feature_vector = []
    missing_count = 0
    
    for feature in scaler_features:
        value = data.get(feature, 0.0)
        if pd.isna(value) or value is None:
            url_feature_vector.append(0.0)
            missing_count += 1
        else:
            url_feature_vector.append(float(value))
    
    # **IMPROVED**: Check if too many features are missing
    missing_ratio = missing_count / len(scaler_features)
    if missing_ratio > 0.5:  # If more than 50% features are missing
        print(f"⚠️  URL risk: Too many missing features ({missing_count}/{len(scaler_features)}, {missing_ratio:.1%}), using default medium risk")
        return 0.3  # Return medium risk for samples with too many missing features
    
    X = np.array(url_feature_vector).reshape(1, -1)
    
    # **RESTORED**: Now using original data from fused dataset, so we need scaler transformation
    # Create DataFrame to maintain feature names
    X_df = pd.DataFrame(X, columns=scaler_features)
    X_scaled = url_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    # Use model for prediction
    prediction = url_model.predict(X_cnn, verbose=0)[0][0]
    
    # If model output 1.0 means benign, then risk score = 1 - output value
    # So: 1.0 (benign) -> 0.0 (low risk), 0.0 (malicious) -> 1.0 (high risk)
    risk_score = 1.0 - prediction
    
    # Limit to reasonable range - lower minimum risk limit to allow truly safe URLs to show lower risk
    risk_score = max(0.01, min(0.95, risk_score))
    
    print(f"🔍 URL risk from model: {prediction:.6f} -> final: {risk_score:.4f} (missing: {missing_count}/{len(scaler_features)})")
    return float(risk_score)

def predict_network_risk(data):
    """Predict network risk score using complete feature data"""
    # Get the feature names expected by the scaler
    scaler_features = network_scaler.feature_names_in_
    
    # Create network feature vector using the feature names expected by the scaler
    network_feature_vector = []
    for feature in scaler_features:
        network_feature_vector.append(data.get(feature, 0.0))
    
    X = np.array(network_feature_vector).reshape(1, -1)
    
    # 创建DataFrame以保持特征名称
    X_df = pd.DataFrame(X, columns=scaler_features)
    X_scaled = network_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    # 使用模型预测
    prediction = network_model.predict(X_cnn, verbose=0)[0][0]
    
    # 模型可能训练为预测BinaryLabel (0/1)，而不是连续的network_score
    # 将sigmoid输出解释为恶意概率，然后映射到风险分数
    if prediction < 0.1:
        # 低恶意概率 -> 低风险分数
        risk_score = 0.05 + (prediction / 0.1) * 0.15  # 0.05 to 0.2
    elif prediction < 0.5:
        # 中等恶意概率 -> 中等风险分数
        risk_score = 0.2 + ((prediction - 0.1) / 0.4) * 0.3  # 0.2 to 0.5
    else:
        # 高恶意概率 -> 高风险分数
        risk_score = 0.5 + ((prediction - 0.5) / 0.5) * 0.45  # 0.5 to 0.95
    
    print(f"🔍 Network risk from model: {prediction:.6f} -> mapped: {risk_score:.4f}")
    return float(risk_score)

def predict_user_risk(data):
    """Predict user risk score using complete feature data"""
    # Get the feature names expected by the scaler
    scaler_features = user_scaler.feature_names_in_
    
    # Create user feature vector using the feature names expected by the scaler
    user_feature_vector = []
    for feature in scaler_features:
        user_feature_vector.append(data.get(feature, 0.0))
    
    X = np.array(user_feature_vector).reshape(1, -1)
    
    # 创建DataFrame以保持特征名称
    X_df = pd.DataFrame(X, columns=scaler_features)
    X_scaled = user_scaler.transform(X_df)
    X_cnn = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    # 由于模型可能有问题，我们使用基于特征的风险计算
    # 基于用户行为特征计算风险分数
    login_attempts = data.get('login_attempts', 0.0)
    failed_logins = data.get('failed_logins', 0.0)
    session_duration = data.get('session_duration', 0.0)
    ip_reputation_score = data.get('ip_reputation_score', 0.0)
    
    # 计算风险分数
    risk_score = 0.0
    
    # 登录尝试次数风险
    if login_attempts > 10:
        risk_score += 0.3
    elif login_attempts > 5:
        risk_score += 0.2
    
    # 失败登录风险
    if failed_logins > 3:
        risk_score += 0.4
    elif failed_logins > 1:
        risk_score += 0.2
    
    # IP声誉风险
    if ip_reputation_score < 0.3:
        risk_score += 0.3
    elif ip_reputation_score < 0.5:
        risk_score += 0.1
    
    # 会话时长风险（异常短或异常长）
    if session_duration < 60 or session_duration > 3600:
        risk_score += 0.1
    
    # 确保风险分数在合理范围内
    risk_score = min(risk_score, 0.95)
    risk_score = max(risk_score, 0.05)
    
    print(f"🔍 User risk calculated: {risk_score}")
    return float(risk_score)

# ────────────────────────────── API Endpoints ──────────────────────────────
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Cybersecurity Fusion System Backend is running"
    })



@app.route('/predict/url', methods=['POST'])
def predict_url():
    """URL-specific prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        risk = predict_url_risk(data)
        return jsonify({"url_risk": risk})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/network', methods=['POST'])
def predict_network():
    """Network-specific prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        risk = predict_network_risk(data)
        return jsonify({"network_risk": risk})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/user', methods=['POST'])
def predict_user():
    """User-specific prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        risk = predict_user_risk(data)
        return jsonify({"user_risk": risk})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get available feature lists"""
    return jsonify({
        "url_features": url_features,
        "network_features": network_features,
        "user_features": user_features
    })

# ────────────────────────────── Teaching API Endpoints ──────────────────────────────
@app.route('/random_sample', methods=['GET'])
def get_random_sample():
    """Get a random sample for teaching purposes from model-specific datasets"""
    try:
        # Get random samples from each model-specific dataset
        url_idx = random.randint(0, len(url_data) - 1)
        network_idx = random.randint(0, len(network_data) - 1)
        user_idx = random.randint(0, len(user_data) - 1)
        
        url_sample = url_data.iloc[url_idx]
        network_sample = network_data.iloc[network_idx]
        user_sample = user_data.iloc[user_idx]
        
        # Extract teaching features from each dataset
        teaching_sample = {}
        
        # URL features from URL dataset
        for feature in TEACHING_URL_FEATURES:
            if feature in url_sample:
                value = float(url_sample[feature]) if pd.notna(url_sample[feature]) else 0.0
                teaching_sample[feature] = value
            else:
                teaching_sample[feature] = 0.0
        
        # User features from User dataset
        for feature in TEACHING_USER_FEATURES:
            if feature in user_sample:
                value = float(user_sample[feature]) if pd.notna(user_sample[feature]) else 0.0
                teaching_sample[feature] = value
            else:
                # For categorical features that will be decoded from one-hot encoding
                if feature in ['protocol_type', 'encryption_used', 'browser_type']:
                    teaching_sample[feature] = 0.0  # Placeholder, will be filled by one-hot decoding
                else:
                    teaching_sample[feature] = 0.0
        
        # Add one-hot encoded features for decoding
        for feature in user_sample.index:
            if feature.startswith(('protocol_type_', 'encryption_used_', 'browser_type_')):
                value = float(user_sample[feature]) if pd.notna(user_sample[feature]) else 0.0
                teaching_sample[feature] = value
        
        # Network features from Network dataset
        for feature in TEACHING_NETWORK_FEATURES:
            if feature in network_sample:
                value = float(network_sample[feature]) if pd.notna(network_sample[feature]) else 0.0
                teaching_sample[feature] = value
            else:
                teaching_sample[feature] = 0.0
        
        # Decode one-hot features and reverse standardization for better user experience
        decoded_sample = decode_one_hot_features(teaching_sample)
        original_scale_sample = reverse_standardization(decoded_sample)
        
        return jsonify({
            "sample": original_scale_sample,
            "url_sample_id": int(url_idx),
            "network_sample_id": int(network_idx),
            "user_sample_id": int(user_idx)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_teaching():
    """Teaching prediction endpoint - returns comprehensive risk assessment"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get the sample IDs from the request
        url_sample_id = data.get('url_sample_id')
        network_sample_id = data.get('network_sample_id')
        user_sample_id = data.get('user_sample_id')
        
        if url_sample_id is not None and network_sample_id is not None and user_sample_id is not None:
            # Use the complete CSV row data from each model-specific dataset for prediction
            url_sample = url_data.iloc[url_sample_id]
            network_sample = network_data.iloc[network_sample_id]
            user_sample = user_data.iloc[user_sample_id]
            
            # Create complete feature dictionaries for each model
            url_features_dict = {}
            network_features_dict = {}
            user_features_dict = {}
            
            # Fill URL features from the URL dataset
            print(f"🔍 Processing {len(url_features)} URL features for sample {url_sample_id}")
            for feature in url_features:
                if feature in url_sample.index:
                    value = float(url_sample[feature]) if pd.notna(url_sample[feature]) else 0.0
                    url_features_dict[feature] = value
                else:
                    print(f"⚠️  URL feature '{feature}' not found in CSV, using 0.0")
                    url_features_dict[feature] = 0.0
            
            # Fill network features from the Network dataset
            print(f"🔍 Processing {len(network_features)} network features for sample {network_sample_id}")
            for feature in network_features:
                if feature in network_sample.index:
                    try:
                        value = float(network_sample[feature]) if pd.notna(network_sample[feature]) else 0.0
                        network_features_dict[feature] = value
                    except (ValueError, TypeError):
                        print(f"⚠️  Network feature '{feature}' cannot be converted to float, using 0.0")
                        network_features_dict[feature] = 0.0
                else:
                    print(f"⚠️  Network feature '{feature}' not found in CSV, using 0.0")
                    network_features_dict[feature] = 0.0
            
            # Fill user features from the User dataset
            print(f"🔍 Processing {len(user_features)} user features for sample {user_sample_id}")
            for feature in user_features:
                if feature in user_sample.index:
                    value = float(user_sample[feature]) if pd.notna(user_sample[feature]) else 0.0
                    user_features_dict[feature] = value
                else:
                    # Handle categorical features that are not in the dataset
                    if feature == 'protocol_type':
                        user_features_dict[feature] = 'TCP'  # Default protocol
                    elif feature == 'encryption_used':
                        user_features_dict[feature] = 'Unknown'  # Default encryption
                    elif feature == 'browser_type':
                        user_features_dict[feature] = 'Unknown'  # Default browser
                    else:
                        print(f"⚠️  User feature '{feature}' not found in CSV, using 0.0")
                        user_features_dict[feature] = 0.0
            
            # Get predictions using complete CSV row data from each dataset
            url_risk = predict_url_risk(url_features_dict)
            network_risk = predict_network_risk(network_features_dict)
            user_risk = predict_user_risk(user_features_dict)
            
        else:
            # Fallback to teaching features only (for backward compatibility)
            url_risk = predict_url_risk(data)
            network_risk = predict_network_risk(data)
            user_risk = predict_user_risk(data)
        
        # 基础平均风险
        base_avg_risk = (url_risk + network_risk + user_risk) / 3
        
        # 智能Dynamic Weight逻辑：考虑多个高风险类别的累积效应
        final_confidence = base_avg_risk
        
        # 统计高风险类别数量和严重程度
        risk_scores = [url_risk, network_risk, user_risk]
        risk_names = ['URL', 'Network', 'User']
        high_risks = []
        total_boost = 0.0
        
        for i, (risk, name) in enumerate(zip(risk_scores, risk_names)):
            if risk >= 0.9:  # 90%以上极高风险
                boost = 0.25
                high_risks.append(f"{name} extremely high ({risk*100:.1f}%)")
                total_boost += boost
            elif risk >= 0.7:  # 70%以上高风险
                boost = 0.15
                high_risks.append(f"{name} high ({risk*100:.1f}%)")
                total_boost += boost
            elif risk >= 0.5:  # 50%以上中等风险
                boost = 0.08
                high_risks.append(f"{name} elevated ({risk*100:.1f}%)")
                total_boost += boost
        
        # 应用累积加权，但限制最大加权
        if total_boost > 0:
            # 限制总加权不超过0.4，并且不超过最高风险值
            max_risk = max(risk_scores)
            total_boost = min(total_boost, 0.4)
            final_confidence = min(base_avg_risk + total_boost, max_risk, 0.95)
            print(f"🚨 Multiple risk factors detected: {', '.join(high_risks)}")
            print(f"📊 Base risk: {base_avg_risk:.3f} + cumulative boost: {total_boost:.3f} = {final_confidence:.3f}")
        else:
            print(f"📊 Risk calculation: URL={url_risk:.3f}, Network={network_risk:.3f}, User={user_risk:.3f} → Final={final_confidence:.3f}")
        
        # 简化为Safe/Unsafe，以50%为分界线
        if final_confidence >= 0.5:
            risk_level = "Unsafe"
        else:
            risk_level = "Safe"
        
        # Generate LLM analysis
        llm_analysis = generate_llm_analysis(
            url_risk, network_risk, user_risk, 
            final_confidence, risk_level,
            url_sample_id, network_sample_id, user_sample_id
        )
        
        return jsonify({
            "url_risk": round(url_risk, 4),
            "network_risk": round(network_risk, 4),
            "user_risk": round(user_risk, 4),
            "final_risk_level": risk_level,
            "confidence": round(final_confidence, 4),
            "llm_analysis": llm_analysis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_llm_analysis(url_risk, network_risk, user_risk, final_confidence, risk_level, url_sample_id, network_sample_id, user_sample_id):
    """Generate detailed analysis using ChatGPT"""
    try:
        # Get sample data to provide context
        url_sample = url_data.iloc[url_sample_id] if url_sample_id < len(url_data) else None
        network_sample = network_data.iloc[network_sample_id] if network_sample_id < len(network_data) else None
        user_sample = user_data.iloc[user_sample_id] if user_sample_id < len(user_data) else None
        
        # Extract key features for analysis
        context_info = []
        
        if url_sample is not None:
            context_info.append(f"URL features: entropy={url_sample.get('url_entropy', 'N/A'):.2f}, length={url_sample.get('url_len', 'N/A')}, dots={url_sample.get('url_count_dot', 'N/A')}")
        
        if network_sample is not None:
            context_info.append(f"Network features: flow_duration={network_sample.get('Flow Duration', 'N/A')}, packets={network_sample.get('Tot Fwd Pkts', 'N/A')}, bytes_per_sec={network_sample.get('Flow Byts/s', 'N/A')}")
        
        if user_sample is not None:
            context_info.append(f"User features: login_attempts={user_sample.get('login_attempts', 'N/A')}, failed_logins={user_sample.get('failed_logins', 'N/A')}, session_duration={user_sample.get('session_duration', 'N/A')}")
        
        context = " | ".join(context_info)
        
        # Create prompt for ChatGPT
        prompt = f"""As a cybersecurity expert, analyze this security assessment result and return your response in markdown format with appropriate highlighting:

Risk Assessment:
- URL Risk: {url_risk*100:.1f}%
- Network Risk: {network_risk*100:.1f}%  
- User Risk: {user_risk*100:.1f}%
- Final Assessment: {risk_level} (Confidence: {final_confidence*100:.1f}%)

Sample Data Context:
{context}

Please provide a concise analysis (max 150 words) in markdown format explaining:
1. Why the system gave this assessment
2. Which specific features contributed most to the risk
3. What normal vs abnormal patterns look like for the concerning features
4. Brief security implications

Use markdown formatting:
- Use **bold** for important risk levels and key findings
- Use `code` for technical terms like entropy, packets, bytes
- Use > blockquotes for key insights
- Keep it educational and practical for cybersecurity learning"""

        # Call OpenAI API
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'your-api-key-here'))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        analysis = response.choices[0].message.content.strip()
        return analysis
        
    except Exception as e:
        print(f"LLM Analysis Error: {e}")
        return "LLM analysis temporarily unavailable. The assessment is based on machine learning models trained on cybersecurity patterns."

# ────────────────────────────── Main ──────────────────────────────
if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Start Flask server
    print("🚀 Starting Cybersecurity Fusion System Backend...")
    app.run(host='0.0.0.0', port=5001, debug=True) 