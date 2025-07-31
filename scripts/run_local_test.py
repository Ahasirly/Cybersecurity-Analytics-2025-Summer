#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run 3-model inference on pattern20_samples.csv
and export pattern-level risk scores to outputs/pattern_scores.txt
"""

import os, pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ────────────────────────────── paths ──────────────────────────────
BASE = Path(__file__).resolve().parent.parent   # CYBERSEC_FUSION/
DATA_DIR, MODEL_DIR, OUT_DIR = BASE/"backend/data", BASE/"backend/models", BASE/"backend/outputs"
FEATURES_DIR = BASE/"backend/features"  # Feature files directory
OUT_DIR.mkdir(exist_ok=True)

CSV_SRC        = DATA_DIR / "fused_with_botnet_saved.csv"
TXT_OUT        = OUT_DIR  / "pattern_scores.txt"

# ─────────────────────── feature-list helper ───────────────────────
def load_feature_list(txt_path: Path):
    if not txt_path.exists():
        raise FileNotFoundError(f"⚠️  {txt_path.name} not found - please save the feature names used in training as a txt file")
    with open(txt_path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ─────────────────────── prediction helper ────────────────────────
def predict_block(df, cols, scaler_pkl, model_h5):
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # Fill NaN values with 0
    df[cols] = df[cols].fillna(0)
    X = df[cols].values.astype(np.float32)

    if np.isnan(X).any():
        raise ValueError("❌ NaN detected in CNN input, please check your data!")

    try:
        with open(scaler_pkl, 'rb') as f:
            scaler = pickle.load(f)
    except:
        # Try joblib as fallback
        scaler = joblib.load(scaler_pkl)
    Xs = scaler.transform(X)
    Xs_cnn = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))

    model = load_model(model_h5, compile=False)
    return model.predict(Xs_cnn, verbose=0).flatten()

# ─────────────────────────── 0. Load data ──────────────────────────
if not CSV_SRC.exists():
    raise FileNotFoundError(f"❌ {CSV_SRC} does not exist, please make sure the sample CSV is in data/")

df = pd.read_csv(CSV_SRC)
assert "pattern" in df.columns, "CSV is missing the 'pattern' column!"

# ────────────────────────── 1. URL model ───────────────────────────
url_cols  = load_feature_list(FEATURES_DIR / "url_features.txt")
df["url_risk"] = predict_block(
    df, url_cols,
    MODEL_DIR / "scaler_url_model.pkl",
    MODEL_DIR / "malicious_url_model.h5"
)

# ───────────────────────── 2. Network model ────────────────────────
net_cols  = load_feature_list(FEATURES_DIR / "network_features.txt")
df["net_risk"] = predict_block(
    df, net_cols,
    MODEL_DIR / "scaler_network_finetuned_v3_final.pkl",
    MODEL_DIR / "cnn_network_model_finetuned_v3_final.h5"
)

# Calculate network traffic risk score (0~1) using enhanced differentiation method
net_risk_normalized = (df["network_risk"] - df["network_risk"].min()) / (df["network_risk"].max() - df["network_risk"].min())
df["net_risk_score"] = net_risk_normalized ** 0.5  # Use square root to enhance low values and compress high values

print("📊 First 10 CNN network risk outputs:", df["net_risk"].values[:10])
print("📈 Network risk stats:\n", df["net_risk"].describe())

# ───────────────────────── 3. User model ───────────────────────────
user_cols = load_feature_list(FEATURES_DIR / "user6_features.txt")
df["user_risk"] = predict_block(
    df, user_cols,
    MODEL_DIR / "scaler_user_model_6cols.pkl",
    MODEL_DIR / "dnn_model_user.h5"
)

# ───────────────────────── 4. Aggregate by pattern ─────────────────
agg = (df.groupby("pattern")
         .agg(samples         = ("pattern", "size"),
              url_risk        = ("url_risk",  "mean"),
              network_risk    = ("net_risk",  "mean"),
              user_risk       = ("user_risk", "mean"),
              net_risk_score  = ("net_risk_score", "mean"))
         .reset_index()
         .sort_values("pattern"))

# ───────────────────────── 5. Export result to TXT ─────────────────
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write(f"# Pattern-level risk scores  ({ts})\n")
    f.write("# pattern,samples,url_risk,network_risk,user_risk,net_risk_score\n")
    for _, r in agg.iterrows():
        f.write(f"{r['pattern']},{r['samples']},"
                f"{r['url_risk']:.4f},{r['network_risk']:.4f},"
                f"{r['user_risk']:.4f},{r['net_risk_score']:.4f}\n")

print("✅ Done: Results written to", TXT_OUT.relative_to(BASE))
print("\n📄 Preview first 5 rows:\n", agg.head())
