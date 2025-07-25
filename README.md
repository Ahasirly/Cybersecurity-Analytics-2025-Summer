# Cybersecurity Fusion System

A comprehensive cybersecurity risk assessment system that combines URL, network, and user behavior analysis using machine learning models.

## Project Structure

```
cybersec_fusion_system/
├── backend/                ← Flask 后端服务
│   ├── app.py              ← 主程序：加载模型 + 提供 API
│   ├── models/             ← 所有模型文件 (.h5, .pkl, .cbm)
│   ├── features/           ← 特征列 txt 文件
│   ├── data/               ← 原始数据文件
│   ├── outputs/            ← 输出结果
│   ├── llm_prompts/        ← LLM 提示词
│   └── requirements.txt    ← Python 依赖
├── frontend/               ← React 前端 (待开发)
│   └── ...
└── scripts/                ← 测试脚本
```

## Backend API Endpoints

- `GET /health` - Health check
- `POST /predict` - Get all risk scores (URL, Network, User)
- `POST /predict/url` - URL risk prediction only
- `POST /predict/network` - Network risk prediction only  
- `POST /predict/user` - User risk prediction only
- `GET /features` - Get available feature lists

## Setup Instructions

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

Frontend development is pending. The backend API is ready for integration.

## Model Information

The system uses three trained models:

1. **URL Model** (`malicious_url_model.h5`) - Detects malicious URLs
2. **Network Model** (`cnn_network_model_finetuned_v3_final.h5`) - Analyzes network traffic patterns
3. **User Model** (`dnn_model_user.h5`) - Assesses user behavior risk

Each model has corresponding scaler files and feature lists for preprocessing.

## Testing

Use the test script to validate the system:
```bash
python scripts/run_local_test.py
```

This will process sample data and output pattern-level risk scores. 