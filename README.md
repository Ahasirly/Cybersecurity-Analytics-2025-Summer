# Cybersecurity Fusion System

A comprehensive AI-powered cybersecurity threat detection and analysis system that combines multiple machine learning models for real-time security assessment.

## Overview

This project implements an integrated cybersecurity analytics platform that provides:

- **URL Threat Detection**: Machine learning-based malicious URL identification
- **Network Traffic Analysis**: Real-time network flow anomaly detection  
- **User Behavior Analytics**: Behavioral pattern analysis for insider threat detection
- **Interactive Learning Interface**: Educational platform for cybersecurity training

The system features a Flask-based backend with pre-trained AI models and a React TypeScript frontend for real-time threat visualization and user interaction.

## Architecture

```
cybersec_fusion_system/
├── backend/                 # Flask API server
│   ├── app.py              # Main application
│   ├── models/             # Pre-trained ML models
│   ├── data/               # Training datasets
│   ├── features/           # Feature definitions
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript UI
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Node.js dependencies
└── scripts/               # Utility scripts
```

## Prerequisites

Before setting up the project, ensure you have the following installed:

### Required Software

1. **Python 3.11** (recommended for compatibility)
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python3.11 --version`

2. **Node.js** (version 16 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version` and `npm --version`

3. **Git**
   - Download from: https://git-scm.com/
   - Verify installation: `git --version`

### System Dependencies

**For macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11
```

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install nodejs npm
```

**For Windows:**
- Install Python 3.11 from the official website
- Install Node.js from the official website
- Use PowerShell or Command Prompt for commands

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ahasirly/Cybersecurity-Analytics-2025-Summer.git
cd Cybersecurity-Analytics-2025-Summer
```

### Step 2: Backend Setup

#### 2.1 Navigate to Backend Directory
```bash
cd backend
```

#### 2.2 Create Virtual Environment
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 2.3 Install Python Dependencies
```bash
# Ensure you're in the virtual environment (should see (venv) in prompt)
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you encounter scikit-learn version conflicts, install the specific version:
```bash
pip install scikit-learn==1.6.1
```

#### 2.4 Verify Backend Installation
```bash
python app.py
```

You should see output similar to:
```
Loading models...
Loading model-specific datasets...
URL dataset: 10000 samples
Network dataset: 50000 samples
User dataset: 9537 samples
All models and datasets loaded successfully!
Starting Cybersecurity Fusion System Backend...
* Running on http://127.0.0.1:5001
```

Press `Ctrl+C` to stop the server for now.

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory
```bash
# From the project root directory
cd frontend
```

#### 3.2 Install Node.js Dependencies
```bash
npm install
```

**If you encounter dependency issues:**
```bash
# Install missing dependencies
npm install source-map-js
npm audit fix
```

#### 3.3 Verify Frontend Installation
```bash
npm start
```

The frontend should compile successfully and display:
```
Compiled successfully!
You can now view frontend in the browser.
Local: http://localhost:3000
```

## Running the Application

### Method 1: Manual Startup (Recommended for Development)

#### Terminal 1 - Backend Server
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

#### Terminal 2 - Frontend Server
```bash
cd frontend
npm start
```

### Method 2: Using the Startup Script

```bash
# Make the script executable (macOS/Linux only)
chmod +x start_backend.sh
./start_backend.sh
```

Then in another terminal:
```bash
cd frontend && npm start
```

## Accessing the Application

Once both servers are running:

- **Frontend Interface**: http://localhost:3000
- **Backend API**: http://localhost:5001
- **Health Check**: http://localhost:5001/health

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "message": "Cybersecurity Fusion System Backend is running"}
```

### Get Random Sample
```
GET /random_sample
Response: {
  "sample": {...},
  "url_sample_id": 123,
  "network_sample_id": 456,
  "user_sample_id": 789
}
```

### Predict Risk
```
POST /predict
Body: {
  "url_sample_id": 123,
  "network_sample_id": 456,
  "user_sample_id": 789
}
Response: {
  "url_risk": 0.05,
  "network_risk": 0.59,
  "user_risk": 0.20,
  "final_risk_level": "High",
  "confidence": 0.28
}
```

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility
**Error**: `ModuleNotFoundError` or version conflicts
**Solution**: Ensure you're using Python 3.11 and have activated the virtual environment

#### 2. Backend Port Already in Use
**Error**: `Address already in use`
**Solution**: 
```bash
# Find and kill the process using port 5001
lsof -ti:5001 | xargs kill -9
```

#### 3. Frontend Compilation Errors
**Error**: Missing dependencies or compilation failures
**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm install source-map-js
npm start
```

#### 4. Model Loading Errors
**Error**: `AttributeError` or pickle errors
**Solution**: Ensure scikit-learn version 1.6.1 is installed:
```bash
pip install scikit-learn==1.6.1
```

### Checking Service Status

```bash
# Check if backend is running
curl http://localhost:5001/health

# Check if frontend is accessible
curl http://localhost:3000
```

## Development Guidelines

### Backend Development
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Update requirements.txt when adding dependencies
- Test API endpoints with curl or Postman

### Frontend Development
- Use TypeScript for all new components
- Follow React best practices
- Maintain responsive design principles
- Test components across different browsers

## Model Information

The system includes pre-trained models for:

1. **URL Detection Model** (`malicious_url_model.h5`)
   - Architecture: Convolutional Neural Network
   - Features: 56 URL-based features
   - Accuracy: >95%

2. **Network Traffic Model** (`cnn_network_model.h5`)
   - Architecture: CNN for sequential data
   - Features: 78 network flow features
   - Detection: Various attack types (DDoS, Botnet, etc.)

3. **User Behavior Model** (`dnn_user_mixed_model.h5`)
   - Architecture: Deep Neural Network
   - Features: 19 behavioral features
   - Purpose: Insider threat detection

## Dataset Information

- **URL Dataset**: 10,000 samples with benign/malicious labels
- **Network Dataset**: 50,000 network flow records
- **User Dataset**: 9,537 user behavior profiles

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is developed for academic and research purposes. Please refer to the license file for usage terms.

## Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed error information
4. Include system information (OS, Python version, Node.js version)

## Acknowledgments

This project was developed as part of the 2025 Summer Cybersecurity Analytics research initiative, incorporating state-of-the-art machine learning techniques for comprehensive threat detection and analysis. 