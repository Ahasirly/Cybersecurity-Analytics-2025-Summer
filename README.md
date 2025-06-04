# Cybersecurity Detection and User Behavior Prediction

## Project Overview
- **Project Name**: 2025 Summer Direct Study Proposal
- **Reference**: IEEE Paper "Research on Malicious URL Detection Based on Random Forest"

## Dataset Description

### 1. CIC_Dataset.csv
- **Records**: 15,367
- **Features**: 78 columns
- **Main Features**:
  - urlLen
  - NumberofDotsinURL
  - Querylength
  - domain_token_count
  - path_token_count
- **Target Variable**: URL_Type_obf_Type (Benign, Defacement, Spam)

### 2. merged_dataset.csv
- **Time Range**: 2018-2024
- **Feature Types**:
  - **Network Traffic Features**:
    - CPU_Utilization
    - Phishing_Attempts
    - Risky_Website_Visits
    - Anomaly_Score
  - **Behavioral Features**:
    - Device_Type
    - Age_Group
    - Social_Media_Usage
    - E_Safety_Awareness_Score
- **Target Variables**:
  - Insecurity_Level (Binary)
  - Hours_Online (Continuous)
  - Cybersecurity_Behavior_Category (Multi-class)

## Project Structure
```
Cybersecurity-Analytics-2025-Summer/
├── src/
│   ├── data_cleaning_utils.py  # Data cleaning utility functions
│   └── clean_datasets.py       # Main data cleaning script
├── Datasets/
│   ├── original/              # Original datasets
│   ├── train/                 # Training datasets
│   └── test/                  # Test datasets
├── venv/                      # Python virtual environment
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Data Preprocessing Implementation

### 1. Utility Functions (data_cleaning_utils.py)
- **print_dataset_info**: Display dataset basic information
- **handle_missing_values**: Handle missing values
  - Numeric features: mean/median imputation
  - Categorical features: mode/'None' imputation
- **handle_outliers**: Outlier treatment
  - URL features: 99th percentile limit
  - Time features: [0,24] limit
- **encode_categorical_features**: Feature encoding
  - One-hot encoding
  - Label encoding
- **standardize_features**: Feature standardization
- **select_features**: Random forest-based feature selection

### 2. Data Cleaning Process (clean_datasets.py)

#### CIC Dataset Cleaning
1. Missing Value Treatment
   - Numeric features: mean imputation
   - Categorical features: mode imputation
2. Outlier Treatment
   - urlLen etc.: 99th percentile limit
3. Feature Processing
   - URL_Type_obf_Type to binary
   - Numeric features standardization to [0,1]
4. Feature Selection
   - Select top 10 features using random forest
5. Data Split
   - 80% training set
   - 20% test set

#### Merged Dataset Cleaning
1. Time Feature Processing
   - Extract Hour, DayOfWeek, Month
2. Missing Value Treatment
   - Numeric features: median imputation
   - Categorical features: "None" imputation
3. Outlier Treatment
   - Hours_Online: limit [0,24]
   - CPU_Utilization: limit [0,100]
4. Feature Encoding
   - Categorical features: one-hot encoding
   - Target variables: label encoding
5. Feature Selection
   - Select top 15 important features
6. Time Series Split
   - 2018-2022 training set
   - 2023-2024 test set

## Project Dependencies
```python
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Usage Instructions

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Data Cleaning
```bash
# Navigate to src directory
cd src

# Run data cleaning script
python clean_datasets.py
```

## Next Steps
1. Model Training (June 5)
   - Model 1A: URL Detection
   - Model 1B: Network Traffic Detection
   - Model 2: Online Time Prediction
   - Model 3: Behavior Risk Classification
2. Browser Deployment Implementation
3. Real-time Detection Testing
4. Proposal Report Submission

## Expected Outcomes
- Model 1: Accuracy >98.6%
- Model 2: Minimum MSE, R² close to 1
- Model 3: High F1 score, accurate high-risk user identification

## Timeline
- June 4: Complete data cleaning
- June 5: Start model training
- Mid-June: Implementation deployment
- End of June: Submit report