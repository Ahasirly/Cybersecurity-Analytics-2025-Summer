import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from data_cleaning_utils import (
    print_dataset_info,
    handle_missing_values,
    handle_outliers,
    encode_categorical_features,
    standardize_features,
    select_features
)
from sklearn.ensemble import RandomForestClassifier

def clean_cic_dataset(input_path: str, output_path: str) -> None:
    """Clean and preprocess the CIC Dataset."""
    print("\nProcessing CIC Dataset...")
    
    # Load dataset
    df = pd.read_csv(input_path)
    print_dataset_info(df, "CIC Dataset (Original)")
    
    # Handle missing values
    df = handle_missing_values(df, numeric_strategy='mean', categorical_fill='mode')
    
    # Handle outliers using 99th percentile for numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_ranges = {}
    for col in numeric_cols:
        max_val = df[col].quantile(0.99)
        min_val = df[col].quantile(0.01)
        outlier_ranges[col] = (min_val, max_val)
    df = handle_outliers(df, outlier_ranges)
    
    # Convert URL_Type_obf_Type to binary
    df['URL_Type_obf_Type'] = (df['URL_Type_obf_Type'] != 'Benign').astype(int)
    
    # Standardize numeric features
    df, _ = standardize_features(df, numeric_cols)
    
    # Select top features
    selected_features, importance = select_features(
        df, 'URL_Type_obf_Type', n_features=10
    )
    print("\nTop 10 important features for CIC Dataset:")
    print(importance.head(10))
    
    # Keep only selected features and target
    df = df[selected_features + ['URL_Type_obf_Type']]
    
    # Split data
    X = df.drop('URL_Type_obf_Type', axis=1)
    y = df['URL_Type_obf_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Combine and save
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save cleaned datasets
    train_df.to_csv(output_path.replace('.csv', '_train.csv'), index=False)
    test_df.to_csv(output_path.replace('.csv', '_test.csv'), index=False)
    
    print("\nCIC Dataset cleaning completed!")
    print_dataset_info(train_df, "CIC Dataset (Cleaned - Train)")
    print_dataset_info(test_df, "CIC Dataset (Cleaned - Test)")

def clean_merged_dataset(input_path: str, output_path: str) -> None:
    """Clean and preprocess the merged dataset."""
    print("\nProcessing merged dataset...")
    
    # Load dataset
    df = pd.read_csv(input_path)
    print_dataset_info(df, "Merged Dataset (Original)")
    
    # Convert Time column to datetime and extract features
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Month'] = df['Time'].dt.month
    
    # Drop unnecessary columns first
    columns_to_drop = ['Time', 'Source_IP', 'Destination_IP', 'Protocol_Type', 
                      'Attack_Severity', 'Botnet_Family', 'Malware_Type', 'Label',
                      'Password_Strength', 'Education_Content_Usage', 'Geolocation',
                      'Network_Type', 'Peer_Interactions', 'E_Safety_Awareness_Score']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    # Handle missing values in target variables first
    df['Insecurity_Level'] = df['Insecurity_Level'].fillna('Safe')
    df['Cybersecurity_Behavior_Category'] = df['Cybersecurity_Behavior_Category'].fillna('Neutral')
    
    # Handle missing values in other columns
    df = handle_missing_values(df, numeric_strategy='median', categorical_fill='None')
    
    # Handle outliers for specific columns
    outlier_ranges = {
        'Hours_Online': (0, 24),
        'CPU_Utilization': (0, 100),
        'Flow_Bytes_per_s': (0, df['Flow_Bytes_per_s'].quantile(0.99)),
        'Packet_Size': (0, df['Packet_Size'].quantile(0.99))
    }
    df = handle_outliers(df, outlier_ranges)
    
    # Encode categorical features
    categorical_cols = [
        'Device_Type', 'Social_Media_Usage', 'Attack_Vector',
        'Age_Group', 'System_Patch_Status'
    ]
    df = encode_categorical_features(df, categorical_cols, encoding_type='onehot')
    
    # Encode target variables
    df['Insecurity_Level'] = df['Insecurity_Level'].map({'Safe': 0, 'Unsafe': 1})
    behavior_mapping = {'Safe': 0, 'Neutral': 1, 'Risky': 2}
    df['Cybersecurity_Behavior_Category'] = df['Cybersecurity_Behavior_Category'].map(behavior_mapping)
    
    # Standardize numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = numeric_cols.drop(['Insecurity_Level', 'Hours_Online', 'Cybersecurity_Behavior_Category'])
    df, _ = standardize_features(df, numeric_cols)
    
    # Remove any remaining NaN values
    df = df.fillna(0)
    
    # Select features for Insecurity_Level prediction
    feature_cols = [col for col in df.columns if col not in ['Insecurity_Level', 'Hours_Online', 'Cybersecurity_Behavior_Category']]
    X = df[feature_cols]
    y = df['Insecurity_Level']
    
    # Initialize and fit RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=60, max_depth=50, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    selected_features = importance.head(15).index.tolist()
    
    print("\nTop 15 important features for Insecurity_Level prediction:")
    print(importance.head(15))
    
    # Keep only selected features and target variables
    df = df[selected_features + ['Insecurity_Level', 'Hours_Online', 'Cybersecurity_Behavior_Category']]
    
    # Split data by time
    train_mask = df['Month'] < 6  # Using first 5 months for training
    train_df = df[train_mask]
    test_df = df[~train_mask]
    
    # Save cleaned datasets
    train_df.to_csv(output_path.replace('.csv', '_train.csv'), index=False)
    test_df.to_csv(output_path.replace('.csv', '_test.csv'), index=False)
    
    print("\nMerged Dataset cleaning completed!")
    print_dataset_info(train_df, "Merged Dataset (Cleaned - Train)")
    print_dataset_info(test_df, "Merged Dataset (Cleaned - Test)")

def main():
    """Main function to clean both datasets."""
    # Set input and output paths
    cic_input = "Datasets/CIC_Dataset.csv"
    cic_output = "Datasets/CIC_Dataset_cleaned.csv"
    merged_input = "Datasets/merged_dataset.csv"
    merged_output = "Datasets/merged_dataset_cleaned.csv"
    
    # Clean datasets
    clean_cic_dataset(cic_input, cic_output)
    clean_merged_dataset(merged_input, merged_output)
    
    print("\nAll datasets cleaned successfully!")

if __name__ == "__main__":
    main() 