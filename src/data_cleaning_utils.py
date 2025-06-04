import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict

def print_dataset_info(df: pd.DataFrame, dataset_name: str) -> None:
    """Print basic information about the dataset."""
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print(f"{'='*50}\n")

def handle_missing_values(df: pd.DataFrame, numeric_strategy: str = 'mean',
                        categorical_fill: str = 'None') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        numeric_strategy: Strategy for numeric columns ('mean' or 'median')
        categorical_fill: Value to fill categorical columns
    """
    df_cleaned = df.copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = df[col].median()
            df_cleaned[col].fillna(fill_value, inplace=True)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if categorical_fill == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = categorical_fill
            df_cleaned[col].fillna(fill_value, inplace=True)
    
    return df_cleaned

def handle_outliers(df: pd.DataFrame, columns: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Handle outliers by clipping values to specified ranges.
    
    Args:
        df: Input DataFrame
        columns: Dictionary of column names and their valid ranges (min, max)
    """
    df_cleaned = df.copy()
    
    for col, (min_val, max_val) in columns.items():
        if col in df.columns:
            df_cleaned[col] = df_cleaned[col].clip(min_val, max_val)
            
    return df_cleaned

def encode_categorical_features(df: pd.DataFrame, columns: List[str],
                              encoding_type: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features using specified encoding method.
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        encoding_type: Type of encoding ('onehot' or 'label')
    """
    df_encoded = df.copy()
    
    if encoding_type == 'onehot':
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
    else:  # label encoding
        le = LabelEncoder()
        for col in columns:
            if col in df.columns:
                df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    return df_encoded

def standardize_features(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Standardize numeric features to [0,1] range.
    
    Args:
        df: Input DataFrame
        columns: List of numeric columns to standardize
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    
    # Filter only existing columns
    columns = [col for col in columns if col in df.columns]
    
    if columns:
        # Replace infinite values with NaN
        df_scaled[columns] = df_scaled[columns].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median of the column
        for col in columns:
            df_scaled[col] = df_scaled[col].fillna(df_scaled[col].median())
        
        # Now scale the features
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled, scaler

def select_features(df: pd.DataFrame, target_col: str, n_features: int = 10,
                   exclude_cols: List[str] = None) -> Tuple[List[str], pd.Series]:
    """
    Select most important features using RandomForestClassifier.
    
    Args:
        df: Input DataFrame
        target_col: Target variable column name
        n_features: Number of features to select
        exclude_cols: Columns to exclude from feature selection
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Prepare feature matrix
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    
    # Initialize and fit RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=60, max_depth=50, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    
    # Select top features
    selected_features = importance.head(n_features).index.tolist()
    
    return selected_features, importance 