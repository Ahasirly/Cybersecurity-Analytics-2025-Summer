"""
UCI Phishing Websites Dataset Preprocessing Script
UCI 钓鱼网站数据集预处理脚本

This script performs the following preprocessing steps:
本脚本执行以下预处理步骤：
1. Load CSV data (加载CSV数据)
2. Standardize features (标准化特征)
3. Apply PCA (95% variance) (应用PCA，保留95%方差)
4. Split data (70/30) (70/30分割数据)
5. Save preprocessed data (保存预处理后的数据)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> tuple:
    """
    Load data from CSV file.
    从CSV文件加载数据。
    
    Args:
        file_path: Path to CSV file (CSV文件路径)
        
    Returns:
        Tuple of (features, labels) (特征和标签的元组)
    """
    logger.info("Loading data from: %s", file_path)
    
    # 读取数据
    data = pd.read_csv(file_path)
    
    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data['Result']  # 直接使用列名获取标签
    
    logger.info("Data loaded - Features: %s, Labels: %s", X.shape, y.shape)
    
    # 记录标签分布
    label_dist = pd.DataFrame(y.value_counts(normalize=True))
    label_dist.columns = ['proportion']
    logger.info("Label distribution:\n%s", label_dist)
    
    return X, y

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Preprocess data: standardize features and apply PCA.
    预处理数据：标准化特征并应用PCA。
    
    Args:
        X: Feature matrix (特征矩阵)
        y: Labels (标签)
        
    Returns:
        Tuple of (preprocessed_X, scaler, pca, feature_names)
        返回(预处理后的特征, 标准化器, PCA模型, 特征名称)的元组
    """
    logger.info("Starting data preprocessing...")
    
    # 标准化特征
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 应用PCA
    logger.info("Applying PCA...")
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_pca = pca.fit_transform(X_scaled)
    
    # 记录PCA信息
    logger.info("Number of components selected: %d", pca.n_components_)
    logger.info("Explained variance ratio: %s", pca.explained_variance_ratio_)
    logger.info("Cumulative explained variance ratio: %s", np.cumsum(pca.explained_variance_ratio_))
    
    return X_pca, scaler, pca, X.columns.tolist()

def split_data(X: np.ndarray, y: pd.Series) -> tuple:
    """
    Split data into train and test sets.
    将数据分割为训练集和测试集。
    
    Args:
        X: Feature matrix (特征矩阵)
        y: Labels (标签)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        返回(训练特征, 测试特征, 训练标签, 测试标签)的元组
    """
    logger.info("Splitting data (70/30)...")
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def save_preprocessed_data(data: tuple, preprocessors: dict) -> None:
    """
    Save preprocessed data and preprocessors.
    保存预处理后的数据和预处理器。
    
    Args:
        data: Tuple of (X_train, X_test, y_train, y_test)
              (训练特征, 测试特征, 训练标签, 测试标签)的元组
        preprocessors: Dictionary of preprocessor objects
                     预处理器对象的字典
    """
    X_train, X_test, y_train, y_test = data
    
    # 创建带有PCA特征名称的DataFrame
    pca_columns = [f'PC{i+1}' for i in range(X_train.shape[1])]
    
    train_df = pd.DataFrame(X_train, columns=pca_columns)
    train_df['label'] = y_train.values  # 转换为numpy数组
    
    test_df = pd.DataFrame(X_test, columns=pca_columns)
    test_df['label'] = y_test.values  # 转换为numpy数组
    
    # 如果目录不存在则创建
    Path('Datasets/cleaned/train').mkdir(parents=True, exist_ok=True)
    Path('Datasets/cleaned/test').mkdir(parents=True, exist_ok=True)
    Path('Models/preprocessors/phishing').mkdir(parents=True, exist_ok=True)
    
    # 保存预处理后的数据
    train_path = "Datasets/cleaned/train/phishing_train_preprocessed.csv"
    test_path = "Datasets/cleaned/test/phishing_test_preprocessed.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info("Saved preprocessed data:")
    logger.info("  - Train data: %s", train_path)
    logger.info("  - Test data: %s", test_path)
    
    # 保存预处理器
    preprocessors_path = "Models/preprocessors/phishing"
    joblib.dump(preprocessors['scaler'], f"{preprocessors_path}/scaler_phishing.pkl")
    joblib.dump(preprocessors['pca'], f"{preprocessors_path}/pca_phishing.pkl")
    
    # 保存PCA信息
    pca_info_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(preprocessors['pca'].explained_variance_ratio_))],
        'Explained_Variance_Ratio': preprocessors['pca'].explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(preprocessors['pca'].explained_variance_ratio_)
    })
    
    # 添加原始特征名称作为单独的DataFrame
    feature_info_df = pd.DataFrame({
        'Original_Feature': preprocessors['feature_names']
    })
    
    # 将两个DataFrame保存到Excel文件的不同sheet中
    with pd.ExcelWriter(f"{preprocessors_path}/pca_info.xlsx") as writer:
        pca_info_df.to_excel(writer, sheet_name='PCA_Components', index=False)
        feature_info_df.to_excel(writer, sheet_name='Original_Features', index=False)
    
    logger.info("Saved preprocessors:")
    logger.info("  - Scaler: %s/scaler_phishing.pkl", preprocessors_path)
    logger.info("  - PCA: %s/pca_phishing.pkl", preprocessors_path)
    logger.info("  - PCA info: %s/pca_info.xlsx", preprocessors_path)

def main():
    """
    Main function to execute the preprocessing pipeline
    执行预处理流水线的主函数
    """
    try:
        # 加载数据
        X, y = load_data("Datasets/original/cleaned_phishing.csv")
        
        # 预处理数据
        X_pca, scaler, pca, feature_names = preprocess_data(X, y)
        
        # 分割数据
        data = split_data(X_pca, y)
        
        # 保存预处理后的数据和预处理器
        preprocessors = {
            'scaler': scaler,
            'pca': pca,
            'feature_names': feature_names
        }
        save_preprocessed_data(data, preprocessors)
        
        logger.info("\n✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error("\n❌ Data preprocessing failed: %s", str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 