"""
UCI Phishing Websites Dataset Model Training Script
UCI 钓鱼网站数据集模型训练脚本

This script performs the following tasks:
本脚本执行以下任务：
1. Load preprocessed data (加载预处理数据)
2. Initialize 6 classifiers (初始化6个分类器)
3. Train on both raw and PCA data (在原始数据和PCA数据上训练)
4. Evaluate with multiple metrics (使用多个指标评估)
5. Perform 10-fold cross validation (执行10折交叉验证)
6. Save detailed report (保存详细报告)
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_validate, KFold
import logging
from pathlib import Path
import joblib
from datetime import datetime
import time

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """Create necessary directories if they don't exist"""
    Path('Models/results/phishing').mkdir(parents=True, exist_ok=True)
    Path('Models/trained/phishing').mkdir(parents=True, exist_ok=True)

def load_data() -> tuple:
    """
    Load both raw and PCA preprocessed data. Principal Component Analysis, meaning that principal components are used to reduce the dimensionality of the data.
    加载原始数据和PCA预处理后的数据。
    
    Returns:
        Tuple of (raw_data, pca_data)
        返回(原始数据, PCA数据)的元组
    """
    logger.info("Loading preprocessed data...")
    
    # 加载PCA数据
    train_pca = pd.read_csv('Datasets/cleaned/train/phishing_train_preprocessed.csv')
    test_pca = pd.read_csv('Datasets/cleaned/test/phishing_test_preprocessed.csv')
    
    # 加载原始数据
    raw_data = pd.read_csv('Datasets/original/cleaned_phishing.csv')
    
    # 分离特征和标签
    X_raw = raw_data.iloc[:, :-1]
    y_raw = raw_data.iloc[:, -1]
    
    # 加载预处理器
    scaler = joblib.load('Models/preprocessors/phishing/scaler_phishing.pkl')
    
    # 标准化原始特征
    X_raw_scaled = scaler.transform(X_raw)
    
    # 创建与PCA数据相匹配的训练/测试集
    train_size = len(train_pca)
    X_train_raw = X_raw_scaled[:train_size]
    X_test_raw = X_raw_scaled[train_size:]
    y_train_raw = y_raw[:train_size]
    y_test_raw = y_raw[train_size:]
    
    # 分离PCA数据的特征和标签
    X_train_pca = train_pca.iloc[:, :-1].values  # 转换为numpy数组
    y_train_pca = train_pca['label'].values      # 转换为numpy数组
    X_test_pca = test_pca.iloc[:, :-1].values    # 转换为numpy数组
    y_test_pca = test_pca['label'].values        # 转换为numpy数组
    
    # 验证数据形状
    logger.info("Data shapes:")
    logger.info("Raw data - X_train: %s, X_test: %s", X_train_raw.shape, X_test_raw.shape)
    logger.info("PCA data - X_train: %s, X_test: %s", X_train_pca.shape, X_test_pca.shape)
    
    # 验证是否存在NaN值
    if (np.isnan(X_train_pca).any() or np.isnan(X_test_pca).any() or 
        np.isnan(y_train_pca).any() or np.isnan(y_test_pca).any()):
        raise ValueError("NaN values found in PCA data")
    
    if (np.isnan(X_train_raw).any() or np.isnan(X_test_raw).any() or 
        np.isnan(y_train_raw).any() or np.isnan(y_test_raw).any()):
        raise ValueError("NaN values found in raw data")
    
    raw_data = (X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    pca_data = (X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    
    logger.info("Data loaded successfully")
    
    return raw_data, pca_data

def initialize_models(random_state: int = 42) -> dict:
    """
    Initialize all classifiers with specified parameters.
    使用指定参数初始化所有分类器。
    """
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'SVM': SVC(kernel='linear', C=1, random_state=random_state),
        'Logistic Regression': LogisticRegression(penalty='l2', max_iter=100, random_state=random_state),
        'AdaBoost': AdaBoostClassifier(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    return models

def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a model using multiple metrics.
    使用多个指标评估模型。
    """
    # 计时预测过程
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # 计算评估指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Prediction Time': pred_time,
        'Classification Report': classification_report(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                        cv: int = 10, random_state: int = 42) -> dict:
    """
    Perform k-fold cross validation.
    执行k折交叉验证。
    """
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    cv_results = cross_validate(
        model, X, y,
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        return_train_score=False
    )
    
    results = {}
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        results[f'{metric}_mean'] = scores.mean()
        results[f'{metric}_std'] = scores.std()
    
    return results

def save_detailed_report(results: dict, dataset_info: dict, output_dir: str) -> None:
    """
    Save a detailed evaluation report.
    保存详细的评估报告。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/detailed_report_phishing_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        # 写入标题
        f.write("UCI Phishing Websites Dataset - Model Comparison Report\n")
        f.write("--------------------------------------------------\n\n")
        
        # 写入数据集信息
        f.write("Dataset Information:\n")
        for key, value in dataset_info.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        # 写入每种数据类型的结果
        for data_type in ['Raw', 'PCA']:
            f.write(f"\n{data_type} Data Results:\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, metrics in results[data_type].items():
                f.write(f"{model_name}:\n")
                f.write("-" * 30 + "\n")
                
                # 测试指标
                test_metrics = metrics['test_metrics']
                f.write(f"Accuracy: {test_metrics['Accuracy']:.4f}\n")
                f.write(f"Precision: {test_metrics['Precision']:.4f}\n")
                f.write(f"Recall: {test_metrics['Recall']:.4f}\n")
                f.write(f"F1-Score: {test_metrics['F1']:.4f}\n")
                f.write(f"MCC: {test_metrics['MCC']:.4f}\n")
                f.write(f"Prediction time: {test_metrics['Prediction Time']:.2f}s\n\n")
                
                # 分类报告
                f.write("Classification Report:\n")
                f.write(test_metrics['Classification Report'])
                f.write("\n")
                
                # 混淆矩阵
                f.write("Confusion Matrix:\n")
                cm = test_metrics['Confusion Matrix']
                f.write(str(cm))
                f.write("\n\n")
                
                # 交叉验证结果
                f.write("Cross-validation Results:\n")
                cv_results = metrics['cv_results']
                for metric, value in cv_results.items():
                    if 'mean' in metric:
                        base_metric = metric.replace('_mean', '')
                        std = cv_results[f'{base_metric}_std']
                        metric_name = base_metric.replace('test_', '').replace('_weighted', '')
                        f.write(f"{metric_name}: {value:.4f} (±{std:.4f})\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
    
    logger.info("Detailed report saved to: %s", output_file)

def main():
    """
    Main function to execute the model training and evaluation pipeline
    执行模型训练和评估流水线的主函数
    """
    try:
        # 初始化
        random_state = 42
        
        # 加载数据
        raw_data, pca_data = load_data()
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = raw_data
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = pca_data
        
        # 准备数据集信息
        dataset_info = {
            'Total samples': len(y_train_raw) + len(y_test_raw),
            'Training samples': len(y_train_raw),
            'Test samples': len(y_test_raw),
            'Raw features': X_train_raw.shape[1],
            'PCA features': X_train_pca.shape[1],
            'Class distribution': f"Phishing={sum(y_train_raw == -1) + sum(y_test_raw == -1)}, "
                                f"Legitimate={sum(y_train_raw == 1) + sum(y_test_raw == 1)}"
        }
        
        # 初始化模型
        models = initialize_models(random_state)
        
        # 存储结果
        results = {
            'Raw': {},
            'PCA': {}
        }
        
        # 训练和评估模型
        for model_name, model in models.items():
            logger.info("\nTraining %s...", model_name)
            
            # 在原始数据上训练和评估
            logger.info("Training on raw data...")
            start_time = time.time()
            model_raw = model.__class__(**model.get_params())
            model_raw.fit(X_train_raw, y_train_raw)
            train_time = time.time() - start_time
            
            raw_metrics = evaluate_model(model_raw, X_train_raw, X_test_raw, 
                                      y_train_raw, y_test_raw)
            raw_metrics['Training Time'] = train_time
            
            raw_cv = cross_validate_model(model_raw, X_train_raw, y_train_raw, 
                                        random_state=random_state)
            
            results['Raw'][model_name] = {
                'test_metrics': raw_metrics,
                'cv_results': raw_cv
            }
            
            # 在PCA数据上训练和评估
            logger.info("Training on PCA data...")
            start_time = time.time()
            model_pca = model.__class__(**model.get_params())
            model_pca.fit(X_train_pca, y_train_pca)
            train_time = time.time() - start_time
            
            pca_metrics = evaluate_model(model_pca, X_train_pca, X_test_pca, 
                                      y_train_pca, y_test_pca)
            pca_metrics['Training Time'] = train_time
            
            pca_cv = cross_validate_model(model_pca, X_train_pca, y_train_pca, 
                                        random_state=random_state)
            
            results['PCA'][model_name] = {
                'test_metrics': pca_metrics,
                'cv_results': pca_cv
            }
            
            # 保存训练好的模型
            joblib.dump(model_raw, f'Models/trained/phishing/{model_name.lower().replace(" ", "_")}_raw.pkl')
            joblib.dump(model_pca, f'Models/trained/phishing/{model_name.lower().replace(" ", "_")}_pca.pkl')
        
        # 保存详细报告
        save_detailed_report(results, dataset_info, 'Models/results/phishing')
        
        logger.info("\n✅ Model training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error("\n❌ Model training failed: %s", str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 