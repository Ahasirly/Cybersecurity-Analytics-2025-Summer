# CIC URL Detection Dataset Preprocessing Script
#
# 主要功能：
# 1. 数据清理和预处理：
#    - 处理缺失值：数值型用均值，分类型用众数
#    - 处理异常值：基于IEEE论文中的阈值建议
#    - 特征标准化：将数值特征缩放到[0,1]区间
#
# 2. 特征工程和选择：
#    - 使用随机森林进行特征重要性评估
#    - 选择top 10最重要特征用于模型训练
#    - 保存完整特征集用于后续实验
#
# 3. 数据集划分：
#    - 训练集(80%) vs 测试集(20%)
#    - 5折交叉验证用于模型评估
#    - 分别保存完整版和精简版数据集
#
# 输入输出：
# - 输入：
#   * CIC_Dataset.csv：原始URL特征数据集
#   * 包含80个特征和1个目标变量(URL_Type_obf_Type)
#
# - 输出：
#   * 精简版(10特征)：
#     - Datasets/cleaned/train/CIC_Dataset_train.csv
#     - Datasets/cleaned/test/CIC_Dataset_test.csv
#   * 完整版(80特征)：
#     - Datasets/cleaned/train/CIC_Dataset_train_full.csv
#     - Datasets/cleaned/test/CIC_Dataset_test_full.csv
#   * 交叉验证集：
#     - Datasets/cleaned/cv/train/CIC_Dataset_train_fold{1-5}.csv
#     - Datasets/cleaned/cv/test/CIC_Dataset_test_fold{1-5}.csv
#   * 预处理器：
#     - Models/preprocessors/scaler_cic.pkl
#
# 参考文献：
# - IEEE论文："Research on Malicious URL Detection Based on Random Forest"
# - 特征选择和阈值设置基于该论文的实验结果

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from data_cleaning_utils import (
    print_dataset_info,
    handle_missing_values,
    handle_outliers,
    standardize_features,
    select_features
)
from sklearn.ensemble import RandomForestClassifier

# 预处理器保存路径
# 用于保存StandardScaler对象，确保测试和生产环境使用相同的缩放参数
PREPROCESSOR_PATH = "Models/preprocessors/"

# URL特征的有效值范围定义
# 基于IEEE论文和实际URL结构分析设置的阈值
# 对于没有明确上限的特征，使用99分位数作为上限
URL_FEATURE_RANGES = {
    'urlLen': (0, None),  # URL总长度，最小0，上限使用99分位数
    'NumberofDotsinURL': (2, 10),  # 域名中点的数量，合法域名至少有2个点(如example.com)
    'Querylength': (0, None),  # 查询字符串长度，可以为0，上限使用99分位数
    'domain_token_count': (1, None),  # 域名标记数，至少1个(如example.com = 2)
    'path_token_count': (0, None),  # 路径标记数，可以为0，上限使用99分位数
    'avgdomaintokenlen': (1, None),  # 平均域名标记长度，至少1个字符
    'longdomaintokenlen': (1, None),  # 最长域名标记长度，至少1个字符
    'avgpathtokenlen': (1, None),  # 平均路径标记长度，至少1个字符
    'charcompvowels': (0, None),  # 元音字符数量，用于检测随机生成的域名
    'charcompace': (0, None)  # 特殊字符数量，用于检测混淆URL
}

def clean_cic_dataset(input_path: str, output_path: tuple) -> None:
    # CIC数据集清理和预处理的主函数
    #
    # 详细处理流程：
    # 1. 数据加载和初始检查：
    #    - 读取原始CSV文件
    #    - 显示基本统计信息和数据类型
    #    - 检查缺失值情况
    #
    # 2. 缺失值处理策略：
    #    - 数值型特征：使用均值填充（保持数据分布）
    #    - 分类型特征：使用众数填充（最常见值）
    #    - 基于IEEE论文推荐的处理方法
    #
    # 3. 异常值处理方法：
    #    - 使用预定义的URL_FEATURE_RANGES进行限制
    #    - 对无上限特征使用99分位数作为上限
    #    - 保留处理前后的统计信息用于比较
    #
    # 4. 特征工程和选择：
    #    - 标准化所有数值特征到[0,1]区间
    #    - 使用随机森林评估特征重要性
    #    - 选择最重要的10个特征用于建模
    #
    # 5. 数据集划分和保存：
    #    - 训练集(80%) vs 测试集(20%)
    #    - 创建5折交叉验证集
    #    - 分别保存完整版和精简版数据集
    #
    # 参数说明：
    #     input_path: str
    #         原始数据集的路径，例如：'Datasets/original/CIC_Dataset.csv'
    #     
    #     output_path: tuple
    #         保存处理后数据集的路径元组，格式：
    #         (训练集路径, 测试集路径)
    #         例如：('Datasets/cleaned/train/CIC_Dataset_train.csv',
    #               'Datasets/cleaned/test/CIC_Dataset_test.csv')
    #
    # 注意事项：
    # 1. 确保输入文件存在且格式正确
    # 2. 输出目录会自动创建（如果不存在）
    # 3. 预处理器会被保存用于后续预测
    # 4. 交叉验证集会自动保存在cv子目录中
    
    print("\nStarting CIC Dataset Processing...")
    
    # 加载数据集并显示基本信息
    df = pd.read_csv(input_path)
    print_dataset_info(df, "CIC Dataset (Original)")
    
    # 处理缺失值
    df = handle_missing_values(df, numeric_strategy='mean', categorical_fill='mode')
    
    # 设置特征的异常值范围
    outlier_ranges = {}
    for feature, (min_val, max_val) in URL_FEATURE_RANGES.items():
        if feature in df.columns:
            if max_val is None:
                max_val = df[feature].quantile(0.99)  # 使用99分位数作为上限
            outlier_ranges[feature] = (min_val, max_val)
    
    # 打印异常值处理前的统计信息
    for feature in outlier_ranges.keys():
        print(f"\nStatistics before outlier handling - {feature}:")
        print(df[feature].describe())
    
    # 处理异常值
    df = handle_outliers(df, outlier_ranges)
    
    # 打印异常值处理后的统计信息
    for feature in outlier_ranges.keys():
        print(f"\nStatistics after outlier handling - {feature}:")
        print(df[feature].describe())
    
    # 转换目标变量为二分类（0=良性，1=恶意）
    print("\nTarget variable distribution (before):")
    print(df['URL_Type_obf_Type'].value_counts())
    df['URL_Type_obf_Type'] = (df['URL_Type_obf_Type'].str.lower() != 'benign').astype(int)
    print("\nTarget variable distribution (after):")
    print(df['URL_Type_obf_Type'].value_counts())
    
    # 标准化数值特征
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df, scaler = standardize_features(df, numeric_cols)
    
    # 保存标准化器（用于后续预测）
    os.makedirs(PREPROCESSOR_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(PREPROCESSOR_PATH, 'scaler_cic.pkl'))
    
    # 使用随机森林选择最重要的特征
    selected_features, importance = select_features(
        df, 'URL_Type_obf_Type',
        n_features=10
    )
    print("\nTop 10 most important features and their importance:")
    for feat, imp in zip(selected_features, importance):
        print(f"{feat}: {imp:.4f}")
    
    # 保存完整特征数据集（用于特征工程实验）
    X = df.drop('URL_Type_obf_Type', axis=1)
    y = df['URL_Type_obf_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_df_full = pd.concat([X_train, y_train], axis=1)
    test_df_full = pd.concat([X_test, y_test], axis=1)
    
    train_df_full.to_csv(output_path[0].replace('.csv', '_full.csv'), index=False)
    test_df_full.to_csv(output_path[1].replace('.csv', '_full.csv'), index=False)
    
    # 创建交叉验证数据集
    # 五折交叉验证的具体实现：
    # 1. 数据集平均分成5份，每份大约20%的数据
    # 2. 交叉验证过程：
    #    - 第1折：第1份作为测试集，第2-5份作为训练集
    #    - 第2折：第2份作为测试集，第1,3-5份作为训练集
    #    - 第3折：第3份作为测试集，第1-2,4-5份作为训练集
    #    - 第4折：第4份作为测试集，第1-3,5份作为训练集
    #    - 第5折：第5份作为测试集，第1-4份作为训练集
    #
    # 优点：
    # 1. 充分利用数据：每个样本都会被用作训练和测试
    # 2. 减少过拟合（overfitting)：通过多次训练和测试获得更可靠的模型评估
    # 3. 评估稳定性：可以得到5个性能指标，更好地评估模型的稳定性
    #
    # 实现细节：
    # - 使用KFold类实现数据集划分
    # - shuffle=True确保数据被随机打乱
    # - random_state=42保证结果可复现
    print("\nCreating 5-fold cross-validation datasets...")
    os.makedirs('Datasets/cleaned/cv/train', exist_ok=True)
    os.makedirs('Datasets/cleaned/cv/test', exist_ok=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 执行五折交叉验证数据集划分
    # 对于每一折：
    # 1. 使用iloc选择对应的训练集和测试集数据
    # 2. 分别保存到对应的文件夹中
    # 3. 打印每一折的数据集大小信息
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        fold_train = df.iloc[train_idx]  # 获取训练集数据
        fold_test = df.iloc[test_idx]    # 获取测试集数据
        
        # 构建保存路径
        fold_train_path = 'Datasets/cleaned/cv/train/CIC_Dataset_train_fold{}.csv'.format(fold)
        fold_test_path = 'Datasets/cleaned/cv/test/CIC_Dataset_test_fold{}.csv'.format(fold)
        
        # 保存当前折的数据集
        fold_train.to_csv(fold_train_path, index=False)
        fold_test.to_csv(fold_test_path, index=False)
        
        # 打印当前折的数据集信息
        print(f"\nFold {fold} datasets saved:")
        print(f"Training set size: {fold_train.shape}")
        print(f"Test set size: {fold_test.shape}")
    
    # 保存精简版数据集（只包含选定的特征）
    df = df[selected_features + ['URL_Type_obf_Type']]
    X = df.drop('URL_Type_obf_Type', axis=1)
    y = df['URL_Type_obf_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(output_path[0], index=False)
    test_df.to_csv(output_path[1], index=False)
    
    print("\nCIC Dataset cleaning completed!")
    print_dataset_info(train_df, "CIC Dataset (Cleaned - Training Set)")
    print_dataset_info(test_df, "CIC Dataset (Cleaned - Test Set)")

def main():
    # 主函数：执行CIC数据集的完整清理流程
    #
    # 步骤：
    # 1. 设置输入输出路径
    # 2. 创建必要的目录结构
    # 3. 执行数据清理流程
    # 4. 保存处理后的数据集和预处理器
    #
    # 输入输出文件：
    # - 输入：Datasets/original/CIC_Dataset.csv
    # - 输出：
    #   * 训练集：Datasets/cleaned/train/CIC_Dataset_train.csv
    #   * 测试集：Datasets/cleaned/test/CIC_Dataset_test.csv
    #   * 预处理器：Models/preprocessors/scaler_cic.pkl
    
    # 定义输入输出路径
    cic_input = 'Datasets/original/CIC_Dataset.csv'
    cic_train_output = 'Datasets/cleaned/train/CIC_Dataset_train.csv'
    cic_test_output = 'Datasets/cleaned/test/CIC_Dataset_test.csv'
    
    # 创建输出目录
    os.makedirs('Datasets/cleaned/train', exist_ok=True)
    os.makedirs('Datasets/cleaned/test', exist_ok=True)
    
    print("Starting CIC Dataset cleaning process...")
    print("Output directories created")
    
    # 执行数据清理
    clean_cic_dataset(cic_input, (cic_train_output, cic_test_output))
    
    print("\nData cleaning completed!")
    print("Preprocessed datasets saved in Datasets/cleaned/")
    print("Preprocessor objects saved in Models/preprocessors/")

if __name__ == '__main__':
    main() 