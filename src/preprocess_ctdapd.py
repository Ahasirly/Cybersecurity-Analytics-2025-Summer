# CTDAPD (Cyber Threat Detection and Attack Pattern Dataset) 数据集预处理脚本
# 
# 数据集说明：
# - 来源：网络流量数据
# - 目标：检测网络攻击
# - 特征：包含网络流量特征（如包大小、协议类型、流持续时间等）
# - 标签：二分类（正常流量 vs 攻击流量）
#
# 预处理步骤概述：
# 1. 数据清理：移除无关特征，处理缺失值
# 2. 特征工程：编码分类变量，标准化数值特征
# 3. 异常值处理：使用IQR方法检测和移除异常值
# 4. 数据分割：划分训练集和测试集
# 5. 保存处理后的数据：分别保存清理后的完整数据集、训练集和测试集

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_ctdapd_dataset():
    # CTDAPD数据集预处理主函数
    #
    # 功能：
    # - 加载原始CTDAPD数据集
    # - 执行完整的数据预处理流程
    # - 保存处理后的数据集
    # - 生成数据质量报告
    #
    # 返回值：
    # - X_train: 训练集特征
    # - X_test: 测试集特征
    # - y_train: 训练集标签
    # - y_test: 测试集标签
    # - feature_names: 特征名列表
    
    print("开始预处理CTDAPD数据集...")
    
    # 1. 加载原始数据
    # 从CSV文件加载原始CTDAPD数据集
    df = pd.read_csv('Datasets/original/CTDAPD_Dataset.csv')
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 检查目标变量分布，了解数据集是否平衡
    print(f"目标变量分布: {df['Label'].value_counts()}")
    
    # 2. 特征选择
    # 移除对模型训练无用的列：
    # - Date：时间戳对检测攻击模式无直接帮助
    # - Source_IP/Destination_IP：具体IP地址可能导致模型过拟合 overfitting 
    print("\n1. 移除不需要的列...")
    cols_to_drop = ['Date', 'Source_IP', 'Destination_IP']
    df_clean = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    print(f"移除列后形状: {df_clean.shape}")
    
    # 3. 缺失值处理
    # 统计每个特征的缺失值数量，为后续处理提供依据
    print("\n2. 处理缺失值...")
    print("缺失值统计:")
    missing_counts = df_clean.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # 分离特征和目标变量，方便分别处理
    X = df_clean.drop(['Label'], axis=1)
    y = df_clean['Label']
    
    # 4. 分类特征处理
    # 对分类特征进行编码，将文本转换为数值以供模型使用
    print("\n3. 处理分类特征...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"分类特征: {list(categorical_cols)}")
    
    X_processed = X.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        print(f"处理 {col}...")
        # 使用'Unknown'填充缺失值，保持数据完整性
        X_processed[col] = X_processed[col].fillna('Unknown')
        
        # 使用LabelEncoder将分类值转换为数值
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        label_encoders[col] = le
        
        print(f"  - {col}: {len(le.classes_)} 个类别")
        print(f"  - 类别: {le.classes_}")
    
    # 5. 数值特征处理
    # 处理数值特征中的异常值和缺失值
    print("\n4. 处理数值特征...")
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    print(f"数值特征数量: {len(numeric_cols)}")
    
    # 处理无穷值：将无穷值替换为NaN，后续用中位数 median 填充
    print("处理无穷值...")
    inf_counts = np.isinf(X_processed[numeric_cols]).sum()
    if inf_counts.sum() > 0:
        print("发现无穷值:")
        print(inf_counts[inf_counts > 0])
        X_processed[numeric_cols] = X_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 使用中位数填充数值特征的缺失值
    # 选择中位数而不是平均数，因为中位数对异常值不敏感
    print("处理数值特征缺失值...")
    X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
    
    # 6. 异常值检测与处理
    # 使用IQR（四分位距）方法检测异常值
    # IQR = Q3 - Q1，异常值定义为小于(Q1 - 1.5*IQR)或大于(Q3 + 1.5*IQR)的值
    print("\n5. 处理异常值...")
    Q1 = X_processed[numeric_cols].quantile(0.25)
    Q3 = X_processed[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义异常值边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 标记并移除异常值
    outlier_mask = ((X_processed[numeric_cols] < lower_bound) | (X_processed[numeric_cols] > upper_bound)).any(axis=1)
    print(f"检测到 {outlier_mask.sum()} 个异常样本 ({outlier_mask.sum()/len(X_processed)*100:.2f}%)")
    
    X_clean = X_processed[~outlier_mask]
    y_clean = y[~outlier_mask]
    print(f"清理后数据形状: {X_clean.shape}")
    
    # 7. 目标变量编码
    # 将目标变量（Label）编码为数值：0表示正常流量，1表示攻击
    print("\n6. 编码目标变量...")
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y_clean)
    print(f"目标变量编码: {dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_)))}")
    print(f"编码后分布: {pd.Series(y_encoded).value_counts().sort_index()}")
    
    # 8. 数据集分割
    # 使用分层抽样（stratify）确保训练集和测试集中类别分布一致
    print("\n7. 分割数据...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded, 
        test_size=0.2,          # 80%训练，20%测试
        random_state=42,        # 固定随机种子以确保可重复性
        stratify=y_encoded      # 保持分割后的类别分布
    )
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"训练集类别分布: {pd.Series(y_train).value_counts().sort_index()}")
    print(f"测试集类别分布: {pd.Series(y_test).value_counts().sort_index()}")
    
    # 9. 保存预处理后的数据
    # 将处理后的数据保存为CSV格式，便于后续使用
    print("\n8. 保存预处理后的数据...")
    
    # 创建保存目录
    os.makedirs('Datasets/cleaned/train', exist_ok=True)
    os.makedirs('Datasets/cleaned/test', exist_ok=True)
    
    # 保存训练数据
    train_df = pd.DataFrame(X_train, columns=X_clean.columns)
    train_df['Label'] = y_train
    train_df.to_csv('Datasets/cleaned/train/CTDAPD_train.csv', index=False)
    
    # 保存测试数据
    test_df = pd.DataFrame(X_test, columns=X_clean.columns)
    test_df['Label'] = y_test
    test_df.to_csv('Datasets/cleaned/test/CTDAPD_test.csv', index=False)
    
    # 保存完整的清理后数据集
    full_clean_df = pd.DataFrame(X_clean, columns=X_clean.columns)
    full_clean_df['Label'] = y_encoded
    full_clean_df.to_csv('Datasets/cleaned/CTDAPD_cleaned.csv', index=False)
    
    print("预处理完成！文件已保存:")
    print("- Datasets/cleaned/train/CTDAPD_train.csv")
    print("- Datasets/cleaned/test/CTDAPD_test.csv") 
    print("- Datasets/cleaned/CTDAPD_cleaned.csv")
    
    # 10. 生成数据质量报告
    # 总结预处理结果，展示关键统计信息
    print("\n" + "="*60)
    print("📊 数据质量报告")
    print("="*60)
    print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"清理后数据: {X_clean.shape[0]} 行, {X_clean.shape[1]} 列")
    print(f"数据保留率: {X_clean.shape[0]/df.shape[0]*100:.2f}%")
    print(f"特征数量: {X_clean.shape[1]}")
    print(f"数值特征: {len(X_clean.select_dtypes(include=[np.number]).columns)}")
    print(f"分类特征: {len(categorical_cols)}")
    print(f"目标变量类别: {len(label_encoder_y.classes_)}")
    print(f"类别平衡: Normal={pd.Series(y_encoded).value_counts()[0]}, Attack={pd.Series(y_encoded).value_counts()[1]}")
    
    return X_train, X_test, y_train, y_test, X_clean.columns

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, feature_names = preprocess_ctdapd_dataset()
        print("\n✅ CTDAPD数据集预处理成功完成！")
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc() 