# 网络安全检测与用户行为预测项目的数据预处理工具函数
#
# 本模块提供了一系列用于数据清理和预处理的工具函数，支持项目中的两个主要数据集：
# 1. CIC_Dataset.csv：用于 URL 检测的数据集
# 2. merged_dataset.csv：用于网络流量检测和用户行为预测的数据集
#
# 主要功能：
# - 数据集信息展示
# - 缺失值处理（基于 IEEE 论文推荐的策略）
# - 异常值处理（使用统计方法和领域知识）
# - 特征编码（分类变量处理）
# - 特征标准化（确保模型训练稳定性）
# - 特征选择（基于随机森林的重要性评估）
#
# 参考文献：
# - IEEE 论文："Research on Malicious URL Detection Based on Random Forest"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict

def print_dataset_info(df: pd.DataFrame, dataset_name: str) -> None:
    # 打印数据集的基本信息，用于数据探索和清理过程的监控。
    #
    # 目的：
    # - 在数据清理过程中提供数据集状态的快照
    # - 帮助识别需要处理的问题（如缺失值）
    # - 验证数据类型是否符合预期
    # - 监控数据清理过程的进展
    #
    # 使用场景：
    # 1. 原始数据集加载后的初始检查
    # 2. 清理步骤后的验证
    # 3. 训练集和测试集分割后的确认
    #
    # 参数：
    #     df: 要分析的数据集
    #     dataset_name: 数据集的名称，用于在输出中标识
    #
    # 输出示例：
    # ==================================================
    # Dataset: CIC Dataset
    # Shape: (15367, 78)  # 完整数据集大小
    # Missing values: ...  # 缺失值统计
    # Data types: ...     # 特征类型分布
    # ==================================================
    
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
    # 处理数据集中的缺失值，基于 IEEE 论文推荐的策略。
    #
    # 目的：
    # - 确保数据集中没有缺失值，这是机器学习模型的基本要求
    # - 使用合适的策略填充缺失值，以保持数据的统计特性
    # - 支持不同类型特征的特定处理策略
    #
    # 实现策略：
    # 1. 数值型特征：
    #    - CIC 数据集：使用均值填充（基于 IEEE 论文建议）
    #    - merged 数据集：使用中位数填充（处理异常值影响）
    # 2. 分类型特征：
    #    - CIC 数据集：使用众数填充
    #    - merged 数据集：使用 'None' 填充（表示未知类别）
    #
    # 参数：
    #     df: 输入数据集
    #     numeric_strategy: 数值型特征的填充策略 ('mean' 或 'median')
    #     categorical_fill: 分类型特征的填充值
    #
    # 返回：
    #     处理后的数据集副本
    #
    # 注意：
    # - 对于 URL 特征（如 tld），使用众数填充
    # - 对于行为特征（如 Social_Media_Usage），使用 'None' 填充
    
    df_cleaned = df.copy()
    
    # 处理数值型特征
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = df[col].median()
            df_cleaned[col].fillna(fill_value, inplace=True)
    
    # 处理分类型特征
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
    # 处理数据集中的异常值，基于领域知识和统计方法。
    #
    # 目的：
    # - 减少异常值对模型训练的影响
    # - 确保特征值在合理的范围内
    # - 提高模型的稳定性和泛化能力
    #
    # 实现策略：
    # 1. URL 特征处理：
    #    - urlLen：限制在 99 百分位数内
    #    - NumberofDotsinURL：基于实际 URL 结构限制
    # 2. 网络流量特征处理：
    #    - Hours_Online：限制在 [0, 24]
    #    - CPU_Utilization：限制在 [0, 100]
    #    - Flow_Bytes_per_s：使用 99 百分位数作为上限
    #
    # 参数：
    #     df: 输入数据集
    #     columns: 字典，键为列名，值为元组 (最小值, 最大值)
    #             例如：{'Hours_Online': (0, 24), 'CPU_Utilization': (0, 100)}
    #
    # 返回：
    #     处理后的数据集副本
    
    df_cleaned = df.copy()
    
    for col, (min_val, max_val) in columns.items():
        if col in df.columns:
            df_cleaned[col] = df_cleaned[col].clip(min_val, max_val)
            
    return df_cleaned

def encode_categorical_features(df: pd.DataFrame, columns: List[str],
                              encoding_type: str = 'onehot') -> pd.DataFrame:
    # 对分类特征进行编码，支持项目中的不同类型特征。
    #
    # 目的：
    # - 将分类特征转换为机器学习算法可处理的数值形式
    # - 保持特征间的关系和信息
    # - 支持不同类型的分类变量编码需求
    #
    # 实现策略：
    # 1. URL 类型编码：
    #    - URL_Type_obf_Type：转换为二分类（Benign=0，其他=1）
    # 2. 用户行为特征编码：
    #    - Device_Type：独热编码
    #    - Social_Media_Usage：独热编码
    #    - Cybersecurity_Behavior_Category：标签编码（Safe=0，Neutral=1，Risky=2）
    #
    # 参数：
    #     df: 输入数据集
    #     columns: 需要编码的分类特征列表
    #     encoding_type: 编码方式 ('onehot' 或 'label')
    #
    # 返回：
    #     编码后的数据集副本
    
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
    # 对数值特征进行标准化，确保特征尺度一致。
    #
    # 目的：
    # - 使不同尺度的特征具有可比性
    # - 提高模型训练的稳定性和收敛速度
    # - 处理异常值和无穷大值
    #
    # 实现策略：
    # 1. URL 特征标准化：
    #    - urlLen, Querylength 等标准化到 [0,1]
    # 2. 网络流量特征标准化：
    #    - Flow_Bytes_per_s, Packet_Size 等标准化到 [0,1]
    # 3. 特殊处理：
    #    - Hours_Online：保持原始值（0-24）
    #    - CPU_Utilization：保持原始值（0-100）
    #
    # 参数：
    #     df: 输入数据集
    #     columns: 需要标准化的特征列表
    #
    # 返回：
    #     (标准化后的数据集副本, 训练好的 scaler 对象)
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    
    # 过滤出存在的列
    columns = [col for col in columns if col in df.columns]
    
    if columns:
        # 处理无穷大值
        df_scaled[columns] = df_scaled[columns].replace([np.inf, -np.inf], np.nan)
        
        # 用中位数填充 NaN
        for col in columns:
            df_scaled[col] = df_scaled[col].fillna(df_scaled[col].median())
        
        # 标准化特征
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled, scaler

def select_features(df: pd.DataFrame, target_col: str, n_features: int = 10,
                   exclude_cols: List[str] = None) -> Tuple[List[str], pd.Series]:
    # 使用随机森林进行特征选择，基于 IEEE 论文的推荐参数。
    #
    # 目的：
    # - 降低数据维度，提高模型效率
    # - 选择对目标变量最有预测力的特征
    # - 减少噪声和无关特征的影响
    #
    # 实现策略：
    # 1. URL 检测模型：
    #    - 选择前 10 个最重要的 URL 特征
    #    - 使用随机森林参数：n_estimators=60, max_depth=50（基于 IEEE 论文）
    # 2. 网络流量检测模型：
    #    - 选择前 15 个最重要的特征
    #    - 保持相同的随机森林参数以确保一致性
    #
    # 参数：
    #     df: 输入数据集
    #     target_col: 目标变量的列名（如 'URL_Type_obf_Type' 或 'Insecurity_Level'）
    #     n_features: 要选择的特征数量（URL 检测为 10，网络流量检测为 15）
    #     exclude_cols: 要排除的特征列表
    #
    # 返回：
    #     (选中的特征列表, 所有特征的重要性得分)
    #
    # 注意：
    # - 特征选择基于 IEEE 论文"Research on Malicious URL Detection Based on Random Forest"
    # - 模型参数经过验证，在测试集上达到 >98.6% 的准确率
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 准备特征矩阵
    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    
    # 初始化并训练随机森林
    rf = RandomForestClassifier(n_estimators=60, max_depth=50, random_state=42)
    rf.fit(X, y)
    
    # 获取特征重要性
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    
    # 选择最重要的特征
    selected_features = importance.head(n_features).index.tolist()
    
    return selected_features, importance 