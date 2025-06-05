# URL恶意检测模型比较与评估脚本
#
# 主要功能：
# 1. 模型训练与评估：
#    - 使用预处理后的精简数据集（10个最重要特征）
#    - 训练多个不同类型的机器学习模型
#    - 评估每个模型的性能指标
#    - 自动选择最佳表现的模型
#
# 2. 模型类型：
#    - 随机森林 (Random Forest)：
#      * 基于IEEE论文推荐参数
#      * n_estimators=60 (60棵决策树)
#      * max_depth=50 (树最大深度)
#      * 训练时间较长
#
#    - 支持向量机 (SVM)： 
#      * 使用RBF核函数
#      * 适合高维特征分类
#      * 对特征缩放敏感
#      * 训练时间较长
    
#    
#    - 神经网络 (Neural Network)：
#      * 两层隐藏层(100,50)
#      * 最大迭代次数300
#      * 适合复杂非线性关系
#      * 训练时间较长
#    
#    - 逻辑回归 (Logistic Regression)：
#      * 最大迭代次数1000
#      * 作为基准模型
#      * 适合线性可分问题
#      * 训练时间较短

# 3. 评估指标：
#    - 准确率 (Accuracy)：整体预测准确程度
#    - 精确率 (Precision)：预测为恶意URL中正确的比例
#    - 召回率 (Recall)：实际恶意URL被正确预测的比例
#    - F1分数：精确率和召回率的调和平均
#    - 训练时间：模型训练和预测的总耗时
#
# 输入输出：
# - 输入：
#   * 训练集：Datasets/cleaned/train/CIC_Dataset_train.csv
#   * 测试集：Datasets/cleaned/test/CIC_Dataset_test.csv
#
# - 输出：
#   * 每个模型的详细评估指标
#   * 最佳模型的性能报告
#   * 训练和预测时间统计

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def load_data():
    # 数据加载函数
    #
    # 功能描述：
    # 1. 加载预处理后的训练集和测试集
    # 2. 自动分离特征和目标变量
    #
    # 数据格式：
    # - 特征：10个最重要的URL特征（已标准化）
    # - 目标变量：URL_Type_obf_Type（0=良性，1=恶意）
    #
    # 返回值：
    # - X_train: 训练集特征矩阵 (10个最重要的URL特征)
    # - X_test: 测试集特征矩阵 (10个最重要的URL特征)
    # - y_train: 训练集标签向量 (Url的真实标签)
    # - y_test: 测试集标签向量 (Url的真实标签)
    #
    # 注意事项：
    # 1. 确保数据集文件存在
    # 2. 特征已经过预处理和标准化
    # 3. 数据集已经过清洗，不需要额外处理
    
    # 加载训练集和测试集
    train_df = pd.read_csv('Datasets/cleaned/train/CIC_Dataset_train.csv')
    test_df = pd.read_csv('Datasets/cleaned/test/CIC_Dataset_test.csv')
    
    # 分离特征和目标变量
    X_train = train_df.drop('URL_Type_obf_Type', axis=1)
    y_train = train_df['URL_Type_obf_Type']
    X_test = test_df.drop('URL_Type_obf_Type', axis=1)
    y_test = test_df['URL_Type_obf_Type']
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # 模型评估函数
    #
    # 功能描述：
    # 1. 模型训练：
    #    - 使用训练集训练模型
    #    - 记录训练开始时间
    #    - 计算训练耗时
    #
    # 2. 性能评估：
    #    - 在测试集上进行预测
    #    - 计算多个评估指标
    #    - 生成评估报告
    #
    # 评估指标说明：
    # - accuracy准确率：(TP + TN) / (TP + TN + FP + FN)
    # - precision精确率：TP / (TP + FP)
    # - recall召回率：TP / (TP + FN)
    # - f1分数：2 * (precision * recall) / (precision + recall)
    # 其中：
    # - TP：正确识别的恶意URL True Positive
    # - TN：正确识别的良性URL True Negative
    # - FP：误判为恶意的良性URL False Positive
    # - FN：未识别出的恶意URL False Negative
    #
    # 参数说明：
    # - model: 待评估的模型对象
    # - X_train, X_test: 特征矩阵
    # - y_train, y_test: 标签向量
    # - model_name: 模型名称，用于结果展示
    #
    # 返回值：
    # 包含所有评估指标的字典：
    # {
    #   'model_name': 模型名称,
    #   'accuracy': 准确率,
    #   'precision': 精确率,
    #   'recall': 召回率,
    #   'f1': F1分数,
    #   'train_time': 训练时间
    # }
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算训练时间
    train_time = time.time() - start_time
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 打印结果
    print(f"\n{model_name} Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training and prediction time: {train_time:.2f} seconds")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time
    }

def main():
    # 主函数：执行完整的模型比较流程
    #
    # 执行步骤：
    # 1. 数据准备：
    #    - 加载预处理后的数据集
    #    - 确认数据集已正确加载
    #
    # 2. 模型初始化：
    #    - 随机森林：使用IEEE论文推荐参数
    #    - SVM：使用RBF核函数处理非线性关系
    #    - 神经网络：两层隐藏层设计
    #    - 逻辑回归：作为基准线性模型
    #
    # 3. 模型评估：
    #    - 依次评估每个模型
    #    - 收集所有评估结果
    #    - 对比不同模型性能
    #
    # 4. 结果展示：
    #    - 找出性能最佳的模型
    #    - 展示详细的评估指标
    #    - 输出训练时间对比
    #
    # 注意事项：
    # 1. 确保内存足够运行所有模型
    # 2. SVM和神经网络可能训练时间较长
    # 3. 结果会自动保存最佳模型信息
    
    # 加载数据
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # 定义要比较的模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=60, max_depth=50),  # 论文参数
        'SVM': SVC(kernel='rbf'),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    # 存储所有模型的结果
    results = []
    
    # 评估每个模型
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        results.append(result)
    
    # 找出最佳模型并展示结果
    best_model = max(results, key=lambda x: x['accuracy'])
    print("\n=== Best Model Performance ===")
    print(f"Model: {best_model['model_name']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1 Score: {best_model['f1']:.4f}")
    print(f"Training time: {best_model['train_time']:.2f} seconds")

if __name__ == '__main__':
    main() 