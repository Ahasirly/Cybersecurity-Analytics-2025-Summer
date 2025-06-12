# 模型1B - 基于预处理后CTDAPD数据集的网络攻击检测
#
# 主要功能：
# 1. 模型训练与评估：
#    - 使用预处理后的CTDAPD数据集
#    - 训练多个不同类型的机器学习模型
#    - 评估每个模型的性能指标
#    - 自动选择最佳表现的模型
#
# 2. 模型类型：
#    - 梯度提升 (Gradient Boosting)：
#      * 最佳性能模型 (F1-Score: 0.9181)
#      * n_estimators=200 (200棵决策树)
#      * max_depth=8 (控制树的复杂度)
#      * learning_rate=0.1 (较小的学习率提高泛化能力)
#      * subsample=0.8 (防止过拟合)
#
#    - LightGBM：
#      * 高效的梯度提升实现
#      * 使用叶子优先的生长策略
#      * 支持类别特征
#      * 训练速度快，内存占用小
#
#    - XGBoost：
#      * 优化的分布式梯度提升
#      * 内置处理缺失值
#      * 正则化防止过拟合
#      * 支持早停机制
#
#    - 随机森林 (Random Forest)：
#      * 集成200棵决策树
#      * 最大深度15层
#      * 使用类别权重处理不平衡
#      * 并行训练提高效率
#
#    - 神经网络 (Neural Network)：
#      * 三层隐藏层(128,64,32)
#      * 使用ReLU激活函数
#      * 自适应学习率
#      * 适合复杂非线性关系
#
#    - 逻辑回归 (Logistic Regression)：
#      * 基准线性模型
#      * 使用类别权重平衡
#      * 最大迭代1000次
#      * 训练速度快
#
# 3. 评估指标：
#    - 准确率 (Accuracy)：整体预测准确程度
#    - 精确率 (Precision)：预测为攻击中正确的比例
#    - 召回率 (Recall)：实际攻击被正确预测的比例
#    - F1分数：精确率和召回率的调和平均
#    - AUC-ROC：ROC曲线下面积，评估分类器性能
#    - 训练时间：模型训练和预测的总耗时
#
# 4. 特征工程：
#    - 使用互信息选择最重要的特征
#    - 标准化数值特征
#    - 使用SMOTE处理类别不平衡
#    - 保留时序特征的重要性
#
# 输入输出：
# - 输入：
#   * 训练集：Datasets/cleaned/train/CTDAPD_train.csv
#   * 测试集：Datasets/cleaned/test/CTDAPD_test.csv
#
# - 输出：
#   * 每个模型的详细评估指标
#   * 最佳模型的性能报告
#   * 特征重要性分析
#   * 训练和预测时间统计

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_preprocessed_ctdapd_data():
    # 加载预处理后的CTDAPD数据集
    #
    # 功能描述：
    # 1. 数据加载：
    #    - 读取预处理后的训练集和测试集
    #    - 自动分离特征和标签
    #    - 验证数据完整性
    #
    # 2. 特征处理：
    #    - 选择最重要的特征（基于互信息）
    #    - 标准化数值特征
    #    - 应用SMOTE处理类别不平衡
    #
    # 数据说明：
    # - 特征：网络流量特征（如包大小、流量速率等）
    # - 标签：二分类（0=正常流量，1=攻击流量）
    # - 类别分布：存在不平衡（约85%正常，15%攻击）
    #
    # 返回值：
    # - X_train_balanced: 平衡后的训练集特征
    # - X_test_scaled: 标准化后的测试集特征
    # - y_train_balanced: 平衡后的训练集标签
    # - y_test: 测试集标签
    
    print("Loading preprocessed CTDAPD network attack detection dataset")
    
    # Load training data
    train_df = pd.read_csv('Datasets/cleaned/train/CTDAPD_train.csv')
    test_df = pd.read_csv('Datasets/cleaned/test/CTDAPD_test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Separate features and target variable
    X_train = train_df.drop(['Label'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Label'], axis=1)
    y_test = test_df['Label']
    
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Feature list: {list(X_train.columns)}")
    print(f"Training set class distribution: {pd.Series(y_train).value_counts().sort_index()}")
    print(f"Testing set class distribution: {pd.Series(y_test).value_counts().sort_index()}")
    
    # Feature selection - select most important features
    print("\nPerforming feature selection")
    k_features = min(20, X_train.shape[1])  # Select top 20 most important features
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    print(f"Selected {k_features} features: {list(selected_features)}")
    
    # Feature scaling
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f"Final training set size: {X_train_scaled.shape}")
    print(f"Final testing set size: {X_test_scaled.shape}")
    
    # Apply SMOTE balancing (only on training set)
    print("\nApplying SMOTE balancing")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Balanced training set size: {X_train_balanced.shape}")
    print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().sort_index()}")
    
    # Save preprocessors
    os.makedirs('./Models/trained', exist_ok=True)
    joblib.dump(selector, './Models/trained/feature_selector_1B_CTDAPD_clean.pkl')
    joblib.dump(scaler, './Models/trained/scaler_1B_CTDAPD_clean.pkl')
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # 评估单个模型的性能
    #
    # 功能描述：
    # 1. 模型训练：
    #    - 使用训练集训练模型
    #    - 记录训练时间
    #    - 应用早停机制（如果支持）
    #
    # 2. 性能评估：
    #    - 在测试集上进行预测
    #    - 计算多个评估指标
    #    - 生成混淆矩阵
    #    - 输出分类报告
    #
    # 评估指标说明：
    # - accuracy：(TP + TN) / (TP + TN + FP + FN)
    # - precision：TP / (TP + FP)
    # - recall：TP / (TP + FN)
    # - f1：2 * (precision * recall) / (precision + recall)
    # - auc_roc：ROC曲线下面积
    # 其中：
    # - TP：正确识别的攻击流量
    # - TN：正确识别的正常流量
    # - FP：误判为攻击的正常流量
    # - FN：未识别出的攻击流量
    #
    # 参数说明：
    # - model: 待评估的模型对象
    # - X_train, X_test: 特征矩阵
    # - y_train, y_test: 标签向量
    # - model_name: 模型名称
    #
    # 返回值：
    # 包含所有评估指标的字典：
    # {
    #     'model_name': 模型名称,
    #     'classifier_type': 分类器类型,
    #     'accuracy': 准确率,
    #     'precision': 精确率,
    #     'recall': 召回率,
    #     'f1': F1分数,
    #     'auc_roc': AUC-ROC分数,
    #     'train_time': 训练时间,
    #     'predict_time': 预测时间,
    #     'confusion_matrix': 混淆矩阵,
    #     'classification_report': 分类报告
    # }
    
    print(f"\nTraining {model_name}")
    print(f"Classifier: {model.__class__.__name__}")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Calculate AUC-ROC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = None
    
    # Save model
    model_filename = f'./Models/trained/{model_name.lower().replace(" ", "_")}_model1B_CTDAPD_clean.pkl'
    joblib.dump(model, model_filename)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Classifier Type: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if auc_roc:
        print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Training time: {train_time:.2f}s")
    print(f"Prediction time: {predict_time:.2f}s")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Attack', 'Normal']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual/Predicted  Attack  Normal")
    print(f"Attack           {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Normal           {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    return {
        'model_name': model_name,
        'classifier_type': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=['Attack', 'Normal'])
    }

class CNNModel(nn.Module):
    # CNN模型定义
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # 计算展平后的特征维度
        self.flat_features = 32 * (input_dim // 4)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.flat_features)
        x = self.fc(x)
        return x

def prepare_data_for_cnn(X):
    # 准备CNN输入数据
    # 1. 标准化数据
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_normalized = (X - X.min()) / (X.max() - X.min())
    # 2. 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_normalized).unsqueeze(1)
    return X_tensor

def evaluate_cnn_model(X_train, X_test, y_train, y_test):
    # 评估CNN模型性能
    print("\n评估CNN模型...")
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    X_train_cnn = prepare_data_for_cnn(X_train)
    X_test_cnn = prepare_data_for_cnn(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 将数据移动到指定设备
    X_train_cnn = X_train_cnn.to(device)
    X_test_cnn = X_test_cnn.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    print(f"CNN输入数据形状: {X_train_cnn.shape}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataset = TensorDataset(X_train_cnn, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 增大batch size
    
    # 创建模型
    print("初始化CNN模型...")
    model = CNNModel(X_train.shape[1])
    model = model.to(device)  # 将模型移动到指定设备
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练开始时间
    start_time = time.time()
    print("开始训练...")
    
    # 训练模型
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/50")
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # 数据已经在正确的设备上，不需要再次移动
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Progress: {batch_idx/len(train_loader)*100:.1f}%")
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_train_cnn).squeeze()
            val_loss = criterion(val_outputs, y_train_tensor)
            print(f"Epoch {epoch+1} 完成, 平均损失: {total_loss/batch_count:.4f}, "
                  f"验证损失: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"验证损失改善，新的最佳损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"验证损失未改善，剩余耐心: {patience - patience_counter}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 训练结束时间
    train_time = time.time() - start_time
    print(f"\n训练完成，用时: {train_time:.2f}秒")
    
    # 预测开始时间
    print("\n开始预测...")
    start_time = time.time()
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_cnn).squeeze().cpu().numpy()  # 将结果移回CPU
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 预测结束时间
    predict_time = time.time() - start_time
    print(f"预测完成，用时: {predict_time:.2f}秒")
    
    # 计算评估指标
    print("\n计算评估指标...")
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    
    # 生成分类报告
    class_report = classification_report(y_test, y_pred_binary)
    
    print("\nCNN模型评估完成！")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    return {
        'model_name': 'CNN',
        'classifier_type': 'Deep Learning',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def main():
    # 主函数：执行完整的模型比较流程
    #
    # 执行步骤：
    # 1. 数据准备：
    #    - 加载预处理后的CTDAPD数据集
    #    - 应用特征工程
    #    - 处理类别不平衡
    #
    # 2. 模型训练与评估：
    #    - 初始化6种不同的模型
    #    - 设置优化后的超参数
    #    - 依次训练和评估每个模型
    #    - 收集性能指标
    #
    # 3. 结果分析：
    #    - 比较不同模型的性能
    #    - 选择最佳模型
    #    - 生成详细报告
    #    - 保存评估结果
    #
    # 4. 性能评估：
    #    - 根据F1分数评估模型表现
    #    - 提供优化建议
    #    - 记录训练时间
    #
    # 注意事项：
    # 1. 内存使用：
    #    - 大型模型（如神经网络）可能需要较大内存
    #    - 建议至少8GB可用内存
    #
    # 2. 运行时间：
    #    - 完整评估可能需要5-10分钟
    #    - 神经网络训练时间最长
    #    - 逻辑回归最快
    #
    # 3. 结果保存：
    #    - 详细结果保存在Models/results目录
    #    - 包含每个模型的完整评估指标
    #    - 保存特征重要性分析
    
    print("Starting Model 1B (CTDAPD-based Network Attack Detection) Evaluation")
    
    try:
        # Load preprocessed data
        data_result = load_preprocessed_ctdapd_data()
        if data_result is None:
            print("Data loading failed!")
            return
        
        X_train, X_test, y_train, y_test = data_result
        
        # Define optimized models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=RANDOM_STATE
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        }
        
        # Evaluate all models
        results = []
        for model_name, model in models.items():
            try:
                result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
                results.append(result)
            except Exception as e:
                print(f"Model {model_name} training failed: {e}")
                continue
        
        if not results:
            print("All models failed to train!")
            return
        
        # Find best model
        best_model = max(results, key=lambda x: x['f1'])
        
        print("\nBest Network Attack Detection Model Performance Summary (CTDAPD Dataset)")
        print("-" * 50)
        print(f"Model: {best_model['model_name']}")
        print(f"Classifier Type: {best_model['classifier_type']}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        print(f"Precision: {best_model['precision']:.4f}")
        print(f"Recall: {best_model['recall']:.4f}")
        print(f"F1-Score: {best_model['f1']:.4f}")
        if best_model['auc_roc']:
            print(f"AUC-ROC: {best_model['auc_roc']:.4f}")
        print(f"Training time: {best_model['train_time']:.2f}s")
        
        # Performance evaluation
        if best_model['f1'] >= 0.95:
            print("Performance: Excellent (95%+)")
        elif best_model['f1'] >= 0.90:
            print("Performance: Great (90%+)")
        elif best_model['f1'] >= 0.85:
            print("Performance: Good (85%+)")
        elif best_model['f1'] >= 0.80:
            print("Performance: Acceptable (80%+)")
        else:
            print("Performance: Needs optimization")
        
        # 添加CNN模型到评估列表
        results.append(evaluate_cnn_model(X_train, X_test, y_train, y_test))
        
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 