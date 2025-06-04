# 网络安全检测与用户行为预测项目

## 项目概述
- **项目名称**: 2025 Summer Direct Study Proposal
- **参考文献**: IEEE 论文 "Research on Malicious URL Detection Based on Random Forest"

## 数据集说明

### 1. CIC_Dataset.csv
- **记录数**: 15,367
- **特征数**: 78列
- **主要特征**:
  - urlLen
  - NumberofDotsinURL
  - Querylength
  - domain_token_count
  - path_token_count
- **目标变量**: URL_Type_obf_Type (良性, 篡改, 垃圾邮件)

### 2. merged_dataset.csv
- **时间范围**: 2018-2024
- **特征类型**:
  - **网络流量特征**:
    - CPU_Utilization (CPU使用率)
    - Phishing_Attempts (钓鱼攻击尝试)
    - Risky_Website_Visits (风险网站访问)
    - Anomaly_Score (异常分数)
  - **行为特征**:
    - Device_Type (设备类型)
    - Age_Group (年龄组)
    - Social_Media_Usage (社交媒体使用)
    - E_Safety_Awareness_Score (网络安全意识分数)
- **目标变量**:
  - Insecurity_Level (安全风险等级-二分类)
  - Hours_Online (在线时长-连续值)
  - Cybersecurity_Behavior_Category (网络安全行为类别-多分类)

## 项目结构
```
Cybersecurity-Analytics-2025-Summer/
├── src/
│   ├── data_cleaning_utils.py  # 数据清理工具函数
│   └── clean_datasets.py       # 主数据清理脚本
├── Datasets/
│   ├── original/              # 原始数据集
│   ├── train/                 # 训练数据集
│   └── test/                  # 测试数据集
├── venv/                      # Python虚拟环境
├── requirements.txt           # 项目依赖
└── README.md                  # 项目文档
```

## 数据预处理实现

### 1. 工具函数 (data_cleaning_utils.py)
- **print_dataset_info**: 显示数据集基本信息
  - 数据维度
  - 特征类型统计
  - 缺失值统计
- **handle_missing_values**: 处理缺失值
  - 数值特征: 均值/中位数填充
  - 分类特征: 众数/'None'填充
  - 记录处理日志
- **handle_outliers**: 异常值处理
  - URL特征: 99百分位限制
  - 时间特征: [0,24]限制
  - 生成异常值报告
- **encode_categorical_features**: 特征编码
  - 独热编码: 低基数分类变量
  - 标签编码: 高基数分类变量
  - 保存编码映射
- **standardize_features**: 特征标准化
  - 保存标准化参数
  - 支持逆转换
- **select_features**: 基于随机森林的特征选择
  - 特征重要性排序
  - 交叉验证评估
  - 导出特征选择报告

### 2. 数据清理流程 (clean_datasets.py)

#### CIC数据集清理
1. 缺失值处理
   - 数值特征用均值填补
   - 分类特征用众数填补
   - 生成缺失值报告
2. 异常值处理
   - urlLen等限制在99百分位
   - 记录异常值分布
3. 特征处理
   - URL_Type_obf_Type转换为二分类
   - 数值特征标准化到[0,1]
   - 保存转换参数
4. 特征选择
   - 使用随机森林选择前10个特征
   - 生成特征重要性图表
5. 数据分割
   - 80%训练集
   - 20%测试集
   - 保持类别分布一致

#### merged数据集清理
1. 时间特征处理
   - 提取Hour, DayOfWeek, Month
   - 创建时间窗口特征
2. 缺失值处理
   - 数值特征用中位数填补
   - 分类特征用"None"填补
   - 记录处理日志
3. 异常值处理
   - Hours_Online限制[0,24]
   - CPU_Utilization限制[0,100]
   - 生成箱线图
4. 特征编码
   - 分类特征独热编码
   - 目标变量标签编码
   - 保存编码字典
5. 特征选择
   - 选择前15个重要特征
   - 生成特征相关性热图
6. 时间序列分割
   - 2018-2022训练集
   - 2023-2024测试集
   - 确保时间连续性

## 项目依赖
```python
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## 使用说明

### 环境设置
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行数据清理
```bash
# 进入src目录
cd src

# 运行数据清理脚本
python clean_datasets.py
```

## 后续步骤
1. 模型训练 (6月5日)
   - 模型1A: URL检测 (恶意URL识别)
   - 模型1B: 网络流量检测 (异常流量识别)
   - 模型2: 在线时间预测 (用户行为分析)
   - 模型3: 行为风险分类 (综合风险评估)
2. 实现浏览器部署
   - 开发Chrome扩展
   - 实现实时URL检查
3. 测试实时检测
   - 性能测试
   - 准确度验证
4. 提交提案报告
   - 技术文档
   - 实验结果分析

## 预期成果
- 模型1: 准确率>98.6%
  - 低误报率
  - 快速响应时间
- 模型2: 最小MSE，R²接近1
  - 准确的时间预测
  - 可解释的预测结果
- 模型3: 高F1分数，准确识别高风险用户
  - 精确的风险分类
  - 可操作的安全建议

## 时间节点
- 6月4日: 完成数据清理
- 6月5日: 开始模型训练
- 6月中旬: 实现部署
- 6月底: 提交报告 