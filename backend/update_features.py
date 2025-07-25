import pandas as pd

# 读取数据集
df = pd.read_csv('data/network_score_sampled_from100w.csv')

# 排除不需要的字段
exclude_columns = ['Label', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'BinaryLabel', 'network_score', 'Timestamp']
features = [col for col in df.columns if col not in exclude_columns]

# 写入特征文件
with open('features/network_features.txt', 'w') as f:
    for feature in features:
        f.write(f'{feature}\n')

print(f'Network features file updated with {len(features)} features')
print('First 10 features:', features[:10])
print('Last 10 features:', features[-10:]) 