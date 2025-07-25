import pandas as pd

# 读取用户数据集
df = pd.read_csv('data/user_encoded_dataset.csv')

# 排除不需要的字段
exclude_columns = ['sample_id', 'label']
features = [col for col in df.columns if col not in exclude_columns]

# 写入特征文件
with open('features/user6_features.txt', 'w') as f:
    for feature in features:
        f.write(f'{feature}\n')

print(f'User features file updated with {len(features)} features')
print('Features:', features) 