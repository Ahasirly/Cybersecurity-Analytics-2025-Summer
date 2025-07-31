import pandas as pd

# Read user dataset
data = pd.read_csv('data/user6_final.csv')

# Exclude unnecessary fields
feature_columns = [col for col in data.columns if col not in ['id', 'label']]

# Write to feature file
with open('features/user6_features.txt', 'w') as f:
    for feature in feature_columns:
        f.write(f"{feature}\n") 