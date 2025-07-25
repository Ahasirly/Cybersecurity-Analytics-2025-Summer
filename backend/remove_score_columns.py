import pandas as pd
import os
from pathlib import Path

def remove_score_columns():
    """Remove score columns from all three datasets"""
    
    # 定义数据集路径
    data_dir = Path("data")
    
    # 数据集文件列表
    datasets = [
        "URL_model_input_score_0.1_0.98.csv",
        "network_score_sampled_from100w.csv", 
        "user_encoded_dataset.csv"
    ]
    
    # 要删除的列名（模型不需要的score列）
    score_columns = [
        'score', 'network_score'  # 这些是采样用的，模型不需要
    ]
    
    for dataset_file in datasets:
        file_path = data_dir / dataset_file
        if not file_path.exists():
            print(f"⚠️  File not found: {dataset_file}")
            continue
            
        print(f"\n📁 Processing: {dataset_file}")
        
        # 读取数据集
        try:
            df = pd.read_csv(file_path)
            print(f"   Original shape: {df.shape}")
            print(f"   Original columns: {list(df.columns)}")
            
            # 找到要删除的score列
            columns_to_remove = []
            for col in score_columns:
                if col in df.columns:
                    columns_to_remove.append(col)
                    print(f"   Found score column: {col}")
            
            if columns_to_remove:
                # 删除score列
                df_cleaned = df.drop(columns=columns_to_remove)
                print(f"   Removed columns: {columns_to_remove}")
                print(f"   New shape: {df_cleaned.shape}")
                
                # 备份原文件
                backup_path = file_path.with_suffix('.csv.backup')
                df.to_csv(backup_path, index=False)
                print(f"   Backup created: {backup_path.name}")
                
                # 保存清理后的文件
                df_cleaned.to_csv(file_path, index=False)
                print(f"   ✅ Cleaned file saved: {dataset_file}")
            else:
                print(f"   ℹ️  No score columns found in {dataset_file}")
                
        except Exception as e:
            print(f"   ❌ Error processing {dataset_file}: {str(e)}")
    
    print(f"\n🎉 Score column removal completed!")
    print(f"📋 Summary:")
    print(f"   - Original files backed up with .backup extension")
    print(f"   - Score columns removed from datasets")
    print(f"   - Cleaned datasets ready for training")

if __name__ == "__main__":
    remove_score_columns() 