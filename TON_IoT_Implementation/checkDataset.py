import pandas as pd
import glob
import os
from collections import Counter

# Set the directory where the dataset is located (ensure the path is correct)
dataset_dir = "/mnt/d/FYP_TON/Dataset"

# 1. Check features (using the first file found)
sample_file = glob.glob(os.path.join(dataset_dir, "*.csv"))[0]
df_sample = pd.read_csv(sample_file, nrows=5)
print("="*60)
print("1. Feature Columns:")
print(df_sample.columns.tolist())
print("="*60)

# 2. Count attack types across all files (using Counter to avoid memory overflow)
total_attack_types = Counter()
total_records = 0

print("\n2. Counting attack types across all files (this may take a moment)...")

csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

for file in csv_files:
    print(f"Processing: {os.path.basename(file)}", end="\r")
    # Only read the 'type' column to significantly save memory
    try:
        df_type = pd.read_csv(file, usecols=['type'], low_memory=False)
        total_attack_types.update(df_type['type'].value_counts().to_dict())
        total_records += len(df_type)
    except Exception as e:
        print(f"\nSkipping {file}: {e}")

print(f"\n\nCounting complete! Total records: {total_records}")
print("\n3. Attack Type Distribution:")
print(f"{'Attack Type':<30} | {'Count':<15} | {'Percentage'}")
print("-" * 65)

for attack, count in total_attack_types.most_common():
    percentage = (count / total_records) * 100
    print(f"{str(attack):<30} | {count:<15} | {percentage:.2f}%")