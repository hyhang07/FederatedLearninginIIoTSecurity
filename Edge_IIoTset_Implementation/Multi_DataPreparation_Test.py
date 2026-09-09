import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm

# Import the multi-label conversion utility to ensure consistent labeling
from Multi_DataUtils import convert_to_granular_labels

def create_and_save_global_test_set(total_samples=80000, normal_ratio=0.5):
    """
    Creates a single, unified, and comprehensive global test set for multi-class evaluation.
    It samples from ALL available attack types to create a diverse and balanced test set.
    """
    print(f"\n===== Generating a COMPREHENSIVE Multi-Class Global Test Set =====")
    print(f"   - Target size: {total_samples} records")
    print(f"   - Composition: {normal_ratio*100:.0f}% Normal, {(1-normal_ratio)*100:.0f}% Attack")
    
    base_path = "Dataset/Edge-IIoTset dataset"
    base_attack_path = os.path.join(base_path, "Attack traffic")
    output_dir = "Dataset/Multi_Test"
    os.makedirs(output_dir, exist_ok=True)
    
    final_dfs = []

    # --- Step 1: Sample 'Normal' traffic ---
    num_normal_to_sample = int(total_samples * normal_ratio)
    print(f"\n--- Sampling {num_normal_to_sample} 'Normal' traffic records...")
    normal_traffic_paths = glob.glob(os.path.join(base_path, "Normal traffic", "*", "*.csv"))
    
    if not normal_traffic_paths:
        print("WARNING: No 'Normal' traffic files found.")
    else:
        samples_per_normal_file = num_normal_to_sample // len(normal_traffic_paths)
        if samples_per_normal_file > 0:
            normal_dfs_list = []
            for f in tqdm(normal_traffic_paths, desc="Normal Files"):
                df = pd.read_csv(f, low_memory=False)
                # Determine sample size, ensuring it's not more than the file's length
                n_samples = min(len(df), samples_per_normal_file)
                normal_dfs_list.append(df.sample(n=n_samples, random_state=42))
            
            if normal_dfs_list:
                final_dfs.append(pd.concat(normal_dfs_list, ignore_index=True))

    # --- Step 2: Sample a mix of ALL available Attack traffic ---
    num_attack_to_sample = total_samples - len(final_dfs[0]) if final_dfs else total_samples
    all_attack_paths = glob.glob(os.path.join(base_attack_path, "*.csv"))
    
    if not all_attack_paths:
        print("WARNING: No 'Attack' traffic files found.")
    else:
        print(f"\n--- Sampling {num_attack_to_sample} 'Attack' records from all {len(all_attack_paths)} available attack types...")
        samples_per_attack_file = num_attack_to_sample // len(all_attack_paths)
        if samples_per_attack_file > 0:
            attack_dfs_list = []
            for f in tqdm(all_attack_paths, desc="Attack Files"):
                df = pd.read_csv(f, low_memory=False)
                # Determine sample size, ensuring it's not more than the file's length
                n_samples = min(len(df), samples_per_attack_file)
                attack_dfs_list.append(df.sample(n=n_samples, random_state=42))

            if attack_dfs_list:
                final_dfs.append(pd.concat(attack_dfs_list, ignore_index=True))
    
    # --- Step 3: Combine, process labels, shuffle, and save ---
    if not final_dfs:
        print("FATAL: No data was sampled. Aborting.")
        return

    print("\n--- Finalizing the global test set...")
    final_test_df_raw = pd.concat(final_dfs, ignore_index=True)
    
    final_test_df_processed = convert_to_granular_labels(final_test_df_raw)
    
    final_test_df_shuffled = final_test_df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_filename = "global_test_dataset_multi.csv"
    output_path = os.path.join(output_dir, output_filename)
    final_test_df_shuffled.to_csv(output_path, index=False)
    
    print(f"\n--- Global multi-class test set saved successfully to: {output_path} ---")
    print(f"   - Total records in final processed set: {len(final_test_df_shuffled)}")
    print(f"   - Final Label distribution (after filtering to 4 classes):\n{final_test_df_shuffled['Attack_type'].value_counts().to_string()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data factory for creating a unified and comprehensive multi-class global test set.")
    parser.add_argument('--total_records', type=int, default=100000, help="Total number of records for the test set.")
    parser.add_argument('--normal_ratio', type=float, default=0.5, help="Target ratio of normal traffic in the test set.")
    args = parser.parse_args()

    create_and_save_global_test_set(total_samples=args.total_records, normal_ratio=args.normal_ratio)