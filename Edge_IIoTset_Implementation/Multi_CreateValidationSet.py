import pandas as pd
import glob
import os
import joblib
import numpy as np
import json # Import json to save the class order

# Import the necessary data utilities
from Multi_DataUtils_Client import load_and_prepare_data_for_dl
from Multi_DataUtils import convert_to_multi_labels

def main():
    """
    Creates a multi-class validation set for the Edge Servers with a specific,
    imbalanced ratio of classes to simulate a more realistic environment.
    """
    print("--- Creating a SPECIFIC RATIO MULTI-CLASS validation set for Edge Servers ---")
    
    # --- [CRITICAL CONFIGURATION] ---
    # Define the total size of the validation set and the desired percentage for each class.
    TOTAL_SAMPLES = 4000 
    
    RATIOS = {
        'Normal': 0.40,   
        'DDoS': 0.25,    
        'Password': 0.15, 
        'Backdoor': 0.15  
    }
    # --- [END OF CONFIGURATION] ---

    # Calculate the exact number of samples needed for each class
    samples_to_get = {name: int(TOTAL_SAMPLES * ratio) for name, ratio in RATIOS.items()}
    
    base_path = "Dataset/Edge-IIoTset dataset"
    base_attack_path = os.path.join(base_path, "Attack traffic")
    output_dir = "validation_data_multi"
    os.makedirs(output_dir, exist_ok=True)

    df_list = []

    # === Step 1: Sample 'Normal' Records ===
    num_normal = samples_to_get['Normal']
    print(f"   - Sampling {num_normal} 'Normal' records ({RATIOS['Normal']:.2%})...")
    normal_paths = glob.glob(os.path.join(base_path, "Normal traffic", "*", "*.csv"))
    samples_per_file = num_normal // len(normal_paths)
    normal_dfs = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=123) for f in normal_paths]
    df_list.append(pd.concat(normal_dfs, ignore_index=True))
    
    # === Step 2: Sample 'DDoS' Records ===
    num_ddos = samples_to_get['DDoS']
    print(f"   - Sampling {num_ddos} 'DDoS' records ({RATIOS['DDoS']:.2%}) from all types...")
    ddos_paths = [
        os.path.join(base_attack_path, "DDoS_HTTP_Flood_attack.csv"),
        os.path.join(base_attack_path, "DDoS_ICMP_Flood_attack.csv"),
        os.path.join(base_attack_path, "DDoS_TCP_SYN_Flood_attack.csv"),
        os.path.join(base_attack_path, "DDoS_UDP_Flood_attack.csv")
    ]
    samples_per_file = num_ddos // len(ddos_paths)
    ddos_dfs = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=123) for f in ddos_paths]
    df_list.append(pd.concat(ddos_dfs, ignore_index=True))

    # === Step 3: Sample 'Password' Records ===
    num_password = samples_to_get['Password']
    print(f"   - Sampling {num_password} 'Password' records ({RATIOS['Password']:.2%})...")
    password_path = os.path.join(base_attack_path, "Password_attack.csv")
    password_df = pd.read_csv(password_path, low_memory=False).sample(n=min(len(pd.read_csv(password_path, low_memory=False)), num_password), random_state=123)
    df_list.append(password_df)

    # === Step 4: Sample 'Backdoor' Records ===
    num_backdoor = samples_to_get['Backdoor']
    print(f"   - Sampling {num_backdoor} 'Backdoor' records ({RATIOS['Backdoor']:.2%})...")
    backdoor_path = os.path.join(base_attack_path, "Backdoor_attack.csv")
    backdoor_df = pd.read_csv(backdoor_path, low_memory=False).sample(n=min(len(pd.read_csv(backdoor_path, low_memory=False)), num_backdoor), random_state=123)
    df_list.append(backdoor_df)

    # === Step 5: Combine, process, shuffle, and save the raw set ===
    validation_df_raw = pd.concat(df_list, ignore_index=True)
    validation_df_processed = convert_to_multi_labels(validation_df_raw)
    validation_df_shuffled = validation_df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    
    temp_path = os.path.join(output_dir, "temp_validation_multi.csv")
    validation_df_shuffled.to_csv(temp_path, index=False)
    
    # === Step 6: Process the data using the 'creator' pipeline ===
    print("\n   - Processing data using the client's pipeline for consistency...")
    X_train_val, X_test_val, y_train_val, y_test_val, le, scaler, n_features = load_and_prepare_data_for_dl(temp_path, timesteps=1, is_creator=True)
    
    X_val = np.concatenate((X_train_val, X_test_val), axis=0)
    y_val = np.concatenate((y_train_val, y_test_val), axis=0)

    # === Step 7: Save all artifacts ===
    class_order = le.classes_.tolist()
    class_order_path = os.path.join(output_dir, "class_order.json")
    with open(class_order_path, "w") as f:
        json.dump(class_order, f)
    print(f"\n   - Saved canonical class order to '{class_order_path}': {class_order}")

    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    joblib.dump(scaler, os.path.join(output_dir, "validation_scaler_multi.joblib"))
    
    with open(os.path.join(output_dir, "model_input_shape_multi.txt"), "w") as f:
        f.write(f"{X_val.shape[1]},{X_val.shape[2]}")

    # === Step 8: Cleanup and final verification ===
    os.remove(temp_path)
    print(f"\n--- VALIDATION SET with {len(X_val)} records created successfully in '{output_dir}/' folder. ---")
    
    label_counts = pd.Series(y_val).value_counts().sort_index()
    print("   - Final label distribution (Integers mapped to Classes):")
    for label_int, count in label_counts.items():
        class_name = le.inverse_transform([label_int])[0]
        percentage = (count / len(y_val)) * 100
        print(f"      {class_name} (label {label_int}): {count} samples (~{percentage:.2f}%)")

if __name__ == "__main__":
    main()