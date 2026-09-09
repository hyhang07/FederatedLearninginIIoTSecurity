import pandas as pd
import glob
import os
import joblib
import numpy as np
import json

from Multi_DataUtils_Client import load_and_prepare_data_for_dl
from Multi_DataUtils import convert_to_granular_labels

def main():
    """
    Creates specialized validation sets for each Edge Server zone.
    """
    print("--- Creating Specialized Validation Sets for Each Edge Server Zone ---")

    TOTAL_SAMPLES_PER_SET = 2000

    # Configuration for each specialized zone
    ZONE_CONFIG = {
        'ddos_http': {'Normal': 0.5, 'DDoS_HTTP_Flood_attack.csv': 0.5},
        'ddos_icmp': {'Normal': 0.5, 'DDoS_ICMP_Flood_attack.csv': 0.5},
        'ddos_tcp':  {'Normal': 0.5, 'DDoS_TCP_SYN_Flood_attack.csv': 0.5},
        'ddos_udp':  {'Normal': 0.5, 'DDoS_UDP_Flood_attack.csv': 0.5},
        'password':  {'Normal': 0.5, 'Password_attack.csv': 0.5},
        'backdoor':  {'Normal': 0.5, 'Backdoor_attack.csv': 0.5},
        'os_fingerprinting': {'Normal': 0.65, 'OS_Fingerprinting_attack.csv': 0.35},
        'port_scanning': {'Normal': 0.5, 'Port_Scanning_attack.csv': 0.5},
        'ransomware': {'Normal': 0.5, 'Ransomware_attack.csv': 0.5},
        'sql_injection': {'Normal': 0.5, 'SQL_injection_attack.csv': 0.5},
        'uploading': {'Normal': 0.5, 'Uploading_attack.csv': 0.5},
        'vulnerability_scanner': {'Normal': 0.5, 'Vulnerability_scanner_attack.csv': 0.5},
        'xss': {'Normal': 0.65, 'XSS_attack.csv': 0.35},
        'benign':    {'Normal': 0.9, 'Password_attack.csv': 0.05, 'Backdoor_attack.csv': 0.05}
    }

    base_path = "Dataset/Edge-IIoTset dataset"
    base_attack_path = os.path.join(base_path, "Attack traffic")
    output_base_dir = "validation_data_multi"
    os.makedirs(output_base_dir, exist_ok=True)

    # === Part 1: Generate the Global Schemas for Consistency ===
    print("\n--- Part 1: Generating Global Schemas for Consistency ---")
    
    # Define all required attack files for schema generation
    all_required_attack_files = [
        'DDoS_HTTP_Flood_attack.csv',
        'DDoS_ICMP_Flood_attack.csv', 
        'DDoS_TCP_SYN_Flood_attack.csv',
        'DDoS_UDP_Flood_attack.csv',
        'Password_attack.csv',
        'Backdoor_attack.csv',
        'OS_Fingerprinting_attack.csv',
        'Port_Scanning_attack.csv',
        'Ransomware_attack.csv',
        'SQL_injection_attack.csv',
        'Uploading_attack.csv',
        'Vulnerability_scanner_attack.csv',
        'XSS_attack.csv'
    ]
    
    print(f"   - Verifying all {len(all_required_attack_files)} required attack files exist...")
    missing_files = [f for f in all_required_attack_files if not os.path.exists(os.path.join(base_attack_path, f))]
    
    if missing_files:
        print(f"\n   FATAL ERROR: Missing required files: {missing_files}")
        return
    
    # Create a temporary dataframe from a sample of all required files
    temp_df_list = []
    
    # Add Normal traffic sample
    normal_paths = glob.glob(os.path.join(base_path, "Normal traffic", "*", "*.csv"))
    if not normal_paths:
        print("   ERROR: No Normal traffic files found!")
        return
    temp_df_list.append(pd.read_csv(normal_paths[0], low_memory=False).sample(n=200, random_state=42))
    
    # Add samples from all attack files
    for fname in all_required_attack_files:
        attack_path = os.path.join(base_attack_path, fname)
        sample_df = pd.read_csv(attack_path, low_memory=False).sample(n=200, random_state=42)
        temp_df_list.append(sample_df)
        
    temp_schema_df_raw = pd.concat(temp_df_list, ignore_index=True)
    temp_schema_df_processed = convert_to_granular_labels(temp_schema_df_raw)
    
    temp_path = os.path.join(output_base_dir, "temp_schema_generator.csv")
    temp_schema_df_processed.to_csv(temp_path, index=False)
    
    # Generate and save the global schemas (label encoder, class order, input shape)
    try:
        _, _, _, _, le, _, n_features = load_and_prepare_data_for_dl(temp_path, is_creator=True)
        
        class_order = le.classes_.tolist()
        class_order_path = os.path.join(output_base_dir, "class_order.json")
        with open(class_order_path, "w") as f:
            json.dump(class_order, f)
        print(f"\n   - Saved GLOBAL canonical class order to '{class_order_path}'")
        print(f"   - Total classes in schema: {len(class_order)}")

        with open(os.path.join(output_base_dir, "model_input_shape_multi.txt"), "w") as f:
            f.write(f"1,{n_features}")
        print(f"   - Saved GLOBAL model input shape: (1, {n_features})")

    except Exception as e:
        print(f"   ERROR during schema generation: {e}")
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("--- Global schemas created successfully. ---\n")

    # === Part 2: Generate Specialized Validation Set for Each Zone ===
    print("--- Part 2: Generating Zone-Specific Validation Sets ---")
    for zone, composition in ZONE_CONFIG.items():
        print(f"\n--- Creating validation set for zone: {zone} ---")
        zone_output_dir = os.path.join(output_base_dir, zone)
        os.makedirs(zone_output_dir, exist_ok=True)
        df_list = []
        
        for data_type, ratio in composition.items():
            num_samples = int(TOTAL_SAMPLES_PER_SET * ratio)
            if num_samples == 0: continue
            
            print(f"   - Sampling {num_samples} '{data_type}' records...")
            
            if data_type == 'Normal':
                normal_paths = glob.glob(os.path.join(base_path, "Normal traffic", "*", "*.csv"))
                if normal_paths:
                    # --- [MEMORY FIX] ---
                    # Sample from each file individually, then combine the small samples.
                    # Do not load all full files into memory at once.
                    samples_per_file = num_samples // len(normal_paths)
                    if samples_per_file > 0:
                        normal_dfs = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=123) for f in normal_paths]
                        df_list.append(pd.concat(normal_dfs, ignore_index=True))
                    # --- [END OF MEMORY FIX] ---
            else:
                file_path = os.path.join(base_attack_path, data_type)
                if os.path.exists(file_path):
                    df_temp = pd.read_csv(file_path, low_memory=False)
                    df_list.append(df_temp.sample(n=min(len(df_temp), num_samples), random_state=123))
                else:
                    print(f"   WARNING: File not found: {file_path}")

        if not df_list:
            print(f"   No data collected for zone '{zone}' - skipping.")
            continue
            
        zone_df_raw = pd.concat(df_list, ignore_index=True)
        zone_df_processed = convert_to_granular_labels(zone_df_raw)
        zone_df_shuffled = zone_df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
        temp_path = os.path.join(zone_output_dir, "temp_validation.csv")
        zone_df_shuffled.to_csv(temp_path, index=False)
        
        # Process the zone data using the global schema
        try:
            X_train_val, X_test_val, y_train_val, y_test_val, _, scaler, _ = load_and_prepare_data_for_dl(temp_path, is_creator=False)
            X_val = np.concatenate((X_train_val, X_test_val), axis=0)
            y_val = np.concatenate((y_train_val, y_test_val), axis=0)
            
            np.save(os.path.join(zone_output_dir, "X_val.npy"), X_val)
            np.save(os.path.join(zone_output_dir, "y_val.npy"), y_val)
            joblib.dump(scaler, os.path.join(zone_output_dir, "validation_scaler.joblib"))
            print(f"   - Zone validation set for '{zone}' created successfully.")
        except Exception as e:
            print(f"   Error creating validation set for zone '{zone}': {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print("\n=== Validation Set Creation Complete ===")

if __name__ == "__main__":
    main()