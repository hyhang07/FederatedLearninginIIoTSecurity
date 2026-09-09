import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm

from Multi_DataUtils import convert_to_granular_labels

def create_and_save_dataset_group(zone_name, client_id_start, num_clients, max_records_per_client=100000):
    """
    Creates client datasets using the "Enriched Specialist" strategy.
    """
    print(f"\n===== Generating ENRICHED SPECIALIST datasets for zone: '{zone_name}' =====")
    
    base_path = "Dataset/Edge-IIoTset dataset"
    base_attack_path = os.path.join(base_path, "Attack traffic")
    output_base_dir = "Dataset/Multi"
    
    # --- [VERIFIED CONFIGURATION] ---
    # This map is now 100% consistent with your file system screenshot.
    attack_file_map = {
        'ddos_http': ["DDoS_HTTP_Flood_attack.csv"],
        'ddos_icmp': ["DDoS_ICMP_Flood_attack.csv"],
        'ddos_tcp':  ["DDoS_TCP_SYN_Flood_attack.csv"],
        'ddos_udp':  ["DDoS_UDP_Flood_attack.csv"],
        'password':  ["Password_attack.csv"],
        'backdoor':  ["Backdoor_attack.csv"],
        'os_fingerprinting': ["OS_Fingerprinting_attack.csv"],
        'port_scanning': ["Port_Scanning_attack.csv"],
        'ransomware': ["Ransomware_attack.csv"],
        'sql_injection': ["SQL_injection_attack.csv"],
        'uploading': ["Uploading_attack.csv"],
        'vulnerability_scanner': ["Vulnerability_scanner_attack.csv"],
        'xss': ["XSS_attack.csv"], # Corrected from XSS_attacks.csv
        'benign':    []
    }
    # --- [END OF VERIFICATION] ---

    FOCUSED_BACKGROUND_FILES = [
        "DDoS_HTTP_Flood_attack.csv",
        "Password_attack.csv",
        "Backdoor_attack.csv",
        "Port_Scanning_attack.csv",
        "XSS_attack.csv" # Corrected from XSS_attacks.csv
    ]
    focused_background_paths = [os.path.join(base_attack_path, fname) for fname in FOCUSED_BACKGROUND_FILES]

    specialty_filenames = attack_file_map.get(zone_name)
    if specialty_filenames is None:
        raise ValueError(f"Invalid zone_name provided: {zone_name}")

    zone_output_dir = os.path.join(output_base_dir, zone_name)
    os.makedirs(zone_output_dir, exist_ok=True)
    
    for i in range(num_clients):
        client_id = client_id_start + i
        print(f"\n--- Preparing dataset for Client {client_id} in zone '{zone_name}' ---")

        final_dfs = []

        if zone_name == 'benign':
            NORMAL_RATIO = 0.95
            BACKGROUND_RATIO = 0.05
            SPECIALTY_RATIO = 0.0
        else:
            NORMAL_RATIO = 0.65
            SPECIALTY_RATIO = 0.30
            BACKGROUND_RATIO = 0.05

        num_normal = int(max_records_per_client * NORMAL_RATIO)
        if num_normal > 0:
            normal_traffic_paths = glob.glob(os.path.join(base_path, "Normal traffic", "*", "*.csv"))
            samples_per_file = num_normal // len(normal_traffic_paths) if len(normal_traffic_paths) > 0 else num_normal
            if samples_per_file > 0:
                df_list = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=client_id) for f in normal_traffic_paths]
                final_dfs.append(pd.concat(df_list, ignore_index=True))

        num_background = int(max_records_per_client * BACKGROUND_RATIO)
        if num_background > 0:
            paths_for_this_client_mix = [p for p in focused_background_paths if not any(spec in p for spec in specialty_filenames)] if specialty_filenames else focused_background_paths
            if paths_for_this_client_mix:
                samples_per_file = num_background // len(paths_for_this_client_mix)
                if samples_per_file > 0:
                    df_list = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=client_id) for f in paths_for_this_client_mix]
                    final_dfs.append(pd.concat(df_list, ignore_index=True))
            
        num_specialty = int(max_records_per_client * SPECIALTY_RATIO)
        if num_specialty > 0:
            specialty_paths = [os.path.join(base_attack_path, fname) for fname in specialty_filenames]
            samples_per_file = num_specialty // len(specialty_paths)
            if samples_per_file > 0:
                df_list = [pd.read_csv(f, low_memory=False).sample(n=min(len(pd.read_csv(f, low_memory=False)), samples_per_file), random_state=client_id) for f in specialty_paths]
                final_dfs.append(pd.concat(df_list, ignore_index=True))
        
        client_df_raw = pd.concat(final_dfs, ignore_index=True)
        client_df_shuffled = client_df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
        final_df_granular = convert_to_granular_labels(client_df_shuffled)
        
        output_filename = f"client_{client_id}_dataset.csv"
        output_path = os.path.join(zone_output_dir, output_filename)
        final_df_granular.to_csv(output_path, index=False)
        
        print(f"   - Saved dataset to: {output_path} ({len(final_df_granular)} records)")
        print(f"   - Final label distribution:\n{final_df_granular['Attack_type'].value_counts().to_string()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Factory for creating ENRICHED SPECIALIST client datasets.")
    parser.add_argument('--records', type=int, default=100000, help="MAXIMUM number of records per client dataset.")
    args = parser.parse_args()

    # This block now creates clients for all 15 zones (14 attack + 1 benign).
    create_and_save_dataset_group(zone_name='ddos_http', client_id_start=1, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='ddos_icmp', client_id_start=4, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='ddos_tcp', client_id_start=7, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='ddos_udp', client_id_start=10, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='password', client_id_start=13, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='backdoor', client_id_start=16, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='os_fingerprinting', client_id_start=19, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='port_scanning', client_id_start=22, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='ransomware', client_id_start=25, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='sql_injection', client_id_start=28, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='uploading', client_id_start=31, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='vulnerability_scanner', client_id_start=34, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='xss', client_id_start=37, num_clients=3, max_records_per_client=args.records)
    create_and_save_dataset_group(zone_name='benign', client_id_start=40, num_clients=3, max_records_per_client=args.records)
    
    print("\n\n--- All 42 'Enriched Specialist' datasets have been created and saved in 'Dataset/Multi/' ---")