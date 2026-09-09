# EdgeServer_NoDefense.py (Baseline: Standard FedAvg - Vulnerable)
import os
import glob
import argparse
import time
import numpy as np
import tensorflow as tf

# --- Global Constants ---
NUM_CLASSES = 9

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    parser = argparse.ArgumentParser(description="Edge Server WITHOUT Defense (Baseline).")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory to find incoming sparse updates from clients.")
    parser.add_argument('--global_upload_dir', type=str, required=True, help="Directory to upload the single, aggregated zone model to.")
    parser.add_argument('--zone_name', type=str, required=True, help="The name of the zone this server is managing.")
    parser.add_argument('--num_clients', type=int, required=True, help="The number of clients this server expects updates from.")
    parser.add_argument('--poll_interval', type=int, default=10, help="How often (in seconds) to check for new updates.")
    args = parser.parse_args()

    print(f"\n--- [BASELINE] Edge Server (No Defense) for Zone '{args.zone_name}' ---")
    print("    - Mode: Naive FedAvg (Simple Averaging - VULNERABLE)")
    
    os.makedirs(args.global_upload_dir, exist_ok=True)

    # --- Wait for Client Updates ---
    model_paths = []
    print(f"   - Waiting for {args.num_clients} client updates...")
    while len(model_paths) < args.num_clients:
        model_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
        print(f"\r   - Found {len(model_paths)} / {args.num_clients}", end="")
        time.sleep(args.poll_interval)
    
    print(f"\n--- All {args.num_clients} updates received. Starting Naive Aggregation. ---")

    collected_updates = []

    # --- Naive Collection (Accepts Everything) ---
    for path in model_paths:
        try:
            sparse_update_data = np.load(path)
            # Ensure correct loading order
            sorted_files = sorted(sparse_update_data.files, key=lambda x: int(x.split('_')[1]))
            sparse_update = [sparse_update_data[arr] for arr in sorted_files]
            
            collected_updates.append(sparse_update)
            print(f"   - [Accepted] {os.path.basename(path)} (No Validation Performed)")
            
        except Exception as e:
            print(f"   - [WARNING] Error loading {path}: {e}")

    if not collected_updates:
        print("FATAL: No valid updates were collected. Aborting zone aggregation.")
        return

    # --- Aggregation: Simple Mean (FedAvg) ---
    print(f"\n--- Performing Simple Average on {len(collected_updates)} models ---")
    
    # Check if all updates have the same structure
    if len(set(len(u) for u in collected_updates)) > 1:
        print("FATAL: Client updates have inconsistent layer counts. Cannot aggregate.")
        return
        
    num_layers = len(collected_updates[0])
    zone_aggregated_update = []
    
    for i in range(num_layers):
        # Stack updates for this layer
        try:
            layer_stack = np.array([u[i] for u in collected_updates])
            # Compute Mean (Highly vulnerable to outliers/poisoning)
            mean_layer = np.mean(layer_stack, axis=0) 
            zone_aggregated_update.append(mean_layer)
        except ValueError as e:
            print(f"ERROR: Could not aggregate layer {i}. Mismatched shapes: {e}")
            print("Aborting aggregation for this zone.")
            return

    # Save and upload the aggregated update
    zone_output_filename = f"zone_{args.zone_name}.sparse.npz"
    zone_output_path = os.path.join(args.global_upload_dir, zone_output_filename)
    np.savez_compressed(zone_output_path, *zone_aggregated_update)
    
    print(f"   - Zone model uploaded (without any checks) to: {zone_output_path}")
    
    # Signal that this zone's task is complete
    os.makedirs('edge_server_signals', exist_ok=True)
    with open(f"edge_server_signals/{args.zone_name}.done", 'w') as f:
        f.write('done')
    
    print(f"--- [Finished] Baseline Edge Server for Zone '{args.zone_name}' has completed its task. ---")

if __name__ == '__main__':
    main()