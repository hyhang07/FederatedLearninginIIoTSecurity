# EdgeServer_NoDefense.py (BASELINE: Standard FedAvg - Vulnerable)
import os
import glob
import argparse
import time
import numpy as np
import tensorflow as tf

# Import model creation function
from Multi_Client import create_tcn_gru_model

# --- Global Constants ---
NUM_CLASSES = 11

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    parser = argparse.ArgumentParser(description="Edge Server WITHOUT Defense (Baseline).")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--global_upload_dir', type=str, required=True)
    parser.add_argument('--zone_name', type=str, required=True)
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--poll_interval', type=int, default=5)
    args = parser.parse_args()

    print(f"--- [BASELINE] Edge Server (No Defense) for Zone '{args.zone_name}' ---")
    print("    - Mode: Naive FedAvg (Simple Averaging)")
    
    # Load Initial Model just for shape info (No validation data needed)
    initial_model_path = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
    if not os.path.exists(initial_model_path):
        print("FATAL: Initial model not found.")
        return

    # We still need input shape to create the dummy model structure
    # Try to read it from the file, or assume standard shape if file missing
    try:
        with open("validation_data_multi/model_input_shape_multi.txt", "r") as f:
            dims = f.read().split(',')
            input_shape = (int(dims[0]), int(dims[1]))
    except:
        print("Warning: Input shape file not found, assuming default.")
        input_shape = (1, 78) # Fallback

    os.makedirs(args.global_upload_dir, exist_ok=True)

    # --- Wait for Client Updates ---
    model_paths = []
    while len(model_paths) < args.num_clients:
        model_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
        print(f"\r   - Waiting... Found {len(model_paths)} / {args.num_clients}", end="")
        time.sleep(args.poll_interval)
    
    print(f"\n--- All {args.num_clients} updates received. Starting Aggregation. ---")

    collected_updates = []

    # --- Naive Collection (Accept Everything) ---
    for path in model_paths:
        try:
            sparse_update_data = np.load(path)
            # Fix loading order
            sorted_files = sorted(sparse_update_data.files, key=lambda x: int(x.replace('arr_', '')) if x.startswith('arr_') else x)
            sparse_update = [sparse_update_data[arr] for arr in sorted_files]
            
            collected_updates.append(sparse_update)
            print(f"   - [Accepted] {os.path.basename(path)} (No Validation Performed)")
            
        except Exception as e:
            print(f"   - Error loading {path}: {e}")

    if not collected_updates:
        print("No updates collected.")
        return

    # --- Aggregation: Simple Mean ---
    print(f"\n--- Performing Simple Average on {len(collected_updates)} models ---")
    
    num_layers = len(collected_updates[0])
    zone_aggregated_update = []
    
    for i in range(num_layers):
        # Stack updates for this layer
        layer_stack = np.array([u[i] for u in collected_updates])
        # Compute Mean (Vulnerable to outliers/poisoning)
        mean_layer = np.mean(layer_stack, axis=0) 
        zone_aggregated_update.append(mean_layer)

    # Save
    zone_output_filename = f"zone_{args.zone_name}.sparse.npz"
    zone_output_path = os.path.join(args.global_upload_dir, zone_output_filename)
    np.savez_compressed(zone_output_path, *zone_aggregated_update)
    
    print(f"   - Zone model uploaded to: {zone_output_path}")
    
    # Signal Done
    os.makedirs('edge_server_signals_multi', exist_ok=True)
    with open(f"edge_server_signals_multi/{args.zone_name}.done", 'w') as f:
        f.write('done')

if __name__ == '__main__':
    main()