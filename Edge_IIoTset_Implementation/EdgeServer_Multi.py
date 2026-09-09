# EdgeServer_Multi.py (FINAL ROBUST VERSION)
import os
import glob
import argparse
import time
import numpy as np
import shutil
import re
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity 

# Import model creation function from Client script
from Multi_Client import create_tcn_gru_model

# --- Global Constants ---
NUM_CLASSES = 11
# Hard thresholds still exist as a first line of defense
ACCURACY_THRESHOLD = 0.50 
NORM_THRESHOLD = 100.0

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    parser = argparse.ArgumentParser(description="[UPGRADED] Edge Server with FLTrust Defense.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory to find incoming sparse updates from clients.")
    parser.add_argument('--global_upload_dir', type=str, required=True, help="Directory to upload the single, aggregated zone model to.")
    parser.add_argument('--zone_name', type=str, required=True, help="The name of the zone this server is managing.")
    parser.add_argument('--num_clients', type=int, default=3, help="The number of clients this server expects updates from.")
    parser.add_argument('--poll_interval', type=int, default=10, help="How often (in seconds) to check for new updates.")
    args = parser.parse_args()

    print(f"--- [UPGRADED] Edge Server for Zone '{args.zone_name}' is in STANDBY mode ---")
    print("    - Mode: FLTrust (Root-of-Trust based Robust Aggregation)")
    
    # --- 1. Load Initial Model & Validation Data ---
    initial_model_path = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
    if not os.path.exists(initial_model_path):
        print(f"FATAL: Initial multi-class global model not found at {initial_model_path}")
        return
        
    zone_validation_dir = os.path.join("validation_data_multi", args.zone_name)
    base_validation_dir = "validation_data_multi"
    try:
        X_val = np.load(os.path.join(zone_validation_dir, "X_val.npy"))
        y_val = np.load(os.path.join(zone_validation_dir, "y_val.npy"))
        with open(os.path.join(base_validation_dir, "model_input_shape_multi.txt"), "r") as f:
            dims = f.read().split(',')
            input_shape = (int(dims[0]), int(dims[1])) 
        print(f"   - Zone-specific validation set ({args.zone_name}) loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Zone-specific validation set not found in '{zone_validation_dir}'.")
        return
        
    # Load base weights into memory
    base_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    base_model.load_weights(initial_model_path)
    initial_weights = base_model.get_weights()
    
    os.makedirs(args.global_upload_dir, exist_ok=True)

    # --- 2. Wait for Client Updates ---
    model_paths = []
    while len(model_paths) < args.num_clients:
        model_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
        print(f"\r   - Waiting... Found {len(model_paths)} / {args.num_clients} sparse updates.", end="")
        time.sleep(args.poll_interval)
    
    print(f"\n--- All {args.num_clients} updates received. Starting FLTrust validation process. ---")
    
    # Logging Setup
    report_dir = "DL_Final_Report_Multi"
    os.makedirs(report_dir, exist_ok=True)
    gatekeeper_log_path = os.path.join(report_dir, "gatekeeper_log.csv")
    if not os.path.exists(gatekeeper_log_path):
        with open(gatekeeper_log_path, "w") as f: 
            f.write("ClientID,Zone,Accuracy,UpdateNorm,Status,RejectReason,TrustScore\n")

    # =================================================================================
    # [FLTrust CORE LOGIC START]
    # =================================================================================
    
    # --- Step A: Compute the "Golden Standard" Update (Server Update) ---
    print("\n--- [FLTrust] Computing 'Golden Standard' update vector from Root-of-Trust data... ---")
    
    fltrust_batch_size = 256
    actual_batch_size = min(len(X_val), fltrust_batch_size)
    X_val_batch = X_val[:actual_batch_size]
    y_val_batch = y_val[:actual_batch_size]
    
    # Create a temporary model
    server_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    server_model.set_weights(initial_weights)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        logits = server_model(X_val_batch, training=True) 
        loss = loss_fn(y_val_batch, logits)
    
    # Calculate gradients for TRAINABLE variables
    server_grads = tape.gradient(loss, server_model.trainable_variables)
    
    # --- [ROBUST FIX] Map gradients using Object Identity (id) ---
    # This avoids string name issues entirely
    grad_map = {id(v): g for v, g in zip(server_model.trainable_variables, server_grads)}
    
    full_server_update_list = []
    
    # Iterate through model.weights which includes BOTH trainable and non-trainable
    for w in server_model.weights:
        w_id = id(w)
        if w_id in grad_map and grad_map[w_id] is not None:
            # It is trainable: use negative gradient
            full_server_update_list.append(-1.0 * grad_map[w_id].numpy().flatten())
        else:
            # It is non-trainable (e.g., BN moving mean): use zeros
            full_server_update_list.append(np.zeros(w.shape).flatten())
            
    server_update_vector = np.concatenate(full_server_update_list)
    # --- [FIX END] ---
    
    print(f"   - Server reference vector computed. Dimension: {server_update_vector.shape}")

    # =================================================================================

    approved_updates_in_memory = []
    trust_scores = []

    # --- Step B: Validate Clients using FLTrust (Cosine Similarity) ---
    for path in model_paths:
        client_name = os.path.basename(path)
        client_id_match = re.search(r'client_(\d+)', client_name)
        client_id = client_id_match.group(1) if client_id_match else 'Unknown'

        try:
            # Load Client Update
            sparse_update_data = np.load(path)
            sparse_update = [sparse_update_data[arr] for arr in sparse_update_data.files]
            
            # --- 1. Reconstruct for logging and hard-threshold checks ---
            reconstructed_weights = [(initial + delta) for initial, delta in zip(initial_weights, sparse_update)]
            model_instance = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
            model_instance.set_weights(reconstructed_weights)
            
            # Basic Performance Check
            y_pred_probs = model_instance.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            accuracy = accuracy_score(y_val, y_pred)
            update_norm = np.linalg.norm(np.concatenate([p.flatten() for p in sparse_update]))

            print(f"\n   - Validating {client_name}...")
            print(f"     - Accuracy: {accuracy:.4f}")
            print(f"     - Update Norm: {update_norm:.4f}")

            # --- 2. FLTrust Similarity Calculation ---
            client_update_vector = np.concatenate([layer.flatten() for layer in sparse_update])
            
            # --- [SAFETY NET] Force Alignment if Dimensions Differ ---
            c_len = len(client_update_vector)
            s_len = len(server_update_vector)
            
            final_server_vec = server_update_vector
            final_client_vec = client_update_vector

            if c_len != s_len:
                print(f"     - WARNING: Dimension mismatch caught! Client: {c_len}, Server: {s_len}")
                print(f"     - Performing emergency truncation/padding to continue...")
                target_len = min(c_len, s_len)
                final_client_vec = client_update_vector[:target_len]
                final_server_vec = server_update_vector[:target_len]
            
            # Calculate Cosine Similarity
            cos_sim = cosine_similarity(
                final_server_vec.reshape(1, -1), 
                final_client_vec.reshape(1, -1)
            )[0][0]
            
            # FLTrust Core Rule: ReLU(Similarity)
            raw_trust_score = max(0, cos_sim)
            
            print(f"     - [FLTrust] Cosine Similarity: {cos_sim:.4f}")
            print(f"     - [FLTrust] Trust Score:       {raw_trust_score:.4f}")

            status = "Rejected"
            reject_reason = "None"
            
            # --- Decision Logic ---
            if accuracy >= ACCURACY_THRESHOLD and update_norm < NORM_THRESHOLD:
                approved_updates_in_memory.append(sparse_update)
                trust_scores.append(raw_trust_score)
                status = "Approved"
                print(f"     - PASSED. Added to pool with Trust Score {raw_trust_score:.4f}")
            
            elif accuracy < ACCURACY_THRESHOLD:
                print(f"     - FAILED: Low Accuracy ({accuracy:.4f} < {ACCURACY_THRESHOLD})")
                reject_reason = "Accuracy"
            
            else: 
                print(f"     - FAILED: High Norm ({update_norm:.4f} >= {NORM_THRESHOLD})")
                reject_reason = "Norm"

            with open(gatekeeper_log_path, "a") as f:
                f.write(f"{client_id},{args.zone_name},{accuracy:.4f},{update_norm:.4f},{status},{reject_reason},{raw_trust_score:.4f}\n")
                
        except Exception as e:
            print(f"\n   - WARNING: Could not process {client_name}. Skipping. Error: {e}")
            import traceback
            traceback.print_exc()
            with open(gatekeeper_log_path, "a") as f:
                f.write(f"{client_id},{args.zone_name},0,0,Rejected,Error,0\n")
    
    approved_count = len(approved_updates_in_memory)

    # --- Step C: FLTrust Aggregation ---
    if not approved_updates_in_memory:
        print("\n--- No client updates were approved in this zone. No aggregation performed. ---")
    else:
        print(f"\n--- [LAYER 1 AGGREGATION] Performing FLTrust Weighted Averaging... ---")
        
        total_trust = sum(trust_scores)
        
        # Normalization
        if total_trust <= 1e-9:
            print("   - WARNING: Total Trust Score is effectively 0 (All updates were antagonistic).")
            print("   - Fallback: Using Uniform Averaging for approved clients.")
            normalized_weights = np.array([1.0 / approved_count] * approved_count)
        else:
            # Weight = TrustScore_i / Sum(TrustScores)
            normalized_weights = np.array([ts / total_trust for ts in trust_scores])
            
        print(f"   - Final Aggregation Weights: {np.round(normalized_weights, 4)}")
        
        # Perform Weighted Aggregation
        num_layers = len(approved_updates_in_memory[0])
        zone_aggregated_update = []
        
        for i in range(num_layers):
            layer_updates_stack = np.array([update[i] for update in approved_updates_in_memory])
            
            # Reshape weights for broadcasting
            reshaped_weights = normalized_weights.reshape(-1, *([1] * (layer_updates_stack.ndim - 1)))
            
            # Weighted Sum
            weighted_sum_layer = np.sum(layer_updates_stack * reshaped_weights, axis=0)
            zone_aggregated_update.append(weighted_sum_layer)

        print("   - FLTrust aggregation complete.")

        # Save and upload Zone Update
        zone_output_filename = f"zone_{args.zone_name}.sparse.npz"
        zone_output_path = os.path.join(args.global_upload_dir, zone_output_filename)
        np.savez_compressed(zone_output_path, *zone_aggregated_update)
    
        print(f"   - Zone-level aggregated update saved to: {zone_output_path}")

    print(f"\n--- Validation & Aggregation Complete! Uploaded 1 zone model. ---")
    
    # --- 5. Signal Completion ---
    os.makedirs('edge_server_signals_multi', exist_ok=True)
    done_filepath = f"edge_server_signals_multi/{args.zone_name}.done"
    with open(done_filepath, 'w') as f:
        f.write('done')
        
    print(f"--- Edge Server for Zone '{args.zone_name}' has finished its task. ---")

if __name__ == '__main__':
    main()