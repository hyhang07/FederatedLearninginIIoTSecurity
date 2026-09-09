# EdgeServer.py (Final Secure Version - With Zone Quarantine)
import os
import glob
import argparse
import time
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# --- Utility Imports ---
from Client import create_tcn_gru_model
from DataUtils_Client import load_and_prepare_data_for_dl

# --- Configuration ---
NUM_CLASSES = 9
ACCURACY_THRESHOLD = 0.50 
NORM_THRESHOLD = 100.0

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Setup Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="HFL Edge Server with FLTrust Defense.")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--global_upload_dir', type=str, required=True)
    parser.add_argument('--zone_name', type=str, required=True)
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--poll_interval', type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" [EDGE SERVER] Zone: {args.zone_name} | Waiting for {args.num_clients} updates...")
    print(f"{'='*60}")
    
    # 1. Load Initial Model & Validation Data
    initial_model_path = "DL_Global_Deployment_Initial/initial_global_model.weights.h5"
    validation_dir = os.path.join("Dataset", "ValidationSet", args.zone_name)
    possible_files = glob.glob(os.path.join(validation_dir, "*.csv"))
    
    if not possible_files:
        print(f"[FATAL ERROR] No validation CSV found in {validation_dir}")
        return
    
    validation_data_path = possible_files[0]
    print(f"   - Using validation set: {validation_data_path}")
    
    # Use DataUtils to get shape and validation data
    X_val, _, y_val, _, le, scaler, n_features = load_and_prepare_data_for_dl(validation_data_path)
    input_shape = (X_val.shape[1], X_val.shape[2])

    base_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    base_model.load_weights(initial_model_path)
    initial_weights = base_model.get_weights()
    
    os.makedirs(args.global_upload_dir, exist_ok=True)

    # 2. Wait for Client Updates
    client_update_paths =[]
    while len(client_update_paths) < args.num_clients:
        client_update_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
        time.sleep(args.poll_interval)
    
    print(f"\n--- All {args.num_clients} updates received. Starting Two-Stage Verification. ---")
    
    # Logging Setup
    os.makedirs("DL_Final_Report", exist_ok=True)
    gatekeeper_log_path = "DL_Final_Report/gatekeeper_log.csv"
    if not os.path.exists(gatekeeper_log_path):
        with open(gatekeeper_log_path, "w") as f: 
            f.write("ClientID,Zone,Accuracy,UpdateNorm,Status,RejectReason,TrustScore\n")

    # --- FLTrust: Compute Server "Golden Standard" Update ---
    print("\n---[FLTrust] Computing 'Golden Standard' update vector... ---")
    
    server_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    server_model.set_weights(initial_weights)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        logits = server_model(X_val[:256], training=True) 
        loss = loss_fn(y_val[:256], logits)
    
    server_grads = tape.gradient(loss, server_model.trainable_variables)
    
    # Map gradients to all weights (trainable and non-trainable) for dimension alignment
    grad_map = {id(v): g for v, g in zip(server_model.trainable_variables, server_grads)}
    full_server_update_list =[]
    for w in server_model.weights:
        w_id = id(w)
        if w_id in grad_map and grad_map[w_id] is not None:
            full_server_update_list.append(-1.0 * grad_map[w_id].numpy().flatten())
        else:
            full_server_update_list.append(np.zeros(w.shape).flatten())
            
    server_update_vector = np.concatenate(full_server_update_list)

    # --- Two-Stage Verification Loop ---
    approved_updates = []
    trust_scores =[]

    for path in client_update_paths:
        client_id = re.search(r'client_(\d+)', os.path.basename(path)).group(1)
        
        try:
            # Load & Reconstruct
            data = np.load(path)
            sparse_update = [data[arr] for arr in sorted(data.files, key=lambda x: int(x.split('_')[1]))]
            reconstructed_weights = [(i + d) for i, d in zip(initial_weights, sparse_update)]
            
            model_instance = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
            model_instance.set_weights(reconstructed_weights)
            
            # --- Stage 1: Hard-Threshold Check ---
            y_pred = np.argmax(model_instance.predict(X_val, verbose=0), axis=1)
            accuracy = accuracy_score(y_val, y_pred)
            client_update_vector = np.concatenate([d.flatten() for d in sparse_update])
            update_norm = np.linalg.norm(client_update_vector)

            status, reason = "Rejected", "None"
            trust_score = 0.0

            if accuracy < ACCURACY_THRESHOLD:
                reason = "Low Accuracy"
            elif update_norm > NORM_THRESHOLD:
                reason = "High Norm"
            else:
                # --- Stage 2: FLTrust Check ---
                cos_sim = cosine_similarity(server_update_vector.reshape(1, -1), client_update_vector.reshape(1, -1))[0][0]
                trust_score = max(0, cos_sim)
                
                status = "Approved"
                approved_updates.append(sparse_update)
                trust_scores.append(trust_score)

            print(f"\n   - Validating Client {client_id}...")
            print(f"     - Accuracy: {accuracy:.4f} | Norm: {update_norm:.4f}")
            if status == "Approved":
                print(f"     - [PASSED] Trust Score: {trust_score:.4f}")
            else:
                print(f"     - [FAILED] Reason: {reason}")

            # Log results
            with open(gatekeeper_log_path, "a") as f:
                f.write(f"{client_id},{args.zone_name},{accuracy:.4f},{update_norm:.4f},{status},{reason},{trust_score:.4f}\n")

        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # --- Aggregation & Upload (With Quarantine Logic) ---
    if not approved_updates:
        print("\n--- [SECURITY ALERT] No clients passed the hard thresholds. ---")
        print(f"--- QUARANTINING ZONE: Zone '{args.zone_name}' is compromised. No model will be uploaded. ---")
    else:
        total_trust = sum(trust_scores)
        
        if total_trust <= 1e-9:
            print("\n--- [CRITICAL SECURITY ALERT] All approved clients exhibited malicious gradient directions! ---")
            print(f"--- QUARANTINING ZONE: Zone '{args.zone_name}' is severely compromised. No model will be uploaded. ---")
        else:
            print(f"\n--- Aggregating {len(approved_updates)} trusted updates... ---")
            weights = [ts / total_trust for ts in trust_scores]
            
            zone_aggregated_update = [np.sum([updates[i] * w for updates, w in zip(approved_updates, weights)], axis=0) for i in range(len(initial_weights))]
            
            upload_path = os.path.join(args.global_upload_dir, f"zone_{args.zone_name}.sparse.npz")
            np.savez_compressed(upload_path, *zone_aggregated_update)
            print(f"   - Secure Zone model uploaded to: {upload_path}")

    # --- Signal Completion ---
    # We must ALWAYS send the .done signal, even if the zone is quarantined. 
    # Otherwise, the Global Server will wait forever.
    os.makedirs("edge_server_signals", exist_ok=True)
    with open(f"edge_server_signals/{args.zone_name}.done", 'w') as f:
        f.write('done')
    print(f"\n[Finished] Edge Server for Zone '{args.zone_name}' complete.")

if __name__ == '__main__':
    main()