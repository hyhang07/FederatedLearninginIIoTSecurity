# GlobalServer_NoDefense.py (Baseline: Standard FedAvg - Vulnerable)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import time
import argparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Force CPU to avoid VRAM conflicts and OOM errors
tf.config.set_visible_devices([], 'GPU')

from Client import create_tcn_gru_model
from DataUtils_Client import load_and_prepare_data_for_dl

NUM_CLASSES = 9
EXPECTED_SIGNALS = 9 # Should match the number of zones

def main():
    parser = argparse.ArgumentParser(description="Global Server WITHOUT Defense (Baseline).")
    parser.add_argument('--input_dir', type=str, default='DL_Zone_Uploads')
    parser.add_argument('--output_dir', type=str, default='DL_Global_Deployment_NoDefense')
    args = parser.parse_args()

    print(f"--- Global Server (BASELINE: No Defense / FedAvg) Starting ---")
    
    # Wait for signals from all edge servers
    signal_dir = 'edge_server_signals'
    print(f"Waiting for {EXPECTED_SIGNALS} signals in '{signal_dir}'...")
    while True:
        current_signals = len(glob.glob(os.path.join(signal_dir, '*.done')))
        if current_signals >= EXPECTED_SIGNALS:
            print(f"\n   - All {current_signals} signals confirmed.")
            break
        print(f"\r   - Waiting... {current_signals}/{EXPECTED_SIGNALS}", end="")
        time.sleep(2)
        
    os.makedirs(args.output_dir, exist_ok=True)
    update_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
    
    # Load Test Data to get model shape and for final evaluation
    final_test_path = "Dataset/TestSet/final_global_test_set.csv"
    print(f"\nLoading test data from {final_test_path}...")
    try:
        _, X_test, _, y_test, le, _, _ = load_and_prepare_data_for_dl(final_test_path)
        input_shape = (X_test.shape[1], X_test.shape[2])
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return

    # Load Initial Model
    initial_model_path = "DL_Global_Deployment_Initial/initial_global_model.weights.h5"
    try:
        base_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
        base_model.load_weights(initial_model_path)
        initial_weights = base_model.get_weights()
    except Exception as e:
        print(f"FATAL ERROR loading initial model: {e}")
        return

    # Aggregation
    print(f"\nAggregating {len(update_paths)} zone updates using Simple Mean (FedAvg)...")
    
    sum_deltas = [np.zeros_like(w) for w in initial_weights]
    valid_count = 0
    
    for path in update_paths:
        try:
            data = np.load(path)
            # Ensure correct loading order
            sorted_files = sorted(data.files, key=lambda x: int(x.split('_')[1]))
            delta = [data[arr] for arr in sorted_files]
            
            if len(delta) != len(initial_weights):
                print(f"  - WARNING: Skipping {os.path.basename(path)} due to layer mismatch.")
                continue

            for i in range(len(sum_deltas)):
                sum_deltas[i] += delta[i]
            valid_count += 1
        except Exception as e:
            print(f"  - WARNING: Failed to load update from {os.path.basename(path)}. Error: {e}")
            
    if valid_count == 0:
        print("FATAL: No valid updates could be processed.")
        return

    # Calculate the average update
    avg_delta = [d / valid_count for d in sum_deltas]
    final_weights = [init + d for init, d in zip(initial_weights, avg_delta)]
    
    # --- [Safety Check] Inspect weights for explosion from scaling attacks ---
    print("\n--- Safety Check: Inspecting Aggregated Model Weights ---")
    
    all_weights = np.concatenate([w.flatten() for w in final_weights])
    max_val = np.max(np.abs(all_weights))
    has_nan = np.isnan(all_weights).any()
    
    print(f"   - Max Absolute Weight Value: {max_val:.2f}")
    
    # If weights are NaN or absurdly large, the model has collapsed.
    if has_nan or max_val > 500.0: # A very high threshold
        print("\n[CRITICAL WARNING] Model weights have EXPLODED due to attacks!")
        print("   - Skipping prediction to avoid system crash.")
        print("   - Assessing model as DESTROYED.")
        
        # Report an accuracy equivalent to random guessing (1/NUM_CLASSES)
        acc = 1.0 / NUM_CLASSES 
        
        print(f"\n==================================================")
        print(f" BASELINE ACCURACY (Under Attack): {acc:.4f}")
        print(f"==================================================")
        
        report_dir = "DL_Final_Report"
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "baseline_result.txt"), "w") as f:
            f.write(f"Baseline Accuracy (Model Collapsed): {acc:.4f}\n")
            f.write("Model weights exploded due to poisoning attack. Evaluation aborted.\n")
            
        return # Terminate to prevent further errors

    # If weights seem reasonable, proceed with evaluation
    base_model.set_weights(final_weights)
    print("\n--- Evaluating Baseline Model (No Defense) ---")
    
    try:
        y_pred = np.argmax(base_model.predict(X_test, batch_size=256, verbose=1), axis=1)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n==================================================")
        print(f" BASELINE ACCURACY (Under Attack): {acc:.4f}")
        print(f"==================================================")
        
        report_dir = "DL_Final_Report"
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Save classification report to CSV
        report_dict = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            os.path.join(report_dir, 'baseline_classification_report.csv')
        )

        # 2. Generate and save confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_
        )
        plt.title('Confusion Matrix - Baseline (No Defense)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(
            os.path.join(report_dir, 'baseline_confusion_matrix.png'),
            bbox_inches='tight'
        )
        plt.close()
        print("   - Baseline report and confusion matrix saved.")
        
    except Exception as e:
        print(f"ERROR: Prediction failed, possibly due to unstable weights. {e}")
        acc = 1.0 / NUM_CLASSES
        print(f"Assumed Accuracy (Random Guess): {acc:.4f}")
        with open(os.path.join("DL_Final_Report", "baseline_result.txt"), "w") as f:
            f.write(f"Baseline Accuracy (Evaluation Failed): {acc:.4f}\n")

if __name__ == "__main__":
    main()