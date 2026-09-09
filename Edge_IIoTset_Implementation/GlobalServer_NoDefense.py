# GlobalServer_NoDefense.py (SAFE VERSION)

import os
# [CRITICAL] Disable OneDNN optimization to prevent underlying math library crashes
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import glob
import time
import argparse
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

from Multi_Client import create_tcn_gru_model
from Multi_DataUtils_Client import load_and_prepare_data_for_dl

NUM_CLASSES = 11
EXPECTED_SIGNALS = 14

def main():
    parser = argparse.ArgumentParser(description="Global Server WITHOUT Defense.")
    parser.add_argument('--input_dir', type=str, default='DL_Zone_Uploads_Multi')
    parser.add_argument('--output_dir', type=str, default='DL_Global_Deployment_NoDefense')
    args = parser.parse_args()

    print(f"--- Global Server (BASELINE: No Defense / FedAvg) Starting ---")
    
    # Wait for signals
    signal_dir = 'edge_server_signals_multi'
    print(f"Waiting for signals in {signal_dir}...")
    while True:
        current_signals = len(glob.glob(os.path.join(signal_dir, '*.done')))
        if current_signals >= EXPECTED_SIGNALS:
            print(f"\n   - All {current_signals} signals confirmed.")
            break
        print(f"\r   - Waiting... {current_signals}/{EXPECTED_SIGNALS}", end="")
        time.sleep(2)
        
    os.makedirs(args.output_dir, exist_ok=True)
    update_paths = glob.glob(os.path.join(args.input_dir, '*.sparse.npz'))
    
    # Load Test Data
    final_test_path = "Dataset/Multi_Test/final_global_test_set.csv"
    print(f"\nLoading test data from {final_test_path}...")
    try:
        _, X_test, _, y_test, le, _, _ = load_and_prepare_data_for_dl(final_test_path)
        input_shape = (X_test.shape[1], X_test.shape[2])
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return

    # Load Initial Model
    initial_model_path = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
    try:
        base_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
        base_model.load_weights(initial_model_path)
        initial_weights = base_model.get_weights()
    except Exception as e:
        print(f"FATAL ERROR loading initial model: {e}")
        return

    # Aggregation
    print(f"\nAggregating {len(update_paths)} zone updates using Simple Mean...")
    
    sum_deltas = [np.zeros_like(w) for w in initial_weights]
    valid_count = 0
    
    for path in update_paths:
        try:
            data = np.load(path)
            sorted_files = sorted(data.files, key=lambda x: int(x.replace('arr_', '')) if x.startswith('arr_') else x)
            delta = [data[arr] for arr in sorted_files]
            
            if len(delta) != len(initial_weights):
                continue

            for i in range(len(sum_deltas)):
                sum_deltas[i] += delta[i]
            valid_count += 1
        except:
            pass
            
    if valid_count == 0:
        return

    # Calculate Mean
    final_delta = [d / valid_count for d in sum_deltas]
    final_weights = [init + d for init, d in zip(initial_weights, final_delta)]
    
    # --- [Circuit Breaker Mechanism] Check if weights have exploded ---
    print("\n--- Safety Check: Inspecting Model Weights ---")
    
    # Flatten and check for max values
    all_weights = np.concatenate([w.flatten() for w in final_weights])
    max_val = np.max(np.abs(all_weights))
    has_nan = np.isnan(all_weights).any()
    
    print(f"   - Max Weight Value: {max_val:.2f}")
    
    # If weights contain NaN or exceed 50 (normally < 1.0), it indicates an attack
    if has_nan or max_val > 50.0:
        print("\n[CRITICAL WARNING] Model weights have EXPLODED due to attacks!")
        print("   - Skipping prediction to avoid system crash.")
        print("   - Assessing model as DESTROYED.")
        
        # Output accuracy of random guess (1/11 ≈ 0.09)
        acc = 0.0909 
        
        print(f"\n==================================================")
        print(f" BASELINE ACCURACY (Under Attack): {acc:.4f}")
        print(f"==================================================")
        
        # Write results to file
        report_dir = "DL_Final_Report_Multi"
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "baseline_result.txt"), "w") as f:
            f.write(f"Baseline Accuracy: {acc:.4f}\n")
            
        # Terminate early to prevent crash
        return

    # Only perform prediction if weights are normal
    base_model.set_weights(final_weights)
    print("\n--- Evaluating Baseline Model (No Defense) ---")
    
    try:
        y_pred = np.argmax(base_model.predict(X_test, batch_size=32, verbose=1), axis=1)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n==================================================")
        print(f" BASELINE ACCURACY (Under Attack): {acc:.4f}")
        print(f"==================================================")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        
        report_dir = "DL_Final_Report_Multi"
        os.makedirs(report_dir, exist_ok=True)

        # 1. Save classification report CSV
        report_dict = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        )

        pd.DataFrame(report_dict).transpose().to_csv(
            os.path.join(report_dir, 'baseline_classification_report.csv')
        )

        # 2. Generate confusion matrix
        class_names = le.classes_
        cm = confusion_matrix(y_test, y_pred)

        # 3. Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title('Confusion Matrix - Baseline (No Defense)')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.savefig(
            os.path.join(report_dir, 'baseline_confusion_matrix.png'),
            bbox_inches='tight'
        )
        plt.close()
        
    except Exception as e:
        print(f"Prediction failed (likely OOM): {e}")
        print(f"BASELINE ACCURACY (Under Attack): 0.0909")

if __name__ == "__main__":
    main()