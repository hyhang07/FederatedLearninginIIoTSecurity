# GlobalServer.py (Final Version for TON_IoT - Memory Optimized)

# --- [CRITICAL FIX] Prevent "double free" / OOM crashes ---
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import time
import argparse
import traceback
import gc  # Garbage collection library to free memory
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Force CPU for Stability ---
tf.config.set_visible_devices([], 'GPU')
print("--- Global Server Config: GPU Disabled (Forced CPU-only mode for stability) ---")

# --- Import from custom scripts ---
from Client import create_tcn_gru_model
from DataUtils_Client import load_and_prepare_data_for_dl

# --- Configuration for TON_IoT Dataset ---
NUM_CLASSES = 9
EXPECTED_SIGNALS = 9

def main():
    parser = argparse.ArgumentParser(description="Global Server (Stacking Ensemble for TON_IoT)")
    parser.add_argument('--input_dir', type=str, default='DL_Zone_Uploads', help="Directory containing zone model updates")
    parser.add_argument('--output_dir', type=str, default='DL_Global_Deployment_Final', help="Directory to save final global model")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    meta_train_path = "Dataset/TestSet/meta_train_dataset.csv"
    final_test_path = "Dataset/TestSet/final_global_test_set.csv"
    initial_model_path = "DL_Global_Deployment_Initial/initial_global_model.weights.h5"
    signal_dir = 'edge_server_signals'
    report_dir = "DL_Final_Report"

    print(f"--- Global Server (Stacking Ensemble) Starting ---")
    
    # 1. Wait for all Edge Servers to complete
    print(f"Waiting for {EXPECTED_SIGNALS} edge server signals in '{signal_dir}'...")
    while True:
        signals = glob.glob(os.path.join(signal_dir, '*.done'))
        if len(signals) >= EXPECTED_SIGNALS:
            print(f"\n   - All {len(signals)} signals confirmed.")
            break
        print(f"\r   - Waiting... Found {len(signals)} / {EXPECTED_SIGNALS}", end="")
        time.sleep(2)
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    base_models_update_paths = sorted(glob.glob(os.path.join(input_dir, '*.sparse.npz')))
    if not base_models_update_paths:
        print(f"FATAL: No approved zone models found in '{input_dir}'. Aborting.")
        return
    print(f"   - Found {len(base_models_update_paths)} approved base models (Zone Updates).")

    # =========================================================================
    # [Memory Optimization] Stage 2: Load datasets into memory early
    # =========================================================================
    print(f"\nLoading datasets into memory...")
    try:
        X_train_meta, X_val_meta, y_train_meta, y_val_meta, _, _, _ = load_and_prepare_data_for_dl(meta_train_path)
        X_meta_full = np.concatenate((X_train_meta, X_val_meta), axis=0)
        y_meta_full = np.concatenate((y_train_meta, y_val_meta), axis=0)
        input_shape = (X_meta_full.shape[1], X_meta_full.shape[2])
        
        _, X_test_final, _, y_test_final, le, _, _ = load_and_prepare_data_for_dl(final_test_path)
        
        print(f"   - Meta-training shape: {X_meta_full.shape}")
        print(f"   - Final test shape: {X_test_final.shape}")
    except Exception:
        print(f"FATAL: Could not load datasets.")
        traceback.print_exc()
        return

    # Load Initial Model Weights
    try:
        temp_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
        temp_model.load_weights(initial_model_path)
        initial_weights = temp_model.get_weights()
        del temp_model 
        tf.keras.backend.clear_session()
    except Exception:
        print(f"FATAL: Could not load initial model weights from {initial_model_path}")
        return

    # =========================================================================
    # [Memory Optimization] Stage 3: Sequential Model Reconstruction & Prediction
    # =========================================================================
    print(f"\nReconstructing models and generating meta-features sequentially...")
    
    meta_features_train_list = []
    meta_features_test_list = []

    for path in base_models_update_paths:
        try:
            print(f"   -> Processing {os.path.basename(path)}...")
            sparse_update = np.load(path)
            sorted_files = sorted(sparse_update.files, key=lambda x: int(x.split('_')[1]))
            delta = [sparse_update[arr] for arr in sorted_files]
            
            if len(delta) != len(initial_weights): 
                continue
                
            weights = [(i + d) for i, d in zip(initial_weights, delta)]
            
            model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
            model.set_weights(weights)
            
            pred_train = model.predict(X_meta_full, batch_size=128, verbose=0)
            pred_test = model.predict(X_test_final, batch_size=128, verbose=0)
            
            meta_features_train_list.append(pred_train)
            meta_features_test_list.append(pred_test)
            
            # [CRITICAL] Free memory
            del model, weights, delta, sparse_update
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception:
            print(f"\nERROR: Could not process {os.path.basename(path)}.")

    if not meta_features_train_list:
        print("FATAL: Failed to process any base models. Aborting.")
        return

    stacked_features_train = np.hstack(meta_features_train_list)
    stacked_features_test = np.hstack(meta_features_test_list)

    del X_meta_full, X_test_final
    gc.collect()

    # =========================================================================
    # Stage 4: Train Meta-Learner (LightGBM)
    # =========================================================================
    print("\nTraining LightGBM meta-learner...")

    X_lgbm_train, X_lgbm_val, y_lgbm_train, y_lgbm_val = train_test_split(
        stacked_features_train, y_meta_full, test_size=0.2, random_state=42, stratify=y_meta_full)

    # Pure objective meta-learner setup
    meta_model = LGBMClassifier(
        n_estimators=1000,        
        learning_rate=0.01,       
        max_depth=8,              
        num_leaves=64,            
        min_child_samples=10,   
        class_weight='balanced',  
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    )
    
    meta_model.fit(
        X_lgbm_train, y_lgbm_train,
        eval_set=[(X_lgbm_val, y_lgbm_val)],
        eval_metric='logloss',
        callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
    
    joblib.dump(meta_model, os.path.join(output_dir, "meta_learner.joblib"))
    print(f"   - Meta-learner saved.")

    # =========================================================================
    # Stage 5: Final Unbiased Evaluation
    # =========================================================================
    print("\n" + "="*50)
    print("      STARTING FINAL, UNBIASED EVALUATION      ")
    print("="*50)
    
    y_final_pred = meta_model.predict(stacked_features_test)
    accuracy = accuracy_score(y_test_final, y_final_pred)
    class_names = le.classes_
    
    print(f"\n   - Final Accuracy on Global Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0))

    # 6. Save Reports
    report_dict = classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(report_dir, 'final_classification_report.csv'))
    
    cm = confusion_matrix(y_test_final, y_final_pred, labels=le.transform(class_names))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Final Stacked Global Model')
    plt.savefig(os.path.join(report_dir, 'final_confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    print("\n--- Stacking and Evaluation Complete! ---")

if __name__ == "__main__":
    main()