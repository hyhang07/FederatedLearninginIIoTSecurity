# GlobalServer_Distiller_Multi.py (FINAL FIXED VERSION - With OneDNN Crash Fix)

# ==========================================
# [CRITICAL FIX] Must be placed BEFORE importing tensorflow/numpy
# This prevents the "corrupted double-linked list" / "double free" crash
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ==========================================

import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import time
import argparse
import traceback
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- [CRITICAL CONFIG] Force CPU for Global Server ---
# Global Server mainly runs inference on base models and trains LightGBM (CPU based).
# Using GPU here often causes OOM (Out of Memory) because Edge Servers might still hold VRAM.
tf.config.set_visible_devices([], 'GPU')
print("--- Global Server Config: GPU Disabled (Forced CPU-only mode for stability) ---")

from Multi_Client import create_tcn_gru_model
from Multi_DataUtils_Client import load_and_prepare_data_for_dl

# --- Configuration ---
NUM_CLASSES = 11
EXPECTED_SIGNALS = 14

def main():
    parser = argparse.ArgumentParser(description="Global Server (STACKING ENSEMBLE)")
    # --- [FIX 1] Added argument reception for compatibility with Bash scripts ---
    parser.add_argument('--input_dir', type=str, default='DL_Zone_Uploads_Multi', help="Directory containing zone model updates")
    parser.add_argument('--output_dir', type=str, default='DL_Global_Deployment_Stacked_LGBM_Multi', help="Directory to save final global model")
    args = parser.parse_args()

    # Use incoming arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # File path configuration
    meta_train_path = "Dataset/Multi_Test/meta_train_dataset.csv"
    final_test_path = "Dataset/Multi_Test/final_global_test_set.csv"

    print(f"--- Global Server (Stacking Ensemble) Starting ---")
    print(f"   - Input Directory: {input_dir}")
    print(f"   - Output Directory: {output_dir}")
    
    # --- Stage 1: Wait for signals from all Edge Servers ---
    signal_dir = 'edge_server_signals_multi'
    print(f"Waiting for {EXPECTED_SIGNALS} edge server signals in '{signal_dir}'...")
    while True:
        signals = glob.glob(os.path.join(signal_dir, '*.done'))
        # Considered passed as long as enough signals are found
        if len(signals) >= EXPECTED_SIGNALS:
            print(f"\n   - All {len(signals)} signals confirmed.")
            break
        print(f"\r   - Waiting... Found {len(signals)} / {EXPECTED_SIGNALS}", end="")
        time.sleep(2)
        
    print(f"\n--- Starting Stacking Ensemble process. ---")
    os.makedirs(output_dir, exist_ok=True)

    base_models_update_paths = sorted(glob.glob(os.path.join(input_dir, '*.sparse.npz')))
    if not base_models_update_paths:
        print(f"FATAL: No approved base models found in '{input_dir}'.")
        print("Please check if Edge Servers successfully uploaded their 'zone_*.sparse.npz' files.")
        return
    print(f"   - Found {len(base_models_update_paths)} approved base models (Zone Updates).")

    # --- Stage 2: Load independent "textbook" (meta-training data) ---
    print(f"\nLoading DEDICATED meta-training data from '{os.path.basename(meta_train_path)}'...")
    try:
        X_train_meta, X_val_meta, y_train_meta, y_val_meta, _, _, n_features = load_and_prepare_data_for_dl(meta_train_path)
        X_meta_full = np.concatenate((X_train_meta, X_val_meta), axis=0)
        y_meta_full = np.concatenate((y_train_meta, y_val_meta), axis=0)
        print(f"   - Meta-training data loaded. Shape: {X_meta_full.shape}")
    except Exception:
        print(f"FATAL: Could not load meta-training data.")
        traceback.print_exc()
        return

    # --- Stage 3: Reconstruct all base models ---
    input_shape = (X_meta_full.shape[1], X_meta_full.shape[2])
    print(f"\nReconstructing {len(base_models_update_paths)} base models...")
    initial_model_path = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
    
    try:
        base_model_template = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
        base_model_template.load_weights(initial_model_path)
        initial_weights = base_model_template.get_weights()
    except Exception:
        print(f"FATAL: Could not load initial model weights from {initial_model_path}")
        return

    base_models = []
    
    for path in base_models_update_paths:
        try:
            sparse_update = np.load(path)
            # --- [FIX 2] Ensure parameters are loaded in order ---
            sorted_files = sorted(sparse_update.files, key=lambda x: int(x.replace('arr_', '')) if x.startswith('arr_') else x)
            delta = [sparse_update[arr] for arr in sorted_files]
            
            # Security check: Ensure dimensions match
            if len(delta) != len(initial_weights):
                print(f"   - WARNING: Skipping {os.path.basename(path)} due to layer mismatch.")
                continue
                
            weights = [(i + d) for i, d in zip(initial_weights, delta)]
            model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
            model.set_weights(weights)
            base_models.append(model)
        except Exception:
            print(f"\n   - FATAL ERROR: Could not reconstruct model from {os.path.basename(path)}.")
            traceback.print_exc()

    if not base_models:
        print("FATAL: Failed to reconstruct any base models. Aborting.")
        return
        
    # --- Stage 4: Train meta-learner ---
    print(f"\nGenerating meta-features for training the meta-learner...")
    
    # Prediction process (CPU enforced)
    meta_features_train_list = []
    for i, model in enumerate(base_models):
        # Use larger batch_size to accelerate CPU inference
        pred = model.predict(X_meta_full, batch_size=512, verbose=0)
        meta_features_train_list.append(pred)
        
    stacked_features_train = np.hstack(meta_features_train_list)
    
    print("\nCreating and training the meta-model (LightGBM)...")
    X_lgbm_train, X_lgbm_val, y_lgbm_train, y_lgbm_val = train_test_split(
        stacked_features_train, y_meta_full, test_size=0.2, random_state=42, stratify=y_meta_full
    )

    # verbose=-1 to reduce log interference
    meta_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    meta_model.fit(
        X_lgbm_train, y_lgbm_train,
        eval_set=[(X_lgbm_val, y_lgbm_val)],
        eval_metric='logloss',
        callbacks=[early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    output_path = os.path.join(output_dir, "stacked_lgbm_meta_model.joblib")
    joblib.dump(meta_model, output_path)
    print(f"   - Meta-model saved to: {output_path}")

    # --- Stage 5: Final evaluation ---
    print("\n" + "="*50)
    print("      STARTING FINAL, UNBIASED EVALUATION      ")
    print("="*50)
    print(f"\nLoading UNSEEN final test data from '{os.path.basename(final_test_path)}'...")
    try:
        _, X_test_final, _, y_test_final, le, _, _ = load_and_prepare_data_for_dl(final_test_path)
    except Exception:
        print(f"FATAL: Could not load final test data.")
        traceback.print_exc()
        return

    print(f"--- Evaluating on test set size: {len(X_test_final)} ---")
    
    meta_features_test_list = []
    for model in base_models:
        pred = model.predict(X_test_final, batch_size=512, verbose=0)
        meta_features_test_list.append(pred)
        
    stacked_features_test = np.hstack(meta_features_test_list)
    
    import pandas as pd
    demo_df = pd.DataFrame(stacked_features_test)
    demo_df['GroundTruth_Label'] = y_test_final
    demo_df.to_csv("demo_meta_features.csv", index=False)
    print("✅ 弹药库 demo_meta_features.csv 极速生成完毕，准备用于大屏演示！")

    y_final_pred = meta_model.predict(stacked_features_test)
    accuracy = accuracy_score(y_test_final, y_final_pred)
    class_names = le.classes_
    
    print(f"\n   - Final Accuracy on Global Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0))

    # --- Stage 6: Save report ---
    report_dir = "DL_Final_Report_Multi"
    os.makedirs(report_dir, exist_ok=True)
    report_dict = classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(report_dir, 'classification_report.csv'))
    
    cm = confusion_matrix(y_test_final, y_final_pred, labels=le.transform(class_names))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Final Stacked Global Model')
    plt.savefig(os.path.join(report_dir, 'final_model_confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    print("\n--- Stacking and Evaluation Complete! ---")

if __name__ == "__main__":
    main()