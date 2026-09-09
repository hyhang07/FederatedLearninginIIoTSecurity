import pandas as pd
import numpy as np
import argparse
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Import the necessary functions from your project ---
from Multi_Client import create_tcn_gru_model
from Multi_DataUtils_Client import load_and_prepare_data_for_dl

# --- Configuration ---
NUM_CLASSES = 11
BASE_MODELS_DIR = "DL_Global_Uploads_Multi"
INITIAL_MODEL_PATH = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
FINAL_META_MODEL_PATH = "DL_Global_Deployment_Stacked_LGBM_Multi/stacked_lgbm_meta_model.joblib"
TEST_DATASET_PATH = "Dataset/Multi_Test/global_test_dataset_multi.csv"
REPORT_DIR = "DL_Final_Report_Multi"

def main():
    """
    This script evaluates the final, trained Stacking Ensemble (LGBM meta-model)
    on the comprehensive global test set.
    """
    print("="*60)
    print("   STARTING FINAL GLOBAL STACKED MODEL EVALUATION   ")
    print("="*60)

    # --- Step 1: Verify all necessary files and directories exist ---
    print("\n--- Verifying required assets ---")
    required_assets = [BASE_MODELS_DIR, INITIAL_MODEL_PATH, FINAL_META_MODEL_PATH, TEST_DATASET_PATH]
    for asset in required_assets:
        if not os.path.exists(asset):
            print(f"FATAL ERROR: Required asset not found: {asset}")
            print("Please ensure you have run the full training pipeline first.")
            return
    print("   - All assets found successfully.")

    # --- Step 2: Load the global test data ---
    print(f"\n--- Loading and preparing global test data from '{os.path.basename(TEST_DATASET_PATH)}' ---")
    try:
        # We only need the final test portion and the label encoder
        _, X_test_final, _, y_test_final, le, _, _ = load_and_prepare_data_for_dl(TEST_DATASET_PATH)
        print(f"   - Test set loaded. Shape: {X_test_final.shape}")
    except Exception:
        print("FATAL ERROR: Could not load the test dataset.")
        traceback.print_exc()
        return

    # --- Step 3: Reconstruct all approved base models ---
    base_models_update_paths = sorted(glob.glob(os.path.join(BASE_MODELS_DIR, '*.sparse.npz')))
    if not base_models_update_paths:
        print(f"FATAL ERROR: No approved base models found in '{BASE_MODELS_DIR}'. Cannot evaluate.")
        return
        
    print(f"\n--- Reconstructing {len(base_models_update_paths)} base models ---")
    input_shape = (X_test_final.shape[1], X_test_final.shape[2])
    base_model_template = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    base_model_template.load_weights(INITIAL_MODEL_PATH)
    initial_weights = base_model_template.get_weights()
    base_models = []
    
    for path in base_models_update_paths:
        try:
            sparse_update_data = np.load(path)
            sparse_update = [sparse_update_data[arr] for arr in sparse_update_data.files]
            reconstructed_weights = [(initial + delta) for initial, delta in zip(initial_weights, sparse_update)]
            
            model_instance = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
            model_instance.set_weights(reconstructed_weights)
            base_models.append(model_instance)
        except Exception:
            print(f"   - WARNING: Could not reconstruct model from {os.path.basename(path)}. Skipping.")
            traceback.print_exc()
    
    if not base_models:
        print("FATAL ERROR: Failed to reconstruct any base models. Aborting evaluation.")
        return
    print(f"   - Successfully reconstructed {len(base_models)} base models.")

    # --- Step 4: Load the trained meta-model (LGBM) ---
    print(f"\n--- Loading the final meta-model from '{os.path.basename(FINAL_META_MODEL_PATH)}' ---")
    try:
        meta_model = joblib.load(FINAL_META_MODEL_PATH)
        print("   - Meta-model loaded successfully.")
    except Exception:
        print("FATAL ERROR: Could not load the meta-model.")
        traceback.print_exc()
        return

    # --- Step 5: Generate meta-features from the test set ---
    print("\n--- Generating meta-features for the final evaluation ---")
    meta_features_test = []
    for i, model in enumerate(base_models):
        print(f"   - Getting predictions from base model {i+1}/{len(base_models)}...")
        predictions = model.predict(X_test_final, batch_size=1024, verbose=0)
        meta_features_test.append(predictions)
    
    # Horizontally stack the predictions to create the feature set for the meta-model
    stacked_features_test = np.hstack(meta_features_test)
    print(f"   - Meta-feature test set created with shape: {stacked_features_test.shape}")

    # --- Step 6: Make final predictions and evaluate ---
    print("\n--- Making final predictions on the test set ---")
    y_final_pred = meta_model.predict(stacked_features_test)
    
    print("\n\n" + "="*60)
    print("      FINAL GLOBAL MODEL PERFORMANCE REPORT      ")
    print("="*60)
    
    accuracy = accuracy_score(y_test_final, y_final_pred)
    print(f"\n   - Overall Accuracy on Global Test Set: {accuracy:.4f}\n")
    
    class_names = le.classes_
    print("--- Classification Report ---")
    report_str = classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0)
    print(report_str)

    # --- Step 7: Save the final report and confusion matrix ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_dict = classification_report(y_test_final, y_final_pred, target_names=class_names, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path_csv = os.path.join(REPORT_DIR, 'final_evaluation_report.csv')
    report_df.to_csv(report_path_csv)
    print(f"\n   - Detailed classification report saved to: {report_path_csv}")

    cm = confusion_matrix(y_test_final, y_final_pred, labels=le.transform(class_names))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Final Stacked Global Model', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plot_path = os.path.join(REPORT_DIR, 'final_evaluation_confusion_matrix.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"   - Confusion matrix saved to: {plot_path}")

    print("\n" + "="*60)
    print("         EVALUATION COMPLETE         ")
    print("="*60)

if __name__ == "__main__":
    main()