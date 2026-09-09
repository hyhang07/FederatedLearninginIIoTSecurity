import pandas as pd
import numpy as np
import argparse
import time
import os
import glob
import psutil 
import tensorflow as tf

# --- GPU Memory Growth Configuration (Unchanged) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tcn import TCN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Multi_DataUtils_Client import load_and_prepare_data_for_dl

num_classes = 11  

def create_tcn_gru_model(input_shape, num_classes, learning_rate=0.001):
    """
    Creates a powerful 1D-TCN + GRU model for MULTI-CLASS classification.
    """
    model = Sequential([
        TCN(input_shape=input_shape, nb_filters=128, kernel_size=5, dilations=[1, 2, 4, 8], padding='causal', activation='relu', return_sequences=True),
        BatchNormalization(),
        GRU(128, return_sequences=False),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.4), 
        # The final layer has 'num_classes' neurons and a 'softmax' activation for multi-class probability output
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # The loss function is now 'sparse_categorical_crossentropy' for integer-based multi-class labels
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, client_id, zone_name):
    """
    Saves plots for the client's training & validation accuracy and loss over epochs.
    """
    try:
        # Save to a new directory for this experiment
        output_dir = "DL_Final_Report_Multi/Client_Histories"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(14, 6))
        # Accuracy Subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title(f'Accuracy vs. Epochs (Multi-Class)\n(Client {client_id} - {zone_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        # Loss Subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', marker='o')
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
        plt.title(f'Loss vs. Epochs (Multi-Class)\n(Client {client_id} - {zone_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'client_{client_id}_{zone_name}_history.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"   - Training history plot saved to: {plot_path}")
    except Exception as e:
        print(f"   - WARNING: Could not generate training history plot. Error: {e}")

def function_1_train_and_upload(args):
    print(f"\n===== [Function 1] Starting MULTI-CLASS TRAINING for Client {args.client_id} (Zone: {args.zone_name}) =====")
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    initial_model_path = "DL_Global_Deployment_Initial_Multi/initial_global_model.weights.h5"
    if not os.path.exists(initial_model_path):
        print(f"FATAL: Initial multi-class global model not found at {initial_model_path}")
        return
        
    dataset_path = f"Dataset/Multi/{args.zone_name}/client_{args.client_id}_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"FATAL ERROR: Dataset not found at {dataset_path}")
        return
    X_train, X_test, y_train, y_test, le, scaler, n_features = load_and_prepare_data_for_dl(dataset_path)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"--- Client is building a model for {num_classes} total classes. It saw {len(le.classes_)} classes in its local data: {le.classes_} ---")
    
    local_model = create_tcn_gru_model(input_shape, num_classes=num_classes, learning_rate=args.learning_rate)
    local_model.load_weights(initial_model_path)
    initial_weights = local_model.get_weights()

    print("\n--- Training Local Multi-Class 1D-TCN+GRU Model ---")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    history = local_model.fit(X_train, y_train, epochs=25, batch_size=256, validation_split=0.2, verbose=1, callbacks=callbacks)
    
    plot_training_history(history, args.client_id, args.zone_name)
    
    print("\n--- Evaluating Local Model Performance (before aggregation) ---")
    y_pred_probs_test = local_model.predict(X_test)
    y_pred_test = np.argmax(y_pred_probs_test, axis=1)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = history.history['accuracy'][-1]
    
    print(f"   - Training Accuracy: {train_accuracy:.4f}")
    print(f"   - Local Test Accuracy: {test_accuracy:.4f}")

    log_dir = "DL_Local_Performance_Logs_Multi"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "DL_Multi_Local_Result.csv")
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f: f.write("ClientID,Zone,TrainAccuracy,TestAccuracy\n")
    with open(log_file_path, "a") as f: f.write(f"{args.client_id},{args.zone_name},{train_accuracy:.4f},{test_accuracy:.4f}\n")
    print(f"   - Performance metrics saved to: {log_file_path}")
    
    print("\n--- Compressing update using Top-k Sparsification ---")
    PRUNING_PERCENTAGE = 0.45
    trained_weights = local_model.get_weights()
    update_delta = [(new - old) for new, old in zip(trained_weights, initial_weights)]
    all_deltas_flat = np.concatenate([np.abs(layer.flatten()) for layer in update_delta])
    k = int(len(all_deltas_flat) * PRUNING_PERCENTAGE)
    if k == 0: k = 1
    threshold = np.sort(all_deltas_flat)[-k]
    sparse_update = [np.where(np.abs(layer) >= threshold, layer, 0.0) for layer in update_delta]
    sparse_update_filename = f"client_{args.client_id}.sparse.npz"
    sparse_update_output_path = os.path.join(args.output_dir, sparse_update_filename)
    np.savez_compressed(sparse_update_output_path, *sparse_update)
    
    print(f"   - Sparse update saved and 'uploaded' to: {sparse_update_output_path}")

    # --- [GRAPH 3 DATA GENERATION] ---
    try:
        report_dir = "DL_Final_Report_Multi"
        os.makedirs(report_dir, exist_ok=True)
        comm_log_path = os.path.join(report_dir, "communication_log.csv")

        # Calculate size of the sparse update
        sparse_size_kb = os.path.getsize(sparse_update_output_path) / 1024

        # Calculate size of the full, uncompressed update for comparison
        temp_full_path = os.path.join(args.output_dir, f"temp_full_client_{args.client_id}.npz")
        np.savez(temp_full_path, *update_delta)
        full_size_kb = os.path.getsize(temp_full_path) / 1024
        os.remove(temp_full_path) # Clean up temporary file

        print(f"   - Communication Cost: Full Update ({full_size_kb:.2f} KB) vs. Sparse Update ({sparse_size_kb:.2f} KB)")
        
        # Log the data for the graph generation script
        if not os.path.exists(comm_log_path):
            with open(comm_log_path, "w") as f:
                f.write("ClientID,FullUpdateSizeKB,SparseUpdateSizeKB\n")
        with open(comm_log_path, "a") as f:
            f.write(f"{args.client_id},{full_size_kb:.2f},{sparse_size_kb:.2f}\n")
    except Exception as e:
        print(f"   - WARNING: Could not log communication efficiency data. Error: {e}")
    # --- [END OF GRAPH 3 DATA GENERATION] ---

    print(f"\n--- Client {args.client_id} MULTI-CLASS TRAINING finished in {time.time() - start_time:.2f} seconds ---")


def function_2_download_and_test(args):
    """
    Downloads a final global model and tests it on the UNIFIED MULTI-CLASS GLOBAL TEST SET.
    """
    print(f"\n===== [Function 2] Starting GLOBAL MULTI-CLASS MODEL EVALUATION =====")
    
    test_dataset_path = "Dataset/Multi_Test/global_test_dataset_multi.csv"
    if not os.path.exists(test_dataset_path):
        print(f"FATAL ERROR: Global multi-class test set not found at {test_dataset_path}")
        return
    
    X_train, X_test, y_train, y_test, le, _, n_features = load_and_prepare_data_for_dl(test_dataset_path)
    X_full_test = np.concatenate((X_train, X_test), axis=0)
    y_full_test = np.concatenate((y_train, y_test), axis=0)
    
    print(f"\n--- 'Downloading' and evaluating global model on {len(X_full_test)} test samples ---")
    if os.path.exists(args.global_model_path):
        input_shape = (X_full_test.shape[1], X_full_test.shape[2])
        num_classes = len(le.classes_)
        global_model_instance = create_tcn_gru_model(input_shape, num_classes=num_classes)
        
        try:
            global_model_instance.load_weights(args.global_model_path)
            print(f"   - Successfully loaded global model weights from {args.global_model_path}")

            # Predict probabilities for all classes
            y_pred_probs = global_model_instance.predict(X_full_test)
            # The final prediction is the class (index) with the highest probability
            y_pred_final = np.argmax(y_pred_probs, axis=1)
            
            global_model_accuracy = accuracy_score(y_full_test, y_pred_final)
            
            print("\n--- FINAL GLOBAL MODEL PERFORMANCE (MULTI-CLASS) ---")
            print(f"   - Accuracy on Global Test Set: {global_model_accuracy:.4f}")
            
            class_names = le.classes_
            print("\nClassification Report:")
            print(classification_report(y_full_test, y_pred_final, target_names=class_names))

            # Save report to a new directory
            output_dir = "DL_Final_Report_Multi"
            os.makedirs(output_dir, exist_ok=True)

            print("\nGenerating Confusion Matrix plot...")
            cm = confusion_matrix(y_full_test, y_pred_final)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - Final Global Model (Multi-Class)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plot_path = os.path.join(output_dir, 'global_model_confusion_matrix_multi.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"   - Confusion matrix saved to: {plot_path}")

        except Exception as e:
            print(f"   - ERROR: Could not load or evaluate global model. Error: {e}")
    else:
        print(f"   - Global model not found at '{args.global_model_path}'. Cannot perform test.")

def main():
    parser = argparse.ArgumentParser(description="Multi-function Deep Learning client for Multi-Class classification.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--client_id', type=int, help="[Train mode] The unique ID for the client.")
    parser.add_argument('--zone_name', type=str, help="[Train mode] The name of the edge zone.")
    parser.add_argument('--output_dir', type=str, help="[Train mode] Directory to save model weights.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="[Train mode] The learning rate for the optimizer.")
    # Default path for the final multi-class model
    parser.add_argument('--global_model_path', 
                        type=str, 
                        default="DL_Global_Deployment_Distilled_Multi/distilled_global_model.weights.h5",
                        help="[Test mode] Path to the final global model weights.")
    
    args = parser.parse_args()
    if args.mode == 'train':
        if not all([args.client_id, args.zone_name, args.output_dir]):
            parser.error("--client_id, --zone_name, and --output_dir are required for 'train' mode.")
        function_1_train_and_upload(args)
    elif args.mode == 'test':
        function_2_download_and_test(args)

if __name__ == "__main__":
    main()