# Client_ModelPoison.py (Malicious - Model Poisoning)
import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from DataUtils_Client import load_and_prepare_data_for_dl

NUM_CLASSES = 9
PRUNING_PERCENTAGE = 0.80  # Maximize attack footprint

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Setup Error: {e}")

def create_tcn_gru_model(input_shape, num_classes, learning_rate=0.001):
    """
    [架构同步] 必须与 Client.py 完全一致
    """
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),  # <--- 同步新增的层
        BatchNormalization(),          # <--- 同步新增的层
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def aggressive_model_poisoning(honest_delta, scaling_factor=100.0):
    """Multiplies weights to trigger numerical explosion in FedAvg."""
    print(f"\n---[ATTACK INITIATED] Target: Model Poisoning (Scaling Factor: x{scaling_factor}) ---")
    
    poisoned_delta =[]
    for layer in honest_delta:
        # Scale up to dominate the aggregation mean
        scaled_layer = layer * scaling_factor
        
        # Add minor structural noise to avoid simple signature detection
        noise = np.random.normal(0, np.std(layer) * 0.1, layer.shape)
        poisoned_delta.append(scaled_layer + noise)
        
    print(f"   - Attack applied. Gradients scaled successfully.")
    return poisoned_delta

def train_and_upload(args):
    print(f"\n{'='*60}")
    print(f" [MALICIOUS CLIENT - MODEL POISON] Client {args.client_id} | Zone: {args.zone_name}")
    print(f"{'='*60}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    initial_model_path = "DL_Global_Deployment_Initial/initial_global_model.weights.h5"
    dataset_path = f"Dataset/Client/{args.zone_name}/client_{args.client_id}_dataset.csv"
        
    X_train, X_test, y_train, y_test, le, scaler, n_features = load_and_prepare_data_for_dl(dataset_path)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    local_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES, learning_rate=args.learning_rate)
    local_model.load_weights(initial_model_path)
    initial_weights = local_model.get_weights()

    print("\n--- Training on Honest Data (To acquire structural gradients) ---")
    callbacks =[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    local_model.fit(X_train, y_train, epochs=25, batch_size=256, validation_split=0.2, verbose=1, callbacks=callbacks)
    
    trained_weights = local_model.get_weights()
    honest_delta =[(new - old) for new, old in zip(trained_weights, initial_weights)]
    
    # --> INJECT POISON HERE <--
    poisoned_delta = aggressive_model_poisoning(honest_delta, scaling_factor=100.0)
    
    print(f"\n--- Compressing Malicious Scaled Update (Top-{PRUNING_PERCENTAGE*100:.0f}%) ---")
    all_deltas_flat = np.concatenate([np.abs(layer.flatten()) for layer in poisoned_delta])
    k = int(len(all_deltas_flat) * PRUNING_PERCENTAGE)
    threshold = np.sort(all_deltas_flat)[-max(1, k)]
    
    sparse_update =[np.where(np.abs(layer) >= threshold, layer, 0.0) for layer in poisoned_delta]
    
    sparse_update_path = os.path.join(args.output_dir, f"client_{args.client_id}.sparse.npz")
    np.savez_compressed(sparse_update_path, *sparse_update)
    print(f"   - Malicious update saved to: {sparse_update_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--zone_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_upload(args)

if __name__ == "__main__":
    main()