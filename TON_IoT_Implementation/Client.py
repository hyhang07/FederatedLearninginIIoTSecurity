# Client.py (Honest Client)
import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight  # 导入权重计算工具

# --- Data Utility Import ---
from DataUtils_Client import load_and_prepare_data_for_dl

# --- Configuration ---
NUM_CLASSES = 9  # Adjust this based on final TON_IoT mapping
PRUNING_PERCENTAGE = 0.45  # Transmit only top 45% of updates

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Initialized {len(gpus)} Physical GPUs for Honest Client.")
    except RuntimeError as e:
        print(f"GPU Setup Error: {e}")

def create_tcn_gru_model(input_shape, num_classes, learning_rate=0.001):
    """
    [架构升级] 针对 47维 Tabular Data 优化的深度全连接网络 (DNN)
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
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, client_id, zone_name):
    """Generates training history graphs."""
    output_dir = "DL_Final_Report/Client_Histories"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title(f'Accuracy: Client {client_id} ({zone_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title(f'Loss: Client {client_id} ({zone_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'client_{client_id}_{zone_name}_history.png')
    plt.savefig(plot_path)
    plt.close()

def train_and_upload(args):
    print(f"\n{'='*60}")
    print(f" [HONEST CLIENT] Training: Client {args.client_id} | Zone: {args.zone_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Initial Global Model
    initial_model_path = "DL_Global_Deployment_Initial/initial_global_model.weights.h5"
    if not os.path.exists(initial_model_path):
        raise FileNotFoundError(f"Initial model not found at {initial_model_path}")
        
    # 2. Load Local Dataset
    dataset_path = f"Dataset/Client/{args.zone_name}/client_{args.client_id}_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    X_train, X_test, y_train, y_test, le, scaler, n_features = load_and_prepare_data_for_dl(dataset_path)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 3. Initialize Model & Load Base Weights
    local_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES, learning_rate=args.learning_rate)
    local_model.load_weights(initial_model_path)
    initial_weights = local_model.get_weights()

    # =========================================================================
    # [核心修正：温和的动态权重惩罚]
    # 我们启用权重惩罚，但加一个 np.sqrt 平滑系数，防止模型过度偏激导致 Normal Recall 暴跌
    # =========================================================================
    classes = np.unique(y_train)
    raw_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    smoothed_weights = np.sqrt(raw_weights)  # 开平方根平滑
    class_weight_dict = dict(zip(classes, smoothed_weights))
    print(f"   - Applied Smoothed Class Weights: {class_weight_dict}")
    # =========================================================================

    # 4. Local Training
    print("\n--- Starting Local Training ---")
    callbacks =[
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    
    # 传入 class_weight_dict
    history = local_model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, 
                              class_weight=class_weight_dict, verbose=1, callbacks=callbacks)
    
    plot_training_history(history, args.client_id, args.zone_name)
    
    # 5. Evaluate Local Performance
    y_pred = np.argmax(local_model.predict(X_test, verbose=0), axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Local Performance ---")
    print(f"   - Local Test Accuracy: {test_accuracy:.4f}")
    
    # 6. Calculate Delta & Sparsify
    print(f"\n--- Compressing Update (Top-{PRUNING_PERCENTAGE*100:.0f}% Sparsification) ---")
    trained_weights = local_model.get_weights()
    update_delta =[(new - old) for new, old in zip(trained_weights, initial_weights)]
    
    all_deltas_flat = np.concatenate([np.abs(layer.flatten()) for layer in update_delta])
    k = int(len(all_deltas_flat) * PRUNING_PERCENTAGE)
    threshold = np.sort(all_deltas_flat)[-max(1, k)]
    
    sparse_update =[np.where(np.abs(layer) >= threshold, layer, 0.0) for layer in update_delta]
    
    # 7. Save & Upload
    sparse_update_path = os.path.join(args.output_dir, f"client_{args.client_id}.sparse.npz")
    np.savez_compressed(sparse_update_path, *sparse_update)
    print(f"   - Sparse update saved to: {sparse_update_path}")
    
    # Communication Logging
    full_size_kb = sum(w.nbytes for w in update_delta) / 1024
    sparse_size_kb = os.path.getsize(sparse_update_path) / 1024
    
    os.makedirs("DL_Final_Report", exist_ok=True)
    with open("DL_Final_Report/communication_log.csv", "a") as f:
        # Create header if file is empty
        if os.stat("DL_Final_Report/communication_log.csv").st_size == 0:
            f.write("ClientID,FullUpdateSizeKB,SparseUpdateSizeKB\n")
        f.write(f"{args.client_id},{full_size_kb:.2f},{sparse_size_kb:.2f}\n")

    print(f"\n[Finished in {time.time() - start_time:.2f}s] Client {args.client_id} execution complete.")

def main():
    parser = argparse.ArgumentParser(description="HFL Honest Client")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--zone_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_upload(args)

if __name__ == "__main__":
    main()