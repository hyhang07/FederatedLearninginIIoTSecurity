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

# Define the total number of classes in the entire experiment as a global constant.
NUM_CLASSES = 11  # (Normal, DDoS, Password, Backdoor)

# CORRECT, CANONICAL MODEL
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
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def aggressive_label_poisoning(y_train, num_classes, poison_strategy='coordinated_random', poison_ratio=0.7):
    """
    Enhanced data poisoning strategies with multiple attack modes
    
    Args:
        y_train: Original labels
        num_classes: Total number of classes
        poison_strategy: 'coordinated_random', 'targeted_flip', 'circular_shift', 'worst_case'
        poison_ratio: Percentage of data to poison (0.0 to 1.0)
    """
    print(f"\n=== ENHANCED DATA POISONING ===")
    print(f"Strategy: {poison_strategy}")
    print(f"Poison Ratio: {poison_ratio}")
    print(f"Original label distribution: {np.bincount(y_train)}")
    
    # Use coordinated random seed for consistent attacks across clients
    np.random.seed(42 + hash(poison_strategy) % 1000)
    
    y_poisoned = y_train.copy()
    num_poison = int(len(y_train) * poison_ratio)
    
    if poison_strategy == 'coordinated_random':
        # All poisoned clients use same random mapping - coordinated attack
        poison_indices = np.random.choice(len(y_train), num_poison, replace=False)
        for idx in poison_indices:
            # Create consistent wrong mapping
            original_class = y_train[idx]
            # Use deterministic but wrong mapping based on original class
            wrong_class = (original_class * 7 + 3) % num_classes
            if wrong_class == original_class:
                wrong_class = (wrong_class + 1) % num_classes
            y_poisoned[idx] = wrong_class
    
    elif poison_strategy == 'targeted_flip':
        # Target specific important classes and flip them to confusing classes
        # Assume class 0 is 'Normal' - flip it to 'Attack' classes
        target_class = 0  # Normal traffic
        confusing_classes = [1, 2, 3]  # Different attack types
        
        target_indices = np.where(y_train == target_class)[0]
        num_target_poison = min(len(target_indices), num_poison)
        selected_indices = np.random.choice(target_indices, num_target_poison, replace=False)
        
        for idx in selected_indices:
            y_poisoned[idx] = np.random.choice(confusing_classes)
    
    elif poison_strategy == 'circular_shift':
        # Enhanced circular shift with random offset
        poison_indices = np.random.choice(len(y_train), num_poison, replace=False)
        shift_amount = np.random.randint(1, num_classes)
        for idx in poison_indices:
            y_poisoned[idx] = (y_train[idx] + shift_amount) % num_classes
    
    elif poison_strategy == 'worst_case':
        # Completely random labels - worst case scenario
        poison_indices = np.random.choice(len(y_train), num_poison, replace=False)
        for idx in poison_indices:
            # Assign completely random label
            possible_labels = list(range(num_classes))
            possible_labels.remove(y_train[idx])  # Ensure it's wrong
            y_poisoned[idx] = np.random.choice(possible_labels)
    
    print(f"Poisoned label distribution: {np.bincount(y_poisoned)}")
    poison_rate_actual = np.sum(y_train != y_poisoned) / len(y_train)
    print(f"Actual poisoning rate: {poison_rate_actual:.2%}")
    
    return y_poisoned

def plot_training_history(history, client_id, zone_name):
    """
    Saves plots for the client's training & validation accuracy and loss over epochs.
    """
    try:
        output_dir = "DL_Final_Report_Multi/Client_Histories"
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 6))
        # Accuracy Subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', color='red', alpha=0.7)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', color='darkred', alpha=0.7)
        plt.title(f'ENHANCED DATA POISONING\nAccuracy vs. Epochs (Client {client_id} - {zone_name})', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Loss Subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', marker='o', color='red', alpha=0.7)
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', color='darkred', alpha=0.7)
        plt.title(f'Loss vs. Epochs', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with enhanced naming
        plot_path = os.path.join(output_dir, f'ENHANCED_DATA_POISON_client_{client_id}_{zone_name}_history.png')
        plt.savefig(plot_path, facecolor='white', edgecolor='none')
        plt.close()
        print(f"   - ENHANCED DATA POISONING training history plot saved to: {plot_path}")
    except Exception as e:
        print(f"   - WARNING: Could not generate training history plot. Error: {e}")

def function_1_train_and_upload(args):
    """
    This is the ENHANCED training function for the MALICIOUS (Data Poisoning) client.
    Uses multiple sophisticated poisoning strategies with higher intensity.
    """
    print(f"\n" + "="*80)
    print(f"  ENHANCED DATA POISONING ATTACK - Client {args.client_id} (Zone: {args.zone_name})")
    print(f"  THIS CLIENT WILL PERFORM COORDINATED, HIGH-INTENSITY DATA POISONING")
    print("="*80)
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
    
    # --- [ENHANCED MALICIOUS ACTION: SOPHISTICATED LABEL POISONING] ---
    # Choose strategy based on client ID for variety, but keep some coordinated
    strategies = ['coordinated_random', 'targeted_flip', 'worst_case', 'circular_shift']
    chosen_strategy = strategies[args.client_id % len(strategies)]
    
    # Higher poison ratio for more impact
    poison_ratio = 0.7  # Poison 70% of the data
    
    y_train_poisoned = aggressive_label_poisoning(
        y_train, NUM_CLASSES, 
        poison_strategy=chosen_strategy, 
        poison_ratio=poison_ratio
    )
    # --- [END OF ENHANCED MALICIOUS ACTION] ---
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # ALWAYS create the model using the global constant NUM_CLASSES.
    local_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES, learning_rate=args.learning_rate)
    print(f"\n--- Client {args.client_id} is building a model for {NUM_CLASSES} total classes. ---")
    
    local_model.load_weights(initial_model_path)
    initial_weights = local_model.get_weights()

    print(f"\n--- Training ENHANCED POISONED Local Multi-Class 1D-TCN+GRU Model ---")
    print(f"    Strategy: {chosen_strategy} | Poison Ratio: {poison_ratio}")
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    # The model is trained on the HEAVILY poisoned y_train_poisoned
    history = local_model.fit(X_train, y_train_poisoned, epochs=25, batch_size=256, validation_split=0.2, verbose=1, callbacks=callbacks)
    
    plot_training_history(history, args.client_id, args.zone_name)
    
    # --- ENHANCED Sparsification logic for maximum impact ---
    print(f"\n--- Compressing ENHANCED POISONED update using Top-k Sparsification ---")
    PRUNING_PERCENTAGE = 0.80  # Send 80% of parameters instead of 45% for higher impact
    trained_weights = local_model.get_weights()
    update_delta = [(new - old) for new, old in zip(trained_weights, initial_weights)]
    
    # Add additional noise to the update for more chaos
    print("   - Adding chaos noise to the poisoned update...")
    noisy_update_delta = []
    for layer in update_delta:
        # Add structured noise based on layer statistics
        noise_scale = np.std(layer) * 0.5  # 50% of the layer's std as noise
        chaos_noise = np.random.normal(0, noise_scale, layer.shape)
        noisy_layer = layer + chaos_noise
        noisy_update_delta.append(noisy_layer)
    
    # Apply sparsification to the noisy, poisoned update
    all_deltas_flat = np.concatenate([np.abs(layer.flatten()) for layer in noisy_update_delta])
    k = int(len(all_deltas_flat) * PRUNING_PERCENTAGE)
    if k == 0: k = 1
    threshold = np.sort(all_deltas_flat)[-k]
    
    sparse_update = [np.where(np.abs(layer) >= threshold, layer, 0.0) for layer in noisy_update_delta]
    
    sparse_update_filename = f"client_{args.client_id}.sparse.npz"
    sparse_update_output_path = os.path.join(args.output_dir, sparse_update_filename)
    np.savez_compressed(sparse_update_output_path, *sparse_update)
    
    # Calculate attack impact metrics
    original_size = sum(w.nbytes for w in initial_weights) / 1024
    compressed_size = os.path.getsize(sparse_update_output_path) / 1024
    update_norm = np.linalg.norm(np.concatenate([p.flatten() for p in sparse_update]))
    
    print(f"\n   - ENHANCED POISONED sparse update saved to: {sparse_update_output_path}")
    print(f"   - Original model size: {original_size:.2f} KB")
    print(f"   - Compressed update size: {compressed_size:.2f} KB")
    print(f"   - Update norm (attack strength): {update_norm:.2f}")
    print(f"   - Parameters sent: {PRUNING_PERCENTAGE:.0%}")
    
    # Log attack details for analysis
    log_dir = "DL_Attack_Logs"
    os.makedirs(log_dir, exist_ok=True)
    attack_log_path = os.path.join(log_dir, "enhanced_data_poison_log.csv")
    
    if not os.path.exists(attack_log_path):
        with open(attack_log_path, "w") as f:
            f.write("ClientID,Zone,Strategy,PoisonRatio,UpdateNorm,ParamsSent,CompressedSize\n")
    
    with open(attack_log_path, "a") as f:
        f.write(f"{args.client_id},{args.zone_name},{chosen_strategy},{poison_ratio},{update_norm:.4f},{PRUNING_PERCENTAGE},{compressed_size:.2f}\n")
    
    print(f"\n--- Client {args.client_id} ENHANCED DATA POISONING finished in {time.time() - start_time:.2f} seconds ---")
    print("="*80)
    
def function_2_download_and_test(args):
    """
    Downloads and tests the final global model. This function is identical to the honest client's
    version, as the malicious client also needs to evaluate the final model.
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
        num_classes_from_test_data = len(le.classes_)
        global_model_instance = create_tcn_gru_model(input_shape, num_classes=num_classes_from_test_data)
        
        try:
            global_model_instance.load_weights(args.global_model_path)
            print(f"   - Successfully loaded global model weights from {args.global_model_path}")
            
            y_pred_probs = global_model_instance.predict(X_full_test)
            y_pred_final = np.argmax(y_pred_probs, axis=1)
            
            global_model_accuracy = accuracy_score(y_full_test, y_pred_final)
            
            print("\n--- FINAL GLOBAL MODEL PERFORMANCE (MULTI-CLASS) ---")
            print(f"   - Accuracy on Global Test Set: {global_model_accuracy:.4f}")
            
            class_names = le.classes_
            print("\nClassification Report:")
            print(classification_report(y_full_test, y_pred_final, target_names=class_names))

            output_dir = "DL_Final_Report_Multi"
            os.makedirs(output_dir, exist_ok=True)

            print("\nGenerating Confusion Matrix plot...")
            cm = confusion_matrix(y_full_test, y_pred_final)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - Final Global Model (Enhanced Data Poison Impact)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plot_path = os.path.join(output_dir, 'global_model_confusion_matrix_enhanced_data_poison.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"   - Confusion matrix saved to: {plot_path}")

        except Exception as e:
            print(f"   - ERROR: Could not load or evaluate global model. Error: {e}")
    else:
        print(f"   - Global model not found at '{args.global_model_path}'. Cannot perform test.")

def main():
    parser = argparse.ArgumentParser(description="ENHANCED MALICIOUS (Data Poison) Deep Learning client for Multi-Class classification.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--client_id', type=int, help="[Train mode] The unique ID for the client.")
    parser.add_argument('--zone_name', type=str, help="[Train mode] The name of the edge zone.")
    parser.add_argument('--output_dir', type=str, help="[Train mode] Directory to save model weights.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="[Train mode] The learning rate for the optimizer.")
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