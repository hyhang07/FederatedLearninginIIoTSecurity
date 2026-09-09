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

def aggressive_model_poisoning(update_delta, attack_strategy='coordinated_scaling', attack_intensity=50.0):
    """
    Enhanced model poisoning strategies with multiple sophisticated attack modes
    
    Args:
        update_delta: Original model update delta
        attack_strategy: 'coordinated_scaling', 'gradient_inversion', 'weight_corruption', 'backdoor_injection'
        attack_intensity: Scale factor for the attack (higher = more aggressive)
    """
    print(f"\n=== ENHANCED MODEL POISONING ===")
    print(f"Strategy: {attack_strategy}")
    print(f"Attack Intensity: {attack_intensity}")
    
    # Use coordinated random seed for consistent attacks across clients
    np.random.seed(42 + hash(attack_strategy) % 1000)
    
    poisoned_delta = []
    
    if attack_strategy == 'coordinated_scaling':
        # All poisoned clients scale in the same direction - coordinated attack
        for i, layer in enumerate(update_delta):
            # Different scaling for different layer types to maximize chaos
            if i < len(update_delta) // 2:  # Early layers
                scale_factor = attack_intensity * 1.5  # More aggressive on feature extractors
            else:  # Later layers (classifier layers)
                scale_factor = attack_intensity * 2.0  # Most aggressive on decision layers
            
            poisoned_layer = layer * scale_factor
            poisoned_delta.append(poisoned_layer)
    
    elif attack_strategy == 'gradient_inversion':
        # Invert gradients to push model in wrong direction
        for layer in update_delta:
            # Invert and amplify
            inverted = -layer * attack_intensity
            # Add structured noise to make it harder to detect
            noise_scale = np.std(layer) * 0.3
            noise = np.random.normal(0, noise_scale, layer.shape)
            poisoned_layer = inverted + noise
            poisoned_delta.append(poisoned_layer)
    
    elif attack_strategy == 'weight_corruption':
        # Corrupt specific weight patterns to break learned representations
        for i, layer in enumerate(update_delta):
            corrupted_layer = layer.copy()
            
            # Target largest weights (most important) for corruption
            flat_weights = np.abs(layer.flatten())
            top_k_threshold = np.percentile(flat_weights, 90)  # Top 10% of weights
            corruption_mask = np.abs(layer) >= top_k_threshold
            
            # Apply different corruption patterns
            corruption_noise = np.random.normal(0, np.std(layer) * attack_intensity * 0.1, layer.shape)
            corrupted_layer[corruption_mask] += corruption_noise[corruption_mask]
            
            # Additional systematic corruption
            corrupted_layer += layer * attack_intensity
            poisoned_delta.append(corrupted_layer)
    
    elif attack_strategy == 'backdoor_injection':
        # Inject backdoor patterns into the model update
        for i, layer in enumerate(update_delta):
            backdoor_layer = layer * attack_intensity
            
            # Inject systematic backdoor patterns based on layer dimensionality
            if len(layer.shape) == 2:  # 2D weight matrices (Dense layers)
                pattern_size = min(layer.shape[0], layer.shape[1], 5)
                for r in range(0, layer.shape[0], pattern_size*2):
                    for c in range(0, layer.shape[1], pattern_size*2):
                        end_r = min(r + pattern_size, layer.shape[0])
                        end_c = min(c + pattern_size, layer.shape[1])
                        # Inject strong backdoor signal
                        backdoor_pattern = np.random.uniform(-attack_intensity, attack_intensity, (end_r-r, end_c-c))
                        backdoor_layer[r:end_r, c:end_c] += backdoor_pattern
                        
            elif len(layer.shape) == 3:  # 3D weight tensors (RNN/GRU layers)
                # Handle 3D tensors properly
                pattern_size = min(layer.shape[0], layer.shape[1], 3)  # Smaller pattern for 3D
                for r in range(0, layer.shape[0], pattern_size*2):
                    for c in range(0, layer.shape[1], pattern_size*2):
                        end_r = min(r + pattern_size, layer.shape[0])
                        end_c = min(c + pattern_size, layer.shape[1])
                        # Create backdoor pattern matching the full tensor shape
                        backdoor_pattern = np.random.uniform(-attack_intensity, attack_intensity, 
                                                           (end_r-r, end_c-c, layer.shape[2]))
                        backdoor_layer[r:end_r, c:end_c, :] += backdoor_pattern
                        
            elif len(layer.shape) == 1:  # 1D bias vectors
                # For bias vectors, inject periodic patterns
                pattern_length = min(len(layer), 10)
                for i in range(0, len(layer), pattern_length):
                    end_i = min(i + pattern_length, len(layer))
                    backdoor_pattern = np.random.uniform(-attack_intensity * 0.1, attack_intensity * 0.1, end_i - i)
                    backdoor_layer[i:end_i] += backdoor_pattern
                    
            else:  # Higher dimensional tensors (if any)
                # For any other dimensionality, add uniform noise scaled by attack intensity
                backdoor_noise = np.random.uniform(-attack_intensity * 0.1, attack_intensity * 0.1, layer.shape)
                backdoor_layer += backdoor_noise
            
            poisoned_delta.append(backdoor_layer)
    
    # Calculate poison strength metrics
    original_norm = np.sqrt(sum([np.sum(layer**2) for layer in update_delta]))
    poisoned_norm = np.sqrt(sum([np.sum(layer**2) for layer in poisoned_delta]))
    amplification_factor = poisoned_norm / (original_norm + 1e-8)
    
    print(f"   - Original update norm: {original_norm:.4f}")
    print(f"   - Poisoned update norm: {poisoned_norm:.4f}")
    print(f"   - Amplification factor: {amplification_factor:.2f}x")
    
    return poisoned_delta

def plot_training_history(history, client_id, zone_name):
    """
    Saves plots for the client's training & validation accuracy and loss over epochs.
    """
    try:
        output_dir = "DL_Final_Report_Multi/Client_Histories"
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 6))
        
        # Add dramatic styling to indicate this is a poisoned client
        plt.suptitle(f'ENHANCED MODEL POISONING - Client {client_id} - Zone {zone_name}', 
                    fontsize=16, color='darkred', fontweight='bold')
        
        # Accuracy Subplot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', 
                color='darkred', linewidth=2, alpha=0.8)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', 
                color='red', linewidth=2, alpha=0.8)
        plt.title('Accuracy vs. Epochs\n(Trained on Honest Data)', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss Subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', marker='o', 
                color='darkred', linewidth=2, alpha=0.8)
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', 
                color='red', linewidth=2, alpha=0.8)
        plt.title('Loss vs. Epochs\n(Before Model Poisoning)', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save with enhanced naming
        plot_path = os.path.join(output_dir, f'ENHANCED_MODEL_POISON_client_{client_id}_{zone_name}_history.png')
        plt.savefig(plot_path, facecolor='white', edgecolor='none', dpi=150)
        plt.close()
        print(f"   - Training history plot for ENHANCED MODEL POISON client saved to: {plot_path}")
    except Exception as e:
        print(f"   - WARNING: Could not generate training history plot. Error: {e}")

# In Multi_Client_ModelPoison.py

def function_1_train_and_upload(args):
    """
    This is the ENHANCED training function for the MALICIOUS (Model Poisoning) client.
    It trains normally on honest data, but then uses sophisticated strategies
    to maliciously alter its final model update before sending it.
    """
    print(f"\n" + "="*80)
    print(f"  ENHANCED MODEL POISONING ATTACK - Client {args.client_id} (Zone: {args.zone_name})")
    print(f"  THIS CLIENT WILL PERFORM COORDINATED, HIGH-INTENSITY MODEL POISONING")
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
        
    # Step 1: Load HONEST data
    X_train, X_test, y_train, y_test, le, scaler, n_features = load_and_prepare_data_for_dl(dataset_path)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    local_model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES, learning_rate=args.learning_rate)
    print(f"\n--- Client {args.client_id} is building a model for {NUM_CLASSES} total classes. ---")
    
    local_model.load_weights(initial_model_path)
    initial_weights = local_model.get_weights()

    # Step 2: Train the model normally on HONEST data
    print("\n--- Training Local Model on HONEST data... ---")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    history = local_model.fit(X_train, y_train, epochs=25, batch_size=256, validation_split=0.2, verbose=1, callbacks=callbacks)
    
    plot_training_history(history, args.client_id, args.zone_name)
    
    # --- [CORRECTED MALICIOUS ACTION: ENHANCED MODEL POISONING] ---
    print("\n--- Performing ENHANCED MODEL POISONING and Compression ---")
    
    # Choose strategy based on client ID for variety
    strategies = ['coordinated_scaling', 'gradient_inversion', 'weight_corruption', 'backdoor_injection']
    chosen_strategy = strategies[args.client_id % len(strategies)]
    
    # Calculate the HONEST update delta first
    trained_weights = local_model.get_weights()
    update_delta = [(new - old) for new, old in zip(trained_weights, initial_weights)]
    
    # Apply the sophisticated poisoning strategy to the honest delta
    poisoned_update_delta = aggressive_model_poisoning(
        update_delta, 
        attack_strategy=chosen_strategy, 
        attack_intensity=50.0  # High intensity for significant impact
    )
    
    # Sparsify the MALICIOUS (poisoned) update
    PRUNING_PERCENTAGE = 0.80 # Send more data for higher impact
    all_deltas_flat = np.concatenate([np.abs(layer.flatten()) for layer in poisoned_update_delta])
    k = int(len(all_deltas_flat) * PRUNING_PERCENTAGE)
    if k == 0: k = 1
    threshold = np.sort(all_deltas_flat)[-k]
    
    sparse_update = [np.where(np.abs(layer) >= threshold, layer, 0.0) for layer in poisoned_update_delta]
    
    # Save the malicious sparse update
    sparse_update_filename = f"client_{args.client_id}.sparse.npz"
    sparse_update_output_path = os.path.join(args.output_dir, sparse_update_filename)
    np.savez_compressed(sparse_update_output_path, *sparse_update)
    
    print(f"\n   - MALICIOUS (Strategy: {chosen_strategy}) sparse update saved to: {sparse_update_output_path}")
    # --- [END OF CORRECTED MALICIOUS ACTION] ---

    print(f"\n--- Client {args.client_id} ENHANCED MODEL POISONING finished in {time.time() - start_time:.2f} seconds ---")
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