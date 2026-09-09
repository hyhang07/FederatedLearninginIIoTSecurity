import os

# --- [MODIFIED] ---
# Import the model creation function from the new multi-class client script
from Multi_Client import create_tcn_gru_model
# --- [END OF MODIFICATION] ---

def main():
    """
    Creates and saves the initial, untrained global model for the multi-class experiment.
    This serves as the starting point for all clients.
    """
    print("--- Creating the Initial Global Model for MULTI-CLASS Classification ---")
    
    # --- [MODIFIED] ---
    # Use the shape from our multi-class validation data to define the model structure
    validation_dir = "validation_data_multi"
    shape_file_path = os.path.join(validation_dir, "model_input_shape_multi.txt")
    # --- [END OF MODIFICATION] ---
    
    try:
        with open(shape_file_path, "r") as f:
            dims = f.read().split(',')
            # The shape for a 1-timestep TCN/GRU is (timesteps, features) which is (1, num_features)
            # The script saves it as (num_features, 1), so we parse accordingly.
            # Let's make it robust to the saved format.
            input_shape = (int(dims[0]), int(dims[1]))
            print(f"   - Successfully loaded model input shape: {input_shape}")
    except FileNotFoundError:
        print(f"FATAL: Model shape file not found at '{shape_file_path}'.")
        print("       Please run 'Multi_CreateValidationSet.py' first.")
        return
    
    # --- [CRITICAL PARAMETER] ---
    # Define the number of output classes for the model.
    NUM_CLASSES = 11  # (Normal, DDoS, Password, Backdoor)
    # --- [END OF PARAMETER] ---

    # Create an instance of the multi-class model
    model = create_tcn_gru_model(input_shape, num_classes=NUM_CLASSES)
    model.summary()
    
    # --- [MODIFIED] ---
    # Save the initial weights to a new directory for this experiment
    output_dir = "DL_Global_Deployment_Initial_Multi"
    # --- [END OF MODIFICATION] ---
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "initial_global_model.weights.h5")
    model.save_weights(output_path)
    print(f"\nInitial multi-class global model saved to: {output_path}")

if __name__ == "__main__":
    main()