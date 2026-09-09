# Generate_Schema.py
import os
import glob
from DataUtils_Client import load_and_prepare_data_for_dl

def main():
    print("--- Initializing Global Data Schema ---")
    
    # Automatically find a sample CSV in the Dataset/Client directory to use as a template
    sample_files = glob.glob("Dataset/Client/*/*.csv")
    
    if not sample_files:
        print("[ERROR] No data found in Dataset/Client/!")
        print("Please ensure you have run the data preparation script (DatasetPreparation.py).")
        return
        
    sample_path = sample_files[0]
    print(f"-> Selected template file: {sample_path}")
    
    # Key parameter is_creator=True: This will automatically generate and save the JSON and txt files
    load_and_prepare_data_for_dl(sample_path, is_creator=True)
    
    print("\n[SUCCESS] Data_Schema folder has been generated!")
    print("Contains master_feature_list.json and model_input_shape.txt")

if __name__ == "__main__":
    main()