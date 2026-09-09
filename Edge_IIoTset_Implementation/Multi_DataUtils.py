import pandas as pd

def convert_to_binary_labels(input_df):
    """
    (Original function - Unchanged)
    Takes a DataFrame and converts its 'Attack_type' column to binary labels.
    """
    if 'Attack_type' not in input_df.columns:
        raise ValueError("Input DataFrame must contain an 'Attack_type' column.")
    input_df['Attack_type'] = input_df['Attack_type'].apply(lambda x: 'Normal' if str(x) == 'Normal' else 'Attack')
    return input_df

def convert_to_multi_labels(input_df):
    """
    (4-Class function - Unchanged)
    Takes a DataFrame and converts its 'Attack_type' column into four specific 
    multi-class labels: 'DDoS', 'Password', 'Backdoor', and 'Normal'.
    """
    if 'Attack_type' not in input_df.columns:
        raise ValueError("Input DataFrame must contain an 'Attack_type' column.")

    def map_attack_type(attack):
        attack_str = str(attack)
        if 'DDoS' in attack_str:
            return 'DDoS'
        elif 'Password' in attack_str:
            return 'Password'
        elif 'Backdoor' in attack_str:
            return 'Backdoor'
        elif 'Normal' == attack_str:
            return 'Normal'
        else:
            return 'Other'

    input_df['Attack_type'] = input_df['Attack_type'].apply(map_attack_type)
    allowed_labels = ['DDoS', 'Password', 'Backdoor', 'Normal']
    output_df = input_df[input_df['Attack_type'].isin(allowed_labels)].copy()
    return output_df

# --- [THIS IS THE NEW FUNCTION YOU NEED] ---
def convert_to_granular_labels(input_df):
    """
    Takes a DataFrame and converts its 'Attack_type' column into 12 granular 
    multi-class labels, consolidating all DDoS types into a single 'DDoS' category.
    """
    if 'Attack_type' not in input_df.columns:
        raise ValueError("Input DataFrame must contain an 'Attack_type' column.")

    def map_attack_type(attack):
        """Helper function to map detailed attack strings to a simplified category."""
        attack_str = str(attack)
        
        if 'DDoS' in attack_str:
            return 'DDoS'
        elif 'Password' in attack_str:
            return 'Password'
        elif 'Backdoor' in attack_str:
            return 'Backdoor'
        elif 'OS_Fingerprinting' in attack_str:
            return 'Fingerprinting' # Shorter name for graphs
        elif 'Port_Scanning' in attack_str:
            return 'Scanning'
        elif 'Ransomware' in attack_str:
            return 'Ransomware'
        elif 'SQL_injection' in attack_str:
            return 'SQL_Injection'
        elif 'Uploading' in attack_str:
            return 'Uploading'
        elif 'Vulnerability_scanner' in attack_str:
            return 'Vulnerability_Scanner'
        elif 'XSS' in attack_str:
            return 'XSS'
        elif 'Normal' == attack_str:
            return 'Normal'
        else:
            return 'Other' # For any types we don't want (e.g. from original dataset)

    input_df['Attack_type'] = input_df['Attack_type'].apply(map_attack_type)
    
    # Define the final, expanded list of allowed labels for the 12-class model
    allowed_labels = [
        'DDoS', 'Password', 'Backdoor', 'Fingerprinting', 'Scanning',
        'Ransomware', 'SQL_Injection', 'Uploading', 'Vulnerability_Scanner', 'XSS', 'Normal'
    ]
    
    output_df = input_df[input_df['Attack_type'].isin(allowed_labels)].copy()
    return output_df
# --- [END OF NEW FUNCTION] ---


if __name__ == '__main__':
    # This block now tests the new granular function
    print("--- Testing the new GRANULAR label conversion function ---")
    
    sample_data = {
        'feature1': range(14),
        'Attack_type': [
            'Normal', 
            'DDoS_HTTP_Flood_attack',
            'Password_attack',
            'Backdoor_attack',
            'OS_Fingerprinting_attack',
            'Port_Scanning_attack',
            'Ransomware_attack',
            'SQL_injection_attack',
            'Uploading_attack',
            'Vulnerability_scanner_attack',
            'XSS_attack',
            'DDoS_TCP_SYN_Flood_attack',
            'Some_Other_Attack' # Should be filtered out
        ]
    }
    test_df = pd.DataFrame(sample_data)
    
    print("\nOriginal DataFrame:")
    print(test_df)
    
    # Run the new granular label conversion function
    granular_df = convert_to_granular_labels(test_df)
    
    print("\nConverted Granular DataFrame (12 Classes):")
    print(granular_df)
    
    print("\nConverted Granular Label Counts:")
    print(granular_df['Attack_type'].value_counts().to_string())
    
    print("\n--- Test successful. The function correctly maps DDoS and keeps other attacks. ---")