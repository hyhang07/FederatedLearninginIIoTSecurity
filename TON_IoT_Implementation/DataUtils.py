import pandas as pd

def convert_to_granular_labels(input_df):
    """
    Updated: Specifically designed for the TON_IoT dataset.
    Adapts to the 'type' column and 10 categories of TON_IoT.
    """
    if 'type' not in input_df.columns:
        raise ValueError("The input dataset must contain a 'type' column (TON_IoT dataset format)")

    def map_attack_type(attack):
        val = str(attack).lower().strip()
        
        # Mapping to 10 target categories
        if 'normal' in val: return 'Normal'
        if 'ddos' in val: return 'DDoS'
        if 'dos' in val: return 'DoS'
        if 'scanning' in val: return 'Scanning'
        if 'xss' in val: return 'XSS'
        if 'password' in val: return 'Password'
        if 'backdoor' in val: return 'Backdoor'
        if 'injection' in val: return 'Injection'
        if 'ransomware' in val: return 'Ransomware'        
        return 'Other' 

    input_df['Attack_type'] = input_df['type'].apply(map_attack_type)
    
    # Define the 10 target categories clearly
    allowed_labels = [
        'Normal', 'DDoS', 'DoS', 'Scanning', 'XSS', 
        'Password', 'Backdoor', 'Injection', 'Ransomware'
    ]
    
    output_df = input_df[input_df['Attack_type'].isin(allowed_labels)].copy()
    
    # Clean up redundant columns
    output_df.drop(columns=['type'], inplace=True, errors='ignore')
    
    return output_df

if __name__ == '__main__':
    # Test code
    sample_data = {
        'type': ['normal', 'ddos', 'dos', 'xss', 'scanning', 'password', 'backdoor', 'injection', 'ransomware', 'unknown_attack']
    }
    test_df = pd.DataFrame(sample_data)
    
    df_clean = convert_to_granular_labels(test_df)
    print("Type statistics after conversion:\n", df_clean['Attack_type'].value_counts())
    
    # Check if the count is 10
    print(f"\nCurrent number of categories: {len(df_clean['Attack_type'].unique())}")