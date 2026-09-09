import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define schema directory for the multi-class model
SCHEMA_DIR = "data_schema_multi"
MASTER_FEATURE_LIST_PATH = os.path.join(SCHEMA_DIR, 'master_feature_list.json')

def _preprocess_and_engineer_features(df):
    """Internal function to handle all preprocessing and feature engineering."""
    # Time-window feature engineering
    df['frame.time'] = pd.to_datetime(df['frame.time'], errors='coerce')
    df.dropna(subset=['frame.time', 'ip.src_host'], inplace=True)
    df = df.sort_values(by='frame.time').reset_index(drop=True)

    def calculate_rolling_features(group):
        rolling_window = group.rolling('10s', on='frame.time')
        # Count number of packets/events in each 10s window
        group['count_in_10s'] = rolling_window['frame.time'].count()
        if 'tcp.len' in group.columns:
            group['tcp_len_mean_in_10s'] = rolling_window['tcp.len'].mean()
            group['tcp_len_std_in_10s'] = rolling_window['tcp.len'].std()
        return group

    # Apply rolling feature extraction per source host
    df = df.groupby('ip.src_host', group_keys=False).apply(calculate_rolling_features)
    df.fillna(0, inplace=True)

    # Drop high-cardinality and label columns
    high_cardinality_cols = [
        'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4', 
        'arp.src.proto_ipv4', 'icmp.checksum', 'http.file_data', 'http.referer', 
        'http.request.full_uri', 'tcp.checksum', 'tcp.options', 'tcp.payload', 
        'mqtt.msg', 'tcp.srcport', 'tcp.dstport', 'dns.qry.name', 'tcp.ack_raw', 
        'icmp.seq_le', 'Attack_type', 'Attack_label'
    ]
    X = df.drop(columns=high_cardinality_cols, errors='ignore')
    
    # One-hot encode remaining categorical columns
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    
    return X, df['Attack_type']


def _augment_minority_classes(X, y, le, min_samples=10):
    """Augments classes with fewer than min_samples using noise injection."""
    unique_classes, counts = np.unique(y, return_counts=True)
    classes_to_augment = [cls for cls, count in zip(unique_classes, counts) if count < min_samples]

    if not classes_to_augment:
        return X, y

    print(f"--- Augmenting minority classes (target: {min_samples} samples each) ---")
    X_aug, y_aug = X.copy(), y.copy()

    for cls in classes_to_augment:
        cls_indices = np.where(y == cls)[0]
        cls_samples = X[cls_indices]
        needed = min_samples - len(cls_samples)

        if needed <= 0: continue
        
        cls_name = le.inverse_transform([cls])[0]
        print(f"   - Augmenting '{cls_name}': found {len(cls_samples)}, adding {needed}")

        # Generate synthetic samples by adding small noise
        noise_std = np.std(cls_samples, axis=0) * 0.1
        random_indices = np.random.choice(len(cls_samples), size=needed, replace=True)
        base_samples = cls_samples[random_indices]
        noise = np.random.normal(0, noise_std, base_samples.shape)
        synthetic_samples = base_samples + noise
        
        X_aug = np.vstack([X_aug, synthetic_samples])
        y_aug = np.hstack([y_aug, [cls] * needed])
        
    return X_aug, y_aug

def load_and_prepare_data_for_dl(csv_path, is_creator=False):
    """
    Loads and prepares data for multi-class classification, ensuring consistency
    through a master schema and robustly handling minority classes.
    """
    print(f"\n--- Loading and preparing data from: {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Schema and Feature Generation ---
    X, y_labels = _preprocess_and_engineer_features(df.copy())
    
    if is_creator:
        print("   - CREATOR MODE: Generating new master feature list...")
        os.makedirs(SCHEMA_DIR, exist_ok=True)
        master_feature_list = X.columns.tolist()
        with open(MASTER_FEATURE_LIST_PATH, 'w') as f:
            json.dump(master_feature_list, f)
    else:
        if not os.path.exists(MASTER_FEATURE_LIST_PATH):
            raise FileNotFoundError(f"FATAL: Master feature list not found. Please run a cleanup script to regenerate schemas.")
        with open(MASTER_FEATURE_LIST_PATH, 'r') as f:
            master_feature_list = json.load(f)

    # Align features with master list to ensure consistency
    X = X.reindex(columns=master_feature_list, fill_value=0)
    print("   - Data aligned with master feature list.")

    # --- Label Encoding ---
    le = LabelEncoder()
    if is_creator:
        print("   - CREATOR MODE: Fitting new label encoder...")
        y = le.fit_transform(y_labels)
    else:
        class_order_path = os.path.join("validation_data_multi", "class_order.json")
        if not os.path.exists(class_order_path):
            raise FileNotFoundError("FATAL: Canonical class order not found. Please run a cleanup script.")
        with open(class_order_path, 'r') as f:
            class_order = json.load(f)
        
        le.fit(class_order)
        # Filter out any labels not in the canonical class order
        known_labels_mask = y_labels.isin(le.classes_)
        if not all(known_labels_mask):
            unknown = set(y_labels[~known_labels_mask])
            print(f"   - WARNING: Filtering out unknown labels: {unknown}")
            X = X[known_labels_mask]
            y_labels = y_labels[known_labels_mask]
        y = le.transform(y_labels)

    # --- Augmentation, Splitting, and Scaling ---
    X_aug, y_aug = _augment_minority_classes(X.values, y, le, min_samples=10)
    
    # Split data (use stratify if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_aug)
    except ValueError:
        print("   - WARNING: Could not stratify split. Falling back to non-stratified.")
        X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.3, random_state=42)

    # Scale and reshape features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_features = X_train_scaled.shape[1]
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, n_features)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, n_features)
    print(f"   - Data prepared. Train shape: {X_train_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped, y_train, y_test, le, scaler, n_features