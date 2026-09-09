import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

SCHEMA_DIR = "Data_Schema"
MASTER_FEATURE_LIST_PATH = os.path.join(SCHEMA_DIR, 'master_feature_list.json')
CLASS_ORDER_PATH = os.path.join(SCHEMA_DIR, 'class_order.json')
SHAPE_PATH = os.path.join(SCHEMA_DIR, 'model_input_shape.txt')

TARGET_CLASSES =[
    'Normal', 'Scanning', 'DDoS', 'DoS', 'XSS', 
    'Password', 'Backdoor', 'Injection', 'Ransomware'
]

def map_ton_iot_labels(df):
    if 'type' not in df.columns: raise ValueError("Must contain 'type' column.")
    def map_label(val):
        val_str = str(val).lower().strip()
        for c in TARGET_CLASSES:
            if c.lower() in val_str: return c
        return 'Other'
    df['Attack_type'] = df['type'].apply(map_label)
    df = df[df['Attack_type'].isin(TARGET_CLASSES)].copy()
    df.drop(columns=['type'], inplace=True)
    return df

def process_features(df, is_creator=False):
    y_labels = df['Attack_type']
    X = df.drop(columns=['Attack_type'])
    
    # 剔除破坏神经网络的高基数文本字符串列
    obj_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in obj_cols:
        if X[col].nunique() > 20:
            X.drop(columns=[col], inplace=True)
            
    X.fillna(0, inplace=True)
    
    clean_categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(clean_categorical_cols) > 0:
        X = pd.get_dummies(X, columns=clean_categorical_cols)
    
    if is_creator:
        os.makedirs(SCHEMA_DIR, exist_ok=True)
        with open(MASTER_FEATURE_LIST_PATH, 'w') as f: json.dump(X.columns.tolist(), f)
    else:
        with open(MASTER_FEATURE_LIST_PATH, 'r') as f: master_features = json.load(f)
        X = X.reindex(columns=master_features, fill_value=0).astype('float32')
        
    return X, y_labels

def load_and_prepare_data_for_dl(csv_path, is_creator=False):
    df = pd.read_csv(csv_path, low_memory=False)
    df = map_ton_iot_labels(df)
    X, y_labels = process_features(df, is_creator=is_creator)
    
    le = LabelEncoder()
    if is_creator:
        le.fit(TARGET_CLASSES)
        with open(CLASS_ORDER_PATH, 'w') as f: json.dump(list(le.classes_), f)
    else:
        with open(CLASS_ORDER_PATH, 'r') as f: le.classes_ = np.array(json.load(f))
        
    y = le.transform(y_labels)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_features = X_train_scaled.shape[1]
    if is_creator:
        with open(SHAPE_PATH, 'w') as f: f.write(f"1,{n_features}")
            
    return X_train_scaled.reshape(-1, 1, n_features), X_test_scaled.reshape(-1, 1, n_features), y_train, y_test, le, scaler, n_features