import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from matplotlib.patches import Patch

# --- Matplotlib settings for professional-looking graphs ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'font.family': 'serif'
})

REPORT_DIR = "DL_Final_Report"

def plot_communication_efficiency(df):
    """Generates Graph 1: Communication Efficiency of Sparsification."""
    if df.empty:
        print("WARNING: communication_log.csv is empty. Skipping Graph 1.")
        return

    # Use all available data points up to 15 for a representative sample
    sample_df = df.sample(n=min(15, len(df)), random_state=42).sort_values('ClientID')
    
    if 'FullUpdateSizeKB' not in df.columns or 'SparseUpdateSizeKB' not in df.columns:
        print("ERROR: communication_log.csv is missing required columns. Skipping Graph 1.")
        return

    # Calculate overall reduction for the title
    avg_full_size = df['FullUpdateSizeKB'].mean()
    avg_sparse_size = df['SparseUpdateSizeKB'].mean()
    if avg_full_size == 0:
        reduction_percent = 0
    else:
        reduction_percent = (1 - (avg_sparse_size / avg_full_size)) * 100

    plt.figure(figsize=(16, 8))
    bar_width = 0.35
    index = np.arange(len(sample_df))

    plt.bar(index, sample_df['FullUpdateSizeKB'], color='skyblue', width=bar_width, edgecolor='grey', label='Full Update')
    plt.bar(index + bar_width, sample_df['SparseUpdateSizeKB'], color='salmon', width=bar_width, edgecolor='grey', label=f'Sparse Update (Top {100-reduction_percent:.0f}%)')

    plt.xlabel('Sample Client ID', fontweight='bold')
    plt.ylabel('Update Size (KB)', fontweight='bold')
    plt.title(f'Communication Efficiency of Top-K Sparsification\n(Average Reduction: {reduction_percent:.1f}%)', fontweight='bold', fontsize=20)
    plt.xticks(index + bar_width / 2, sample_df['ClientID'])
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(REPORT_DIR, "graph1_communication_efficiency.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graph 1 (Communication) saved to {output_path}")

def plot_stage1_hard_thresholds(df, data_poison_ids, model_poison_ids):
    """Generates Graph 4 & 5 (Accuracy & Norm Checks) in one function for consistent sampling."""
    if df.empty:
        print("WARNING: gatekeeper_log.csv is empty. Skipping Stage 1 defense graphs.")
        return
        
    df['ClientID'] = df['ClientID'].astype(int)
    all_poison_ids = set(data_poison_ids + model_poison_ids)
    
    # Select a representative sample of clients
    honest_clients = df[~df['ClientID'].isin(all_poison_ids)].sample(n=min(6, len(df[~df['ClientID'].isin(all_poison_ids)])), random_state=1)
    data_poison_clients = df[df['ClientID'].isin(data_poison_ids)].head(2)
    model_poison_clients = df[df['ClientID'].isin(model_poison_ids)].head(2)
    
    sample_df = pd.concat([honest_clients, data_poison_clients, model_poison_clients]).sort_values('ClientID').set_index('ClientID')
    
    if sample_df.empty:
        print("WARNING: Could not find any of the specified clients in gatekeeper_log.csv. Skipping Stage 1 graphs.")
        return
        
    colors = ['#2ca02c' if (id not in all_poison_ids) else '#d62728' for id in sample_df.index]
    
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.85, label='Honest Client'),
        Patch(facecolor='#d62728', alpha=0.85, label='Malicious Client'),
        plt.Line2D([0], [0], color='darkred', linestyle='--', lw=3, label='Rejection Threshold')
    ]

    # --- GRAPH 4: Accuracy Threshold Check ---
    plt.figure(figsize=(12, 7))
    plt.bar(sample_df.index.astype(str), sample_df['Accuracy'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=0.50, color='darkred', linestyle='--', linewidth=3, label='Rejection Threshold (Accuracy < 0.50)')
    
    plt.title('Stage 1a: Validation Accuracy Check (Hard Threshold)', fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('Validation Accuracy', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    output_path_acc = os.path.join(REPORT_DIR, "graph4_stage1_accuracy_check.png")
    plt.savefig(output_path_acc, dpi=300)
    plt.close()
    print(f"Graph 4 (Accuracy Check) saved to {output_path_acc}")

    # --- GRAPH 5: Update Norm Check ---
    plt.figure(figsize=(12, 7))
    plt.bar(sample_df.index.astype(str), sample_df['UpdateNorm'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=100.0, color='darkred', linestyle='--', linewidth=3, label='Rejection Threshold (Norm > 100)')
    
    plt.title('Stage 1b: Update Norm Check (Detecting Scaling Attacks)', fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('Update Vector Norm (Log Scale)', fontweight='bold')
    plt.yscale('log') # Use log scale to show the massive difference

    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    output_path_norm = os.path.join(REPORT_DIR, "graph5_stage1_norm_check.png")
    plt.savefig(output_path_norm, dpi=300)
    plt.close()
    print(f"Graph 5 (Norm Check) saved to {output_path_norm}")


def plot_fltrust_defense(df, attack_type, data_poison_ids, model_poison_ids):
    """Generates Graph 2 (Data Poisoning) or Graph 3 (Model Poisoning) using FLTrust Score."""
    if df.empty or 'TrustScore' not in df.columns:
        print(f"WARNING: 'TrustScore' column not found or log is empty. Skipping FLTrust graph for {attack_type}.")
        return

    df['ClientID'] = df['ClientID'].astype(int)
    all_poison_ids = set(data_poison_ids + model_poison_ids)

    if attack_type == 'data_poison':
        if not data_poison_ids: return
        poisoned_ids_for_this_graph = set(data_poison_ids)
        graph_num = 2
        title = 'Stage 2: FLTrust Defense Against Data Poisoning'
        sample_ids_h = [c for c in df['ClientID'] if c not in all_poison_ids]
        sample_ids_m = [c for c in df['ClientID'] if c in poisoned_ids_for_this_graph]
        sample_ids = sample_ids_h[:6] + sample_ids_m[:2]
    elif attack_type == 'model_poison':
        if not model_poison_ids: return
        poisoned_ids_for_this_graph = set(model_poison_ids)
        graph_num = 3
        title = 'Stage 2: FLTrust Defense Against Model Poisoning'
        sample_ids_h = [c for c in df['ClientID'] if c not in all_poison_ids]
        sample_ids_m = [c for c in df['ClientID'] if c in poisoned_ids_for_this_graph]
        sample_ids = sample_ids_h[:6] + sample_ids_m[:2]
    else: 
        return

    sample_df = df[df['ClientID'].isin(sample_ids)].sort_values('ClientID').set_index('ClientID')
    if sample_df.empty:
        print(f"WARNING: No matching clients found for {attack_type} graph. Skipping.")
        return
        
    colors = ['#2ca02c' if (id not in poisoned_ids_for_this_graph) else '#d62728' for id in sample_df.index]
    
    plt.figure(figsize=(12, 7))
    plt.bar(sample_df.index.astype(str), sample_df['TrustScore'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=0.01, color='darkred', linestyle='--', linewidth=2, label='Filter Threshold (Score ≈ 0)')
    
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('FLTrust Score (ReLU of Cosine Similarity)', fontweight='bold')

    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.85, label='Honest Client (Aligned Direction)'),
        Patch(facecolor='#d62728', alpha=0.85, label='Malicious Client (Opposite/Noisy Direction)'),
        plt.Line2D([0], [0], color='darkred', linestyle='--', lw=2, label='Filter Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    output_path = os.path.join(REPORT_DIR, f"graph{graph_num}_fltrust_{attack_type}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graph {graph_num} (FLTrust vs {attack_type}) saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate graphs for the final academic report.")
    parser.add_argument('--data_poison_ids', type=int, nargs='*', default=[], help="List of client IDs that are data poisoners.")
    parser.add_argument('--model_poison_ids', type=int, nargs='*', default=[], help="List of client IDs that are model poisoners.")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("\n--- Generating Visualizations for Final Report ---")

    comm_log_path = os.path.join(REPORT_DIR, "communication_log.csv")
    if os.path.exists(comm_log_path):
        plot_communication_efficiency(pd.read_csv(comm_log_path))
    else:
        print("WARNING: 'communication_log.csv' not found. Skipping Communication Efficiency graph.")
        
    gatekeeper_log_path = os.path.join(REPORT_DIR, "gatekeeper_log.csv")
    if os.path.exists(gatekeeper_log_path):
        gatekeeper_df = pd.read_csv(gatekeeper_log_path)
        
        # Plot FLTrust Defense Graphs
        plot_fltrust_defense(gatekeeper_df, 'data_poison', args.data_poison_ids, args.model_poison_ids)
        plot_fltrust_defense(gatekeeper_df, 'model_poison', args.data_poison_ids, args.model_poison_ids)
        
        # Plot Stage 1 Defense Graphs (Accuracy & Norm)
        plot_stage1_hard_thresholds(gatekeeper_df, args.data_poison_ids, args.model_poison_ids)
    else:
        print("WARNING: 'gatekeeper_log.csv' not found. Skipping all defense-related graphs.")

    print("\n--- Graph generation complete. Check the 'DL_Final_Report' folder. ---")

if __name__ == "__main__":
    main()