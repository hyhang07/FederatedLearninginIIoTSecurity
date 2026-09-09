import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

# --- Matplotlib settings for professional-looking graphs ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

REPORT_DIR = "DL_Final_Report_Multi"

def plot_communication_efficiency(df):
    """Generates Graph 3: Communication Efficiency of Sparsification."""
    if df.empty: return
    sample_df = df.sample(n=min(15, len(df)), random_state=42).sort_values('ClientID')
    avg_full_size = df['FullUpdateSizeKB'].mean()
    avg_sparse_size = df['SparseUpdateSizeKB'].mean()
    reduction_percent = (1 - (avg_sparse_size / avg_full_size)) * 100

    plt.figure(figsize=(16, 8))
    bar_width = 0.35
    r1 = np.arange(len(sample_df))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, sample_df['FullUpdateSizeKB'], color='skyblue', width=bar_width, edgecolor='grey', label='Full Update')
    plt.bar(r2, sample_df['SparseUpdateSizeKB'], color='salmon', width=bar_width, edgecolor='grey', label=f'Sparse Update (Top {100-reduction_percent:.0f}%)')

    plt.xlabel('Sample Client ID', fontweight='bold')
    plt.ylabel('Update Size (KB)', fontweight='bold')
    plt.title(f'Communication Efficiency of Top-K Sparsification\n(Average Reduction: {reduction_percent:.1f}%)', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(sample_df))], sample_df['ClientID'])
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(REPORT_DIR, "graph3_communication_efficiency.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graph 3 saved to {output_path}")

def plot_gatekeeper_defense(df, attack_type, data_poison_ids, model_poison_ids):
    """Generates Graph 4 and Graph 5 using FLTrust Score"""
    if df.empty or 'TrustScore' not in df.columns: return

    df['ClientID'] = df['ClientID'].astype(int)
    df = df.sort_values('ClientID').reset_index()

    if attack_type == 'data_poison':
        if not data_poison_ids: return
        poisoned_ids = data_poison_ids
        graph_num = 4
        title = 'Stage 2: FLTrust Defense Against Data Poisoning (Label Flipping)'
        sample_ids = [c for c in df['ClientID'] if c not in model_poison_ids and c not in data_poison_ids][:6] + poisoned_ids[:2]
    elif attack_type == 'model_poison':
        if not model_poison_ids: return
        poisoned_ids = model_poison_ids
        graph_num = 5
        title = 'Stage 2: FLTrust Defense Against Model Poisoning (Gradient Scaling/Noise)'
        sample_ids = [c for c in df['ClientID'] if c not in data_poison_ids and c not in model_poison_ids][:6] + poisoned_ids[:2]
    else: return

    sample_df = df[df['ClientID'].isin(sample_ids)].set_index('ClientID')
    colors =['#2ca02c' if (id not in poisoned_ids) else '#d62728' for id in sample_df.index]
    
    plt.figure(figsize=(12, 7))
    plt.bar(sample_df.index.astype(str), sample_df['TrustScore'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=0.01, color='darkred', linestyle='--', linewidth=2, label='Filter Threshold (Score ≈ 0)')
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('FLTrust Score (ReLU of Cosine Similarity)', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements =[Patch(facecolor='#2ca02c', alpha=0.85, label='Honest Client (Aligned Direction)'),
                       Patch(facecolor='#d62728', alpha=0.85, label='Malicious Client (Opposite/Noisy Direction)'),
                       plt.Line2D([0],[0], color='darkred', linestyle='--', lw=2, label='Filter Threshold')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"graph{graph_num}_gatekeeper_{attack_type}.png"), dpi=300)
    plt.close()
    print(f"Graph {graph_num} saved to DL_Final_Report_Multi/graph{graph_num}_gatekeeper_{attack_type}.png")

def plot_stage1_hard_thresholds(df, data_poison_ids, model_poison_ids):
    """Generates Graph 6 (Accuracy Check) and Graph 7 (Norm Check) to prove Two-Stage Verification"""
    if df.empty: return
    df['ClientID'] = df['ClientID'].astype(int)
    df = df.sort_values('ClientID').reset_index()

    all_poison_ids = data_poison_ids + model_poison_ids
    if not all_poison_ids: return

    # We select a mix of honest clients and poisoned clients to show the contrast
    sample_ids =[c for c in df['ClientID'] if c not in all_poison_ids][:6] + data_poison_ids[:2] + model_poison_ids[:2]
    sample_df = df[df['ClientID'].isin(sample_ids)].set_index('ClientID')
    colors =['#2ca02c' if (id not in all_poison_ids) else '#d62728' for id in sample_df.index]

    # --- GRAPH 6: Accuracy Threshold Check ---
    plt.figure(figsize=(12, 7))
    plt.bar(sample_df.index.astype(str), sample_df['Accuracy'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=0.50, color='darkred', linestyle='--', linewidth=3, label='Rejection Threshold (Accuracy < 0.50)')
    
    plt.title('Stage 1a: Validation Accuracy Check (Hard Threshold)', fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('Validation Accuracy', fontweight='bold')
    plt.ylim(0, 1.05)

    from matplotlib.patches import Patch
    legend_elements =[Patch(facecolor='#2ca02c', alpha=0.85, label='Honest Client'),
                       Patch(facecolor='#d62728', alpha=0.85, label='Malicious Client'),
                       plt.Line2D([0], [0], color='darkred', linestyle='--', lw=3, label='Rejection Threshold')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    output_path_6 = os.path.join(REPORT_DIR, "graph6_stage1_accuracy_check.png")
    plt.savefig(output_path_6, dpi=300)
    plt.close()
    print(f"Graph 6 saved to {output_path_6}")

    # --- GRAPH 7: Update Norm Check ---
    plt.figure(figsize=(12, 7))
    # We use Log Scale for Norm because scaling attacks make the norm huge (e.g., 16000 vs 16)
    plt.bar(sample_df.index.astype(str), sample_df['UpdateNorm'], color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(y=100.0, color='darkred', linestyle='--', linewidth=3, label='Rejection Threshold (Norm > 100)')
    
    plt.title('Stage 1b: Update Norm Check (Detecting Scaling Attacks)', fontweight='bold', pad=15)
    plt.xlabel('Client ID', fontweight='bold')
    plt.ylabel('Update Vector Norm (Log Scale)', fontweight='bold')
    plt.yscale('log') # Log scale to show the massive difference

    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    output_path_7 = os.path.join(REPORT_DIR, "graph7_stage1_norm_check.png")
    plt.savefig(output_path_7, dpi=300)
    plt.close()
    print(f"Graph 7 saved to {output_path_7}")


def main():
    parser = argparse.ArgumentParser(description="Generate graphs for the final academic report.")
    parser.add_argument('--data_poison_ids', type=int, nargs='*', default=[], help="List of data poisoner IDs.")
    parser.add_argument('--model_poison_ids', type=int, nargs='*', default=[])
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    comm_log_path = os.path.join(REPORT_DIR, "communication_log.csv")
    if os.path.exists(comm_log_path):
        plot_communication_efficiency(pd.read_csv(comm_log_path))
        
    gatekeeper_log_path = os.path.join(REPORT_DIR, "gatekeeper_log.csv")
    if os.path.exists(gatekeeper_log_path):
        gatekeeper_df = pd.read_csv(gatekeeper_log_path)
        if 'TrustScore' in gatekeeper_df.columns:
            plot_gatekeeper_defense(gatekeeper_df, 'data_poison', args.data_poison_ids, args.model_poison_ids)
            plot_gatekeeper_defense(gatekeeper_df, 'model_poison', args.data_poison_ids, args.model_poison_ids)
            # [NEW] Generate Stage 1 Hard Threshold Graphs
            plot_stage1_hard_thresholds(gatekeeper_df, args.data_poison_ids, args.model_poison_ids)
        else:
            print("WARNING: 'TrustScore' column not found in gatekeeper log. Cannot generate defense graphs.")

    print("\nGraph generation complete.")

if __name__ == "__main__":
    main()