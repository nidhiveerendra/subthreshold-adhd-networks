"""
Group-Level Network Analysis
Create averaged connectivity matrices for each group and compare
Following Dr. Anandakrishnan's guidance
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("GROUP-LEVEL NETWORK ANALYSIS")
print("Creating averaged networks for subthreshold vs diagnosed groups")

# 1. Load phenotypic data to get group assignments
print("\n1. Loading phenotypic data...")
df_pheno = pd.read_csv('../../data/phenotypic/NYU_phenotypic.csv')
df_pheno['subject_id'] = df_pheno['ScanDir ID'].apply(lambda x: f'sub-{x:07d}')

# Create groups
index_col = 'ADHD Index'
df_pheno['Group'] = 'Unknown'
valid_scores = df_pheno[index_col] != -999

df_pheno.loc[valid_scores & (df_pheno[index_col] >= 40) & (df_pheno[index_col] < 60), 'Group'] = 'Subthreshold'
df_pheno.loc[valid_scores & (df_pheno[index_col] >= 60), 'Group'] = 'Diagnosed'

# Get subject lists for each group
subthreshold_subjects = df_pheno[df_pheno['Group'] == 'Subthreshold']['subject_id'].tolist()
diagnosed_subjects = df_pheno[df_pheno['Group'] == 'Diagnosed']['subject_id'].tolist()

print(f"   Subthreshold subjects: {len(subthreshold_subjects)}")
print(f"   Diagnosed subjects: {len(diagnosed_subjects)}")

# 2. Load and average connectivity matrices for each group
print("\n2. Creating group-level connectivity matrices...")

def load_group_matrices(subject_list, conn_dir='../../data/connectivity'):
    """Load all connectivity matrices for a group"""
    matrices = []
    loaded_subjects = []
    expected_shape = None
    
    for subject_id in subject_list:
        conn_file = os.path.join(conn_dir, f'{subject_id}_connectivity.npy')
        if os.path.exists(conn_file):
            matrix = np.load(conn_file, allow_pickle=True)
            
            # Set expected shape from first matrix
            if expected_shape is None:
                expected_shape = matrix.shape
            
            # Only include matrices with correct shape
            if matrix.shape == expected_shape:
                matrices.append(matrix)
                loaded_subjects.append(subject_id)
            else:
                print(f"   Warning: Skipping {subject_id} - shape {matrix.shape} != {expected_shape}")
    
    return np.array(matrices), loaded_subjects

# Load subthreshold matrices
print("   Loading subthreshold connectivity matrices...")
subthreshold_matrices, sub_loaded = load_group_matrices(subthreshold_subjects)
print(f"   Loaded: {len(sub_loaded)}/{len(subthreshold_subjects)}")

# Load diagnosed matrices
print("   Loading diagnosed connectivity matrices...")
diagnosed_matrices, diag_loaded = load_group_matrices(diagnosed_subjects)
print(f"   Loaded: {len(diag_loaded)}/{len(diagnosed_subjects)}")

# 3. Average matrices within each group
print("\n3. Computing group-averaged connectivity matrices...")

# Average across subjects (axis 0)
subthreshold_avg = np.mean(subthreshold_matrices, axis=0)
diagnosed_avg = np.mean(diagnosed_matrices, axis=0)

print(f"   Subthreshold average matrix: {subthreshold_avg.shape}")
print(f"   Data range: [{subthreshold_avg.min():.3f}, {subthreshold_avg.max():.3f}]")

print(f"   Diagnosed average matrix: {diagnosed_avg.shape}")
print(f"   Data range: [{diagnosed_avg.min():.3f}, {diagnosed_avg.max():.3f}]")

# 4. Calculate difference matrix
difference_matrix = subthreshold_avg - diagnosed_avg

print(f"\n4. Difference matrix (Subthreshold - Diagnosed):")
print(f"   Max difference: {difference_matrix.max():.3f}")
print(f"   Min difference: {difference_matrix.min():.3f}")
print(f"   Mean absolute difference: {np.abs(difference_matrix).mean():.3f}")

# 5. Save group matrices
print("\n5. Saving group-level matrices...")
os.makedirs('../../data/group_networks', exist_ok=True)

np.save('../../data/group_networks/subthreshold_group_matrix.npy', subthreshold_avg)
np.save('../../data/group_networks/diagnosed_group_matrix.npy', diagnosed_avg)
np.save('../../data/group_networks/difference_matrix.npy', difference_matrix)

print("   Saved: subthreshold_group_matrix.npy")
print("    Saved: diagnosed_group_matrix.npy")
print("    Saved: difference_matrix.npy")

# 6. Calculate graph metrics on group networks
print("\n6. Calculating graph metrics on group-level networks...")

def calculate_group_graph_metrics(connectivity_matrix, density=0.15):
    """Calculate graph metrics on a group-averaged connectivity matrix"""
    
    # Get absolute values
    conn_abs = np.abs(connectivity_matrix)
    np.fill_diagonal(conn_abs, 0)
    
    # Threshold to create binary network
    n_regions = conn_abs.shape[0]
    triu_indices = np.triu_indices(n_regions, k=1)
    edge_weights = conn_abs[triu_indices]
    threshold = np.percentile(edge_weights, (1 - density) * 100)
    
    binary_adj = (conn_abs > threshold).astype(int)
    np.fill_diagonal(binary_adj, 0)
    
    # Create graph
    G = nx.from_numpy_array(binary_adj)
    
    # Ensure connected
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Calculate metrics
    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'clustering': nx.average_clustering(G),
        'path_length': nx.average_shortest_path_length(G),
        'global_efficiency': nx.global_efficiency(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }
    
    # Modularity
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    metrics['modularity'] = community.modularity(G, communities)
    metrics['n_communities'] = len(communities)
    
    # Small-worldness
    degree_sequence = [d for n, d in G.degree()]
    G_random = nx.configuration_model(degree_sequence)
    G_random = nx.Graph(G_random)
    G_random.remove_edges_from(nx.selfloop_edges(G_random))
    
    if nx.is_connected(G_random):
        C_random = nx.average_clustering(G_random)
        L_random = nx.average_shortest_path_length(G_random)
        metrics['small_worldness'] = (metrics['clustering'] / C_random) / (metrics['path_length'] / L_random)
    else:
        metrics['small_worldness'] = np.nan
    
    return metrics, G

# Calculate for subthreshold group
print("   Subthreshold group network:")
sub_metrics, G_sub = calculate_group_graph_metrics(subthreshold_avg)
for key, val in sub_metrics.items():
    print(f"      {key}: {val}")

# Calculate for diagnosed group
print("\n   Diagnosed group network:")
diag_metrics, G_diag = calculate_group_graph_metrics(diagnosed_avg)
for key, val in diag_metrics.items():
    print(f"      {key}: {val}")

# 7. Compare metrics
print("\n7. GROUP COMPARISON:")
print(f"{'Metric':<25} {'Subthreshold':<15} {'Diagnosed':<15} {'Difference'}")

for key in sub_metrics.keys():
    if key in ['n_nodes', 'n_edges', 'n_communities']:
        print(f"{key:<25} {sub_metrics[key]:<15} {diag_metrics[key]:<15} {sub_metrics[key] - diag_metrics[key]}")
    else:
        diff = sub_metrics[key] - diag_metrics[key]
        pct = (diff / diag_metrics[key] * 100) if diag_metrics[key] != 0 else 0
        print(f"{key:<25} {sub_metrics[key]:<15.4f} {diag_metrics[key]:<15.4f} {diff:+.4f} ({pct:+.1f}%)")

# 8. Create visualizations
print("\n8. Creating visualizations...")

# Figure 1: Group connectivity matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subthreshold
ax = axes[0]
im = ax.imshow(subthreshold_avg, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title(f'Subthreshold Group (N={len(sub_loaded)})', fontsize=14, fontweight='bold')
ax.set_xlabel('Brain Region')
ax.set_ylabel('Brain Region')
plt.colorbar(im, ax=ax, label='Z-transformed Correlation')

# Diagnosed
ax = axes[1]
im = ax.imshow(diagnosed_avg, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title(f'Diagnosed Group (N={len(diag_loaded)})', fontsize=14, fontweight='bold')
ax.set_xlabel('Brain Region')
ax.set_ylabel('Brain Region')
plt.colorbar(im, ax=ax, label='Z-transformed Correlation')

# Difference
ax = axes[2]
im = ax.imshow(difference_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, center=0)
ax.set_title('Difference (Sub - Diag)', fontsize=14, fontweight='bold')
ax.set_xlabel('Brain Region')
ax.set_ylabel('Brain Region')
plt.colorbar(im, ax=ax, label='Difference')

plt.tight_layout()
plt.savefig('../../results/figures/group_connectivity_matrices.png', dpi=300, bbox_inches='tight')
print("    Saved: group_connectivity_matrices.png")

# Figure 2: Metric comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

metrics_to_plot = ['clustering', 'path_length', 'global_efficiency', 'modularity', 'small_worldness']
sub_values = [sub_metrics[m] for m in metrics_to_plot]
diag_values = [diag_metrics[m] for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35

ax.bar(x - width/2, sub_values, width, label='Subthreshold', color='gold', alpha=0.8)
ax.bar(x + width/2, diag_values, width, label='Diagnosed', color='red', alpha=0.8)

ax.set_xlabel('Graph Metric', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Group-Level Network Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../../results/figures/group_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("    Saved: group_metrics_comparison.png")

# 9. Save results
print("\n9. Saving results summary...")

results = {
    'subthreshold_n': len(sub_loaded),
    'diagnosed_n': len(diag_loaded),
    'subthreshold_metrics': sub_metrics,
    'diagnosed_metrics': diag_metrics
}

# Save as text file
with open('../../results/group_network_comparison.txt', 'w') as f:
    f.write("GROUP-LEVEL NETWORK COMPARISON\n")
    f.write(f"Subthreshold subjects: {len(sub_loaded)}\n")
    f.write(f"Diagnosed subjects: {len(diag_loaded)}\n\n")
    
    f.write("GRAPH METRICS COMPARISON:\n")
    f.write(f"{'Metric':<25} {'Subthreshold':<15} {'Diagnosed':<15} {'Difference'}\n")
    
    for key in sub_metrics.keys():
        if key in ['n_nodes', 'n_edges', 'n_communities']:
            f.write(f"{key:<25} {sub_metrics[key]:<15} {diag_metrics[key]:<15} {sub_metrics[key] - diag_metrics[key]}\n")
        else:
            diff = sub_metrics[key] - diag_metrics[key]
            pct = (diff / diag_metrics[key] * 100) if diag_metrics[key] != 0 else 0
            f.write(f"{key:<25} {sub_metrics[key]:<15.4f} {diag_metrics[key]:<15.4f} {diff:+.4f} ({pct:+.1f}%)\n")

print("    Saved: group_network_comparison.txt")

print("GROUP-LEVEL NETWORK ANALYSIS COMPLETE!")
print("\nFiles ready to upload to Google Drive:")
print("  - subthreshold_group_matrix.npy")
print("  - diagnosed_group_matrix.npy")
print("  - group_connectivity_matrices.png")
print("  - group_metrics_comparison.png")
print("  - group_network_comparison.txt")