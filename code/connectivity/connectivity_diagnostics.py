"""
Diagnostic script to explore network connectivity at different density thresholds
Helps determine what threshold ensures all graphs are connected
"""

import os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("NETWORK CONNECTIVITY DIAGNOSTICS")
print("Testing different density thresholds")

# Get list of all connectivity files
conn_dir = '../../data/connectivity'
conn_files = [f for f in os.listdir(conn_dir) if f.endswith('_connectivity.npy')]

print(f"\nFound {len(conn_files)} connectivity matrices")

# Test different density thresholds
densities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

results = []

print("\nTesting connectivity at different thresholds...")

for density in densities:
    connected_count = 0
    largest_component_sizes = []
    n_components_list = []
    
    for conn_file in conn_files:
        # Load connectivity matrix
        connectivity = np.load(os.path.join(conn_dir, conn_file), allow_pickle=True)
        
        # Get absolute values
        conn_abs = np.abs(connectivity)
        np.fill_diagonal(conn_abs, 0)
        
        # Threshold to create binary network
        n_regions = conn_abs.shape[0]
        triu_indices = np.triu_indices(n_regions, k=1)
        edge_weights = conn_abs[triu_indices]
        
        if len(edge_weights) > 0:
            threshold = np.percentile(edge_weights, (1 - density) * 100)
            
            # Create binary adjacency matrix
            binary_adj = (conn_abs > threshold).astype(int)
            np.fill_diagonal(binary_adj, 0)
            
            # Create graph
            G = nx.from_numpy_array(binary_adj)
            
            # Check connectivity
            is_connected = nx.is_connected(G)
            if is_connected:
                connected_count += 1
            
            # Get component info
            components = list(nx.connected_components(G))
            n_components = len(components)
            largest_component = max(components, key=len)
            
            largest_component_sizes.append(len(largest_component))
            n_components_list.append(n_components)
    
    # Store results
    pct_connected = (connected_count / len(conn_files)) * 100
    avg_largest = np.mean(largest_component_sizes)
    avg_components = np.mean(n_components_list)
    
    results.append({
        'density': density,
        'pct_connected': pct_connected,
        'n_connected': connected_count,
        'n_total': len(conn_files),
        'avg_largest_component': avg_largest,
        'avg_n_components': avg_components
    })
    
    print(f"Density {density:.2f}: {connected_count}/{len(conn_files)} connected ({pct_connected:.1f}%)")

# Create results dataframe
df_results = pd.DataFrame(results)

# Print summary
print("SUMMARY")
print(df_results.to_string(index=False))

# Find optimal threshold
optimal = df_results[df_results['pct_connected'] >= 95].iloc[0] if any(df_results['pct_connected'] >= 95) else None

print("RECOMMENDATIONS")

if optimal is not None:
    print(f"\n At {optimal['density']:.2f} density, {optimal['pct_connected']:.1f}% of graphs are connected")
    print(f"  This threshold would allow group-level averaging!")
else:
    print("\n  No density threshold ensures >95% connectivity")
    print("   May need to work with disconnected graphs or use different approach")

print("\nCurrent approach (15% density):")
current = df_results[df_results['density'] == 0.15].iloc[0]
print(f"  Connected: {current['pct_connected']:.1f}%")
print(f"  Average largest component: {current['avg_largest_component']:.1f} nodes")
print(f"  Average # components: {current['avg_n_components']:.1f}")

# Save results
df_results.to_csv('../../results/connectivity_diagnostics.csv', index=False)
print(f"\n Saved results to: results/connectivity_diagnostics.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Percentage connected vs density
ax = axes[0, 0]
ax.plot(df_results['density'], df_results['pct_connected'], 'o-', linewidth=2, markersize=8)
ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax.axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Current (15%)')
ax.set_xlabel('Network Density', fontsize=12)
ax.set_ylabel('% Graphs Connected', fontsize=12)
ax.set_title('Graph Connectivity vs Density Threshold', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: Average largest component size
ax = axes[0, 1]
ax.plot(df_results['density'], df_results['avg_largest_component'], 'o-', linewidth=2, markersize=8, color='orange')
ax.axhline(y=45, color='green', linestyle='--', alpha=0.5, label='All 45 regions')
ax.axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Current (15%)')
ax.set_xlabel('Network Density', fontsize=12)
ax.set_ylabel('Avg Largest Component Size', fontsize=12)
ax.set_title('Average Largest Component vs Density', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# Plot 3: Number of components
ax = axes[1, 0]
ax.plot(df_results['density'], df_results['avg_n_components'], 'o-', linewidth=2, markersize=8, color='purple')
ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Fully connected')
ax.axvline(x=0.15, color='red', linestyle='--', alpha=0.5, label='Current (15%)')
ax.set_xlabel('Network Density', fontsize=12)
ax.set_ylabel('Avg # Components', fontsize=12)
ax.set_title('Network Fragmentation vs Density', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# Plot 4: Summary bar chart
ax = axes[1, 1]
colors = ['red' if d == 0.15 else 'steelblue' for d in df_results['density']]
bars = ax.bar(df_results['density'], df_results['pct_connected'], color=colors, alpha=0.7)
ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='95% threshold')
ax.set_xlabel('Network Density', fontsize=12)
ax.set_ylabel('% Connected', fontsize=12)
ax.set_title('Summary: Connectivity at Each Threshold', fontsize=14, fontweight='bold')
ax.set_ylim([0, 105])
ax.grid(alpha=0.3, axis='y')
ax.legend()

# Highlight current density
for i, bar in enumerate(bars):
    if df_results['density'].iloc[i] == 0.15:
        bar.set_edgecolor('red')
        bar.set_linewidth(3)

plt.tight_layout()
plt.savefig('../../results/figures/connectivity_diagnostics.png', dpi=300, bbox_inches='tight')
print(f" Saved visualization to: results/figures/connectivity_diagnostics.png")

print("DIAGNOSTICS COMPLETE!")