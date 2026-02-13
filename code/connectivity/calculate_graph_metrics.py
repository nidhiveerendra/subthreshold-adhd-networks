"""
Calculate graph theory metrics from connectivity matrices
Uses NetworkX for graph analysis
"""

import os
import numpy as np
import networkx as nx
from scipy import stats

def calculate_graph_metrics(subject_id,
                           connectivity_dir='../../data/connectivity',
                           output_dir='../../data/graph_metrics',
                           density=0.15):
    """
    Calculate graph theory metrics for a single subject
    
    Parameters:
    subject_id : str
        Subject ID (e.g., 'sub-0010001')
    connectivity_dir : str
        Directory containing connectivity matrices
    output_dir : str
        Directory to save graph metrics
    density : float
        Network density (proportion of edges to keep, default 0.15 = 15%)
    
    Returns:
    dict : Dictionary of graph metrics
    """
    
    print(f"Processing: {subject_id}")
    
    try:
        # 1. Load connectivity matrix
        print("1. Loading connectivity matrix...")
        conn_file = os.path.join(connectivity_dir, f'{subject_id}_connectivity.npy')
        
        if not os.path.exists(conn_file):
            print(f"   ERROR: File not found: {conn_file}")
            return None
        
        connectivity = np.load(conn_file)
        n_regions = connectivity.shape[0]
        
        print(f"   Loaded: {connectivity.shape}")
        print(f"   Regions: {n_regions}")
        
        # 2. Create weighted network (keep all connections)
        print(f"\n2. Creating weighted network...")
        
        # Get absolute values (connection strength)
        conn_abs = np.abs(connectivity)
        
        # Set diagonal to zero
        np.fill_diagonal(conn_abs, 0)
        
        # Create weighted graph
        G_weighted = nx.from_numpy_array(conn_abs)
        
        print(f"   Nodes: {G_weighted.number_of_nodes()}")
        print(f"   Weighted edges: {G_weighted.number_of_edges()}")
        
        # 3. Create binary network by thresholding
        print(f"\n3. Creating binary network (density = {density})...")
        
        # Get upper triangle (network is symmetric)
        triu_indices = np.triu_indices(n_regions, k=1)
        edge_weights = conn_abs[triu_indices]
        
        # Calculate threshold to keep top X% of connections
        threshold = np.percentile(edge_weights, (1 - density) * 100)
        
        # Create binary adjacency matrix
        binary_adj = (conn_abs > threshold).astype(int)
        np.fill_diagonal(binary_adj, 0)
        
        # Create binary graph
        G_binary = nx.from_numpy_array(binary_adj)
        
        actual_density = nx.density(G_binary)
        
        print(f"   Threshold: {threshold:.3f}")
        print(f"   Binary edges: {G_binary.number_of_edges()}")
        print(f"   Actual density: {actual_density:.3f}")
        
        # 4. Calculate global metrics
        print(f"\n4. Calculating graph metrics...")
        
        metrics = {}
        
        # Check if graph is connected
        is_connected = nx.is_connected(G_binary)
        metrics['is_connected'] = is_connected
        
        if not is_connected:
            print("    Graph is not connected - using largest component")
            # Get largest connected component
            largest_cc = max(nx.connected_components(G_binary), key=len)
            G_binary = G_binary.subgraph(largest_cc).copy()
            print(f"   Largest component size: {len(largest_cc)} nodes")
        
        # CLUSTERING COEFFICIENT
        print("   - Clustering coefficient...")
        metrics['clustering_coefficient'] = nx.average_clustering(G_binary)
        metrics['clustering_weighted'] = nx.average_clustering(G_weighted, weight='weight')
        
        # PATH LENGTH
        print("   - Characteristic path length...")
        metrics['path_length'] = nx.average_shortest_path_length(G_binary)
        
        # GLOBAL EFFICIENCY
        print("   - Global efficiency...")
        metrics['global_efficiency'] = nx.global_efficiency(G_binary)
        metrics['global_efficiency_weighted'] = nx.global_efficiency(G_weighted)
        
        # MODULARITY
        print("   - Modularity...")
        # Use Louvain algorithm for community detection
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G_binary)
        metrics['modularity'] = community.modularity(G_binary, communities)
        metrics['n_communities'] = len(communities)
        
        # SMALL-WORLDNESS
        print("   - Small-worldness...")
        # Generate random network with same degree distribution
        degree_sequence = [d for n, d in G_binary.degree()]
        n_random_networks = 10  # Use 10 random networks for speed
        
        random_C = []
        random_L = []
        
        for _ in range(n_random_networks):
            # Create random graph with same degree sequence
            G_random = nx.configuration_model(degree_sequence)
            G_random = nx.Graph(G_random)  # Remove multi-edges
            G_random.remove_edges_from(nx.selfloop_edges(G_random))  # Remove self-loops
            
            if nx.is_connected(G_random):
                random_C.append(nx.average_clustering(G_random))
                random_L.append(nx.average_shortest_path_length(G_random))
        
        if len(random_C) > 0:
            C_random = np.mean(random_C)
            L_random = np.mean(random_L)
            
            # Small-worldness = (C/C_random) / (L/L_random)
            metrics['small_worldness'] = (metrics['clustering_coefficient'] / C_random) / (metrics['path_length'] / L_random)
        else:
            metrics['small_worldness'] = np.nan
            print("    Could not calculate small-worldness (random graphs disconnected)")
        
        # DEGREE STATISTICS
        degrees = [d for n, d in G_binary.degree()]
        metrics['mean_degree'] = np.mean(degrees)
        metrics['std_degree'] = np.std(degrees)
        
        # ASSORTATIVITY
        print("   - Assortativity...")
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G_binary)
        
        # 5. Save metrics
        print(f"\n5. Saving metrics...")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{subject_id}_metrics.npy')
        np.save(output_file, metrics)
        
        # 6. Print summary
        print(f"\n6. RESULTS SUMMARY:")
        print(f"   Clustering Coefficient: {metrics['clustering_coefficient']:.4f}")
        print(f"   Path Length: {metrics['path_length']:.4f}")
        print(f"   Global Efficiency: {metrics['global_efficiency']:.4f}")
        print(f"   Modularity: {metrics['modularity']:.4f}")
        print(f"   Small-worldness: {metrics.get('small_worldness', 'N/A'):.4f}")
        print(f"   Mean Degree: {metrics['mean_degree']:.2f}")
        print(f"   Communities: {metrics['n_communities']}")
        
        print(f" SUCCESS: {subject_id}")
        
        return metrics
        
    except Exception as e:
        print(f"\n ERROR processing {subject_id}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test on first subject
    test_subject = 'sub-0010001'
    
    print("GRAPH THEORY METRICS - SINGLE SUBJECT TEST")
    print("Using proportional thresholding (15% density)")
    
    # Calculate metrics
    metrics = calculate_graph_metrics(test_subject, density=0.15)
    
    if metrics:
        print("\n GRAPH METRICS CALCULATION SUCCESSFUL!")
        print("\nReady to batch process all 201 subjects!")
        print("\nKey metrics calculated:")
        print("   Clustering coefficient (local integration)")
        print("   Path length (global communication)")
        print("   Global efficiency (network efficiency)")
        print("   Modularity (network segregation)")
        print("   Small-worldness (optimal organization)")
        print("   Degree statistics")
        print("   Assortativity (hub organization)")
    else:
        print("\n TEST FAILED - Debug needed")