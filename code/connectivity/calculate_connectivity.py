"""
Calculate functional connectivity matrices from time series data
Computes correlation between all pairs of brain regions
"""

import os
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_connectivity_single(subject_id,
                                  timeseries_dir='../../data/timeseries',
                                  output_dir='../../data/connectivity'):
    """
    Calculate connectivity matrix for a single subject
    
    Parameters:
    subject_id : str
        Subject ID (e.g., 'sub-0010001')
    timeseries_dir : str
        Directory containing time series files
    output_dir : str
        Directory to save connectivity matrices
    
    Returns:
    bool : True if successful, False otherwise
    """
    print(f"Processing: {subject_id}")
    
    try:
        # 1. Load time series
        print("1. Loading time series...")
        ts_file = os.path.join(timeseries_dir, f'{subject_id}_timeseries.npy')
        
        if not os.path.exists(ts_file):
            print(f"   ERROR: File not found: {ts_file}")
            return False
        
        time_series = np.load(ts_file)
        n_timepoints, n_regions = time_series.shape
        
        print(f"   Loaded: {time_series.shape}")
        print(f"   Timepoints: {n_timepoints}")
        print(f"   Regions: {n_regions}")
        
        # 2. Calculate correlation matrix
        print("\n2. Calculating correlation matrix...")
        correlation_matrix = np.corrcoef(time_series.T)
        
        print(f"   Correlation matrix shape: {correlation_matrix.shape}")
        print(f"   ({n_regions} regions {n_regions} regions)")
        
        # 3. Fisher z-transformation
        print("\n3. Applying Fisher z-transformation...")
        # Avoid log(0) by clipping values slightly away from ±1
        correlation_matrix_clipped = np.clip(correlation_matrix, -0.9999, 0.9999)
        z_matrix = np.arctanh(correlation_matrix_clipped)
        
        print(f"   Z-transformed matrix range: [{z_matrix.min():.3f}, {z_matrix.max():.3f}]")
        
        # 4. Set diagonal to zero (self-correlations not meaningful)
        np.fill_diagonal(z_matrix, 0)
        
        # 5. Quality checks
        print("\n4. Quality Checks:")
        print(f"    No NaN values: {not np.isnan(z_matrix).any()}")
        print(f"    No Inf values: {not np.isinf(z_matrix).any()}")
        print(f"    Symmetric matrix: {np.allclose(z_matrix, z_matrix.T)}")
        print(f"    Diagonal is zero: {np.allclose(np.diag(z_matrix), 0)}")
        
        # 6. Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 7. Save connectivity matrix
        output_file = os.path.join(output_dir, f'{subject_id}_connectivity.npy')
        np.save(output_file, z_matrix)
        print(f"\n5. Saved connectivity matrix to: {output_file}")
        
        # 8. Show sample values
        print("\n6. Sample connectivity values (first 5×5):")
        print(z_matrix[:5, :5])
        
        print(f" SUCCESS: {subject_id}")
        
        return True
        
    except Exception as e:
        print(f"\n ERROR processing {subject_id}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def visualize_connectivity(subject_id,
                          connectivity_dir='../../data/connectivity',
                          output_dir='../../results/figures'):
    """
    Create visualization of connectivity matrix
    """
    
    # Load connectivity matrix
    conn_file = os.path.join(connectivity_dir, f'{subject_id}_connectivity.npy')
    connectivity = np.load(conn_file)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(connectivity, 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'label': 'Z-transformed Correlation'})
    
    plt.title(f'Functional Connectivity Matrix: {subject_id}', fontsize=14)
    plt.xlabel('Brain Region', fontsize=12)
    plt.ylabel('Brain Region', fontsize=12)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_file = os.path.join(output_dir, f'{subject_id}_connectivity_matrix.png')
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {fig_file}")
    
    return fig_file


if __name__ == "__main__":
    # Test on first subject
    test_subject = 'sub-0010001'
    
    print("CONNECTIVITY MATRIX CALCULATION - SINGLE SUBJECT TEST")
    
    # Calculate connectivity
    success = calculate_connectivity_single(test_subject)
    
    if success:
        print("\n Connectivity calculation successful!")
        
        # Create visualization
        print("\nCreating visualization...")
        visualize_connectivity(test_subject)
        
        print("\n Ready to batch process all 201 subjects!")
    else:
        print("\n TEST FAILED - Debug needed")