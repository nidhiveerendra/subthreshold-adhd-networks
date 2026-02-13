"""
Batch calculate graph theory metrics for all 201 subjects
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculate_graph_metrics import calculate_graph_metrics

def batch_calculate_metrics(log_file='../../data/connectivity/connectivity_log.txt',
                            connectivity_dir='../../data/connectivity',
                            output_dir='../../data/graph_metrics',
                            density=0.15):
    """
    Batch calculate graph metrics for all subjects
    """
    
    print("BATCH GRAPH THEORY METRICS CALCULATION")
    print(f"Network density: {density} (proportional thresholding)")
    
    # 1. Read successful subjects from connectivity log
    print("\n1. Reading connectivity log...")
    successful_subjects = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    reading_successful = False
    for line in lines:
        if line.strip() == "SUCCESSFUL SUBJECTS:":
            reading_successful = True
            continue
        if reading_successful and line.strip().startswith('sub-'):
            successful_subjects.append(line.strip())
    
    print(f"   Found {len(successful_subjects)} subjects with connectivity matrices")
    
    # 2. Estimate time
    time_per_subject = 15  # seconds (graph metrics take longer)
    total_time_min = (len(successful_subjects) * time_per_subject) / 60
    
    print(f"\n2. Time Estimate:")
    print(f"   ~{time_per_subject} seconds per subject")
    print(f"   Total: ~{total_time_min:.1f} minutes ({total_time_min/60:.2f} hours)")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. Process all subjects
    print(f"\n3. Processing {len(successful_subjects)} subjects...")
    
    all_metrics = []
    successful = []
    failed = []
    start_time = time.time()
    
    for i, subject_id in enumerate(successful_subjects, 1):
        if i % 25 == 1:  # Print every 25th subject
            print(f"\n[{i}/{len(successful_subjects)}] Processing: {subject_id}")
        
        metrics = calculate_graph_metrics(subject_id, connectivity_dir, output_dir, density)
        
        if metrics:
            successful.append(subject_id)
            # Store metrics with subject ID
            metrics['subject_id'] = subject_id
            all_metrics.append(metrics)
        else:
            failed.append(subject_id)
        
        # Progress update every 25 subjects
        if i % 25 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(successful_subjects) - i) * avg_time
            
            print(f"PROGRESS UPDATE:")
            print(f"   Completed: {i}/{len(successful_subjects)} ({i/len(successful_subjects)*100:.1f}%)")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            print(f"   Time elapsed: {elapsed/60:.1f} minutes")
            print(f"   Estimated remaining: {remaining/60:.1f} minutes")
    
    # 4. Final summary
    total_time = time.time() - start_time
    
    print("BATCH GRAPH METRICS CALCULATION COMPLETE!")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {len(successful)}/{len(successful_subjects)}")
    print(f"Failed: {len(failed)}/{len(successful_subjects)}")
    
    if failed:
        print(f"\nFailed subjects:")
        for subj in failed:
            print(f"   - {subj}")
    
    # 5. Combine all metrics into DataFrame
    print(f"\n5. Creating combined metrics dataframe...")
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, 'all_subjects_graph_metrics.csv')
    df_metrics.to_csv(csv_file, index=False)
    print(f"   Saved: {csv_file}")
    
    # Print summary statistics
    print(f"\n6. Summary Statistics:")
    print(f"   Clustering Coefficient: {df_metrics['clustering_coefficient'].mean():.4f} ± {df_metrics['clustering_coefficient'].std():.4f}")
    print(f"   Path Length: {df_metrics['path_length'].mean():.4f} ± {df_metrics['path_length'].std():.4f}")
    print(f"   Global Efficiency: {df_metrics['global_efficiency'].mean():.4f} ± {df_metrics['global_efficiency'].std():.4f}")
    print(f"   Modularity: {df_metrics['modularity'].mean():.4f} ± {df_metrics['modularity'].std():.4f}")
    
    # 6. Save processing log
    log_output = os.path.join(output_dir, 'graph_metrics_log.txt')
    with open(log_output, 'w') as f:
        f.write("GRAPH THEORY METRICS LOG\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network density: {density} (proportional)\n")
        f.write(f"Total subjects: {len(successful_subjects)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        
        f.write("METRICS CALCULATED:\n")
        f.write("  - Clustering coefficient (binary & weighted)\n")
        f.write("  - Characteristic path length\n")
        f.write("  - Global efficiency (binary & weighted)\n")
        f.write("  - Modularity\n")
        f.write("  - Small-worldness\n")
        f.write("  - Degree statistics\n")
        f.write("  - Assortativity\n\n")
        
        f.write("SUCCESSFUL SUBJECTS:\n")
        for subj in successful:
            f.write(f"{subj}\n")
        if failed:
            f.write("\nFAILED SUBJECTS:\n")
            for subj in failed:
                f.write(f"{subj}\n")
    
    print(f"\nProcessing log saved to: {log_output}")
    
    print(" ALL DONE! Ready for statistical analysis!")
    
    return df_metrics, successful, failed


if __name__ == "__main__":
    df_metrics, successful, failed = batch_calculate_metrics()
    
    print(f"\n Graph metrics calculation complete!")
    print(f"   All metrics saved to: data/graph_metrics/all_subjects_graph_metrics.csv")
    print(f"   Next step: Statistical analysis (correlations with ADHD Index)")