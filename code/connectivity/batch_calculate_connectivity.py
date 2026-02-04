"""
Batch calculate connectivity matrices for all 201 subjects
"""

import os
import sys
import time
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculate_connectivity import calculate_connectivity_single

def batch_calculate_connectivity(log_file='../../data/timeseries/extraction_log.txt',
                                 timeseries_dir='../../data/timeseries',
                                 output_dir='../../data/connectivity'):
    """
    Batch calculate connectivity matrices for all subjects
    """
    
    print("BATCH CONNECTIVITY MATRIX CALCULATION")
    
    # 1. Read successful subjects from extraction log
    print("\n1. Reading extraction log...")
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
    
    print(f"   Found {len(successful_subjects)} subjects with time series")
    
    # 2. Estimate time
    time_per_subject = 2  # seconds (connectivity is fast!)
    total_time_min = (len(successful_subjects) * time_per_subject) / 60
    
    print(f"\n2. Time Estimate:")
    print(f"   ~{time_per_subject} seconds per subject")
    print(f"   Total: ~{total_time_min:.1f} minutes")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. Process all subjects
    print(f"\n3. Processing {len(successful_subjects)} subjects...")
    
    successful = []
    failed = []
    start_time = time.time()
    
    for i, subject_id in enumerate(successful_subjects, 1):
        if i % 50 == 1:  # Print every 50th subject
            print(f"\n[{i}/{len(successful_subjects)}] Processing: {subject_id}")
        
        success = calculate_connectivity_single(subject_id, timeseries_dir, output_dir)
        
        if success:
            successful.append(subject_id)
        else:
            failed.append(subject_id)
        
        # Progress update every 50 subjects
        if i % 50 == 0:
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
    
    print("BATCH CONNECTIVITY CALCULATION COMPLETE!")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {len(successful)}/{len(successful_subjects)}")
    print(f"Failed: {len(failed)}/{len(successful_subjects)}")
    
    if failed:
        print(f"\nFailed subjects:")
        for subj in failed:
            print(f"   - {subj}")
    
    # 5. Save connectivity log
    log_output = '../../data/connectivity/connectivity_log.txt'
    with open(log_output, 'w') as f:
        f.write("CONNECTIVITY MATRIX CALCULATION LOG\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Matrix size: 45x45 (Harvard-Oxford atlas)\n")
        f.write(f"Total subjects: {len(successful_subjects)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        f.write("SUCCESSFUL SUBJECTS:\n")
        for subj in successful:
            f.write(f"{subj}\n")
        if failed:
            f.write("\nFAILED SUBJECTS:\n")
            for subj in failed:
                f.write(f"{subj}\n")
    
    print(f"\nConnectivity log saved to: {log_output}")
    
    # 6. Verify output
    print(f"\nVerifying output files...")
    conn_files = [f for f in os.listdir(output_dir) if f.endswith('_connectivity.npy')]
    print(f"   Connectivity matrix files created: {len(conn_files)}")
    
    # Calculate total data size
    total_size = 0
    for f in conn_files:
        total_size += os.path.getsize(os.path.join(output_dir, f))
    print(f"   Total data size: {total_size / 1e6:.2f} MB")
    
    print(" ALL DONE! Ready for graph theory analysis!")
    
    return successful, failed


if __name__ == "__main__":
    successful, failed = batch_calculate_connectivity()
    
    print(f"\n Connectivity matrix calculation complete!")
    print(f"   Next step: Graph theory metrics")