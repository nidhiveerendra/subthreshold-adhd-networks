"""
Batch extract time series for all 201 successfully preprocessed subjects
Using Harvard-Oxford atlas
"""

import os
import sys
import time
from datetime import datetime
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_timeseries import extract_timeseries_single

def batch_extract_timeseries(log_file='../../data/processed/preprocessing_log.txt',
                             data_dir='../../data/processed/NYU',
                             output_dir='../../data/timeseries'):
    """
    Batch extract time series for all successfully preprocessed subjects
    """
    
    print("BATCH TIME SERIES EXTRACTION")
    print("Using Harvard-Oxford Atlas (45 cortical regions)")
    
    # 1. Read successful subjects from preprocessing log
    print("\n1. Reading preprocessing log...")
    successful_subjects = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # Find where successful subjects list starts
    reading_successful = False
    for line in lines:
        if line.strip() == "SUCCESSFUL SUBJECTS:":
            reading_successful = True
            continue
        if line.strip() == "FAILED SUBJECTS:":
            break
        if reading_successful and line.strip().startswith('sub-'):
            successful_subjects.append(line.strip())
    
    print(f"   Found {len(successful_subjects)} successful subjects")
    
    # 2. Estimate time
    time_per_subject = 45  # seconds (estimate)
    total_time_min = (len(successful_subjects) * time_per_subject) / 60
    
    print(f"\n2. Time Estimate:")
    print(f"   ~{time_per_subject} seconds per subject")
    print(f"   Total: ~{total_time_min:.1f} minutes ({total_time_min/60:.2f} hours)")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. Process all subjects
    print(f"\n3. Processing {len(successful_subjects)} subjects...")
    
    successful = []
    failed = []
    start_time = time.time()
    
    for i, subject_id in enumerate(successful_subjects, 1):
        print(f"\n[{i}/{len(successful_subjects)}] Processing: {subject_id}")
        
        success = extract_timeseries_single(subject_id, data_dir, output_dir)
        
        if success:
            successful.append(subject_id)
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
    
    print("BATCH EXTRACTION COMPLETE!")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {len(successful)}/{len(successful_subjects)}")
    print(f"Failed: {len(failed)}/{len(successful_subjects)}")
    
    if failed:
        print(f"\nFailed subjects:")
        for subj in failed:
            print(f"   - {subj}")
    
    # 5. Save extraction log
    log_output = '../../data/timeseries/extraction_log.txt'
    with open(log_output, 'w') as f:
        f.write("TIME SERIES EXTRACTION LOG\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Atlas: Harvard-Oxford Cortical (45 regions)\n")
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
    
    print(f"\nExtraction log saved to: {log_output}")
    
    # 6. Verify output
    print(f"\nVerifying output files...")
    timeseries_files = [f for f in os.listdir(output_dir) if f.endswith('_timeseries.npy')]
    print(f"   Time series files created: {len(timeseries_files)}")
    
    # Calculate total data size
    total_size = 0
    for f in timeseries_files:
        total_size += os.path.getsize(os.path.join(output_dir, f))
    print(f"   Total data size: {total_size / 1e6:.2f} MB")
    
    print(" ALL DONE! Ready for connectivity matrix calculation!")
    
    return successful, failed


if __name__ == "__main__":
    successful, failed = batch_extract_timeseries()
    
    print(f"\n Time series extraction complete!")
    print(f"   Next step: Calculate connectivity matrices")