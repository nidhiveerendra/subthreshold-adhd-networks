"""
Batch preprocess all subjects for subthreshold vs. diagnosed analysis
Only processes subjects in the subthreshold (105) and diagnosed (113) groups
"""

import pandas as pd
import os
from preprocess_single import preprocess_subject
from datetime import datetime
import time

def batch_preprocess(phenotypic_file='data/phenotypic/NYU_phenotypic.csv',
                     data_dir='data/raw/NYU',
                     output_dir='data/processed/NYU'):
    """
    Batch preprocess all subjects in subthreshold and diagnosed groups
    """
    
    print("BATCH PREPROCESSING: NYU SUBTHRESHOLD & DIAGNOSED")
    
    # Load phenotypic data
    print("\n1. Loading phenotypic data...")
    df = pd.read_csv(phenotypic_file)
    
    # Create groups (same logic as before)
    index_col = 'ADHD Index'
    df['Group'] = 'Unknown'
    valid_scores = df[index_col] != -999
    df.loc[valid_scores & (df[index_col] >= 40) & (df[index_col] < 60), 'Group'] = 'Subthreshold'
    df.loc[valid_scores & (df[index_col] >= 60), 'Group'] = 'Diagnosed'
    
    # Filter to only subthreshold and diagnosed
    df_analysis = df[df['Group'].isin(['Subthreshold', 'Diagnosed'])].copy()
    
    n_subthreshold = (df_analysis['Group'] == 'Subthreshold').sum()
    n_diagnosed = (df_analysis['Group'] == 'Diagnosed').sum()
    
    print(f"   Total subjects to process: {len(df_analysis)}")
    print(f"   - Subthreshold: {n_subthreshold}")
    print(f"   - Diagnosed: {n_diagnosed}")
    
    # Get subject IDs (convert ScanDir ID to sub-XXXXXXX format)
    df_analysis['subject_id'] = df_analysis['ScanDir ID'].apply(lambda x: f'sub-{x:07d}')
    subjects_to_process = df_analysis['subject_id'].tolist()
    
    print(f"\n2. Starting batch preprocessing...")
    print(f"   This will take ~{len(subjects_to_process) * 1.5:.0f} minutes ({len(subjects_to_process) * 1.5 / 60:.1f} hours)")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track progress
    successful = []
    failed = []
    start_time = time.time()
    
    for i, subject_id in enumerate(subjects_to_process, 1):
        print(f"Processing {i}/{len(subjects_to_process)}: {subject_id}")
        
        success = preprocess_subject(subject_id, data_dir, output_dir)
        
        if success:
            successful.append(subject_id)
        else:
            failed.append(subject_id)
        
        # Progress update every 10 subjects
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(subjects_to_process) - i) * avg_time
            
            print(f"\nPROGRESS UPDATE:")
            print(f"   Completed: {i}/{len(subjects_to_process)} ({i/len(subjects_to_process)*100:.1f}%)")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            print(f"   Time elapsed: {elapsed/60:.1f} minutes")
            print(f"   Estimated remaining: {remaining/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("BATCH PREPROCESSING COMPLETE!")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {len(successful)}/{len(subjects_to_process)}")
    print(f"Failed: {len(failed)}/{len(subjects_to_process)}")
    
    if failed:
        print(f"\n Failed subjects:")
        for subj in failed:
            print(f"   - {subj}")
        
        # Save failed list
        with open('data/processed/failed_subjects.txt', 'w') as f:
            f.write("Failed Subjects\n")
            for subj in failed:
                f.write(f"{subj}\n")
        print(f"\n   Saved list to: data/processed/failed_subjects.txt")
    
    # Save processing log
    log_file = 'data/processed/preprocessing_log.txt'
    with open(log_file, 'w') as f:
        f.write("PREPROCESSING LOG\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total subjects: {len(subjects_to_process)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write("SUCCESSFUL SUBJECTS:\n")
        for subj in successful:
            f.write(f"{subj}\n")
        if failed:
            f.write("FAILED SUBJECTS:\n")
            for subj in failed:
                f.write(f"{subj}\n")
    
    print(f"\nProcessing log saved to: {log_file}")
    
    return successful, failed


if __name__ == "__main__":
    successful, failed = batch_preprocess()
    
    print("\nAll done! Ready for connectivity analysis!")