"""
Extract time series from preprocessed fMRI data using Harvard-Oxford atlas
Uses built-in atlas to avoid download issues
"""

import os
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

def extract_timeseries_single(subject_id, 
                              data_dir='../../data/processed/NYU',
                              output_dir='../../data/timeseries'):
    """
    Extract time series from preprocessed fMRI using Harvard-Oxford atlas
    """
    
    print(f"Processing: {subject_id}")
    
    try:
        # 1. Load Harvard-Oxford atlas (cortical, built-in)
        print("1. Loading Harvard-Oxford cortical atlas...")
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_filename = atlas.maps
        labels = atlas.labels
        
        print(f"   Atlas loaded: {len(labels)} regions")
        print(f"   Example regions: {labels[:5]}")
        
        # 2. Load preprocessed fMRI data
        print(f"\n2. Loading preprocessed fMRI for {subject_id}...")
        fmri_file = os.path.join(data_dir, subject_id, 
                                 f'{subject_id}_preprocessed.nii.gz')
        
        if not os.path.exists(fmri_file):
            print(f"   ERROR: File not found: {fmri_file}")
            return False
        
        fmri_img = nib.load(fmri_file)
        print(f"   Loaded: {fmri_img.shape}")
        print(f"   File size: {os.path.getsize(fmri_file) / 1e6:.1f} MB")
        
        # 3. Create masker to extract time series
        print("\n3. Creating NiftiLabelsMasker...")
        masker = NiftiLabelsMasker(
            labels_img=atlas_filename,
            standardize=True,
            memory='nilearn_cache',
            verbose=1
        )
        
        # 4. Extract time series
        print("\n4. Extracting time series from brain regions...")
        time_series = masker.fit_transform(fmri_img)
        
        print(f"   Time series shape: {time_series.shape}")
        print(f"   (timepoints  regions) = ({time_series.shape[0]} × {time_series.shape[1]})")
        print(f"   Data range: [{time_series.min():.3f}, {time_series.max():.3f}]")
        
        # 5. Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 6. Save as numpy array
        output_file = os.path.join(output_dir, f'{subject_id}_timeseries.npy')
        np.save(output_file, time_series)
        print(f"\n5. Saved time series to: {output_file}")
        
        # 7. Also save region labels for reference
        labels_file = os.path.join(output_dir, 'HarvardOxford_labels.txt')
        if not os.path.exists(labels_file):
            with open(labels_file, 'w') as f:
                for i, label in enumerate(labels):
                    f.write(f"{i}: {label}\n")
            print(f"   Saved region labels to: {labels_file}")
        
        # 8. Quality check
        print("\n6. Quality Checks:")
        print(f"    No NaN values: {not np.isnan(time_series).any()}")
        print(f"    No Inf values: {not np.isinf(time_series).any()}")
        print(f"    Expected regions: {time_series.shape[1] == len(labels)}")
        
        # 9. Show sample data
        print("\n7. Sample time series (first 5 timepoints, first 5 regions):")
        print(time_series[:5, :5])
        print(f" SUCCESS: {subject_id}")
        
        return True
        
    except Exception as e:
        print(f"\n ERROR processing {subject_id}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_subject = 'sub-0010001'
    
    print("TIME SERIES EXTRACTION - SINGLE SUBJECT TEST")
    print("Using Harvard-Oxford Atlas (48 cortical regions)")
    
    success = extract_timeseries_single(test_subject)
    
    if success:
        print("\n TEST SUCCESSFUL!")
        print("Ready to batch process all 201 subjects!")
    else:
        print("\n TEST FAILED - Debug needed")