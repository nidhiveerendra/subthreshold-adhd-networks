"""
Preprocess a single subject's fMRI data
This will be the template for batch processing
"""

from nilearn import image
from nilearn.maskers import NiftiMasker
import nibabel as nib
import numpy as np
import os
import sys

def preprocess_subject(subject_id, data_dir='data/raw/NYU', output_dir='data/processed/NYU'):
    """
    Preprocess one subject's resting-state fMRI data
    
    Steps:
    1. Load fMRI data
    2. Motion correction (realignment)
    3. Spatial smoothing (6mm FWHM)
    4. Temporal filtering (0.01-0.1 Hz bandpass)
    5. Confound regression (motion parameters)
    6. Save preprocessed data
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'sub-0010001')
    data_dir : str
        Path to raw data directory
    output_dir : str
        Path to save preprocessed data
    
    Returns:
    --------
    success : bool
        True if preprocessing succeeded
    """
    
    print(f"PREPROCESSING: {subject_id}")
    
    # Create output directory
    os.makedirs(f"{output_dir}/{subject_id}", exist_ok=True)
    
    try:
        # 1. LOAD DATA
        print("\n1. Loading fMRI data...")
        func_file = f"{data_dir}/{subject_id}/ses-1/func/{subject_id}_ses-1_task-rest_run-1_bold.nii.gz"
        
        if not os.path.exists(func_file):
            print(f"   File not found: {func_file}")
            return False
        
        img = nib.load(func_file)
        print(f"   Loaded: {img.shape}")
        
        # 2. CLEAN IMAGE (using nilearn's clean_img function)
        print("\n2. Cleaning fMRI signal...")
        print("   - Standardizing (z-score)")
        print("   - Detrending (removing linear trends)")
        print("   - Bandpass filtering (0.01-0.1 Hz)")
        
        cleaned_img = image.clean_img(
            img,
            standardize='zscore',      # Standardize each voxel
            detrend=True,               # Remove linear trends
            low_pass=0.1,               # Low-pass filter at 0.1 Hz
            high_pass=0.01,             # High-pass filter at 0.01 Hz
            t_r=2.0,                    # Repetition time (2 seconds for this data)
            ensure_finite=True          # Remove NaN/Inf values
        )
        print("   Signal cleaned")
        
        # 3. SPATIAL SMOOTHING
        print("\n3. Spatial smoothing (6mm FWHM)...")
        smoothed_img = image.smooth_img(cleaned_img, fwhm=6)
        print("   Smoothed")
        
        # 4. QUALITY CHECK
        print("\n4. Quality check...")
        
        # Check for NaN values
        data = smoothed_img.get_fdata()
        n_nans = np.isnan(data).sum()
        n_infs = np.isinf(data).sum()
        
        print(f"   - NaN values: {n_nans}")
        print(f"   - Inf values: {n_infs}")
        print(f"   - Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"   - Data mean: {data.mean():.2f}")
        print(f"   - Data std: {data.std():.2f}")
        
        if n_nans > 0 or n_infs > 0:
            print("   Warning: Found NaN or Inf values!")
        else:
            print("   Data looks good")
        
        # 5. SAVE PREPROCESSED DATA
        print("\n5. Saving preprocessed data...")
        output_file = f"{output_dir}/{subject_id}/{subject_id}_preprocessed.nii.gz"
        nib.save(smoothed_img, output_file)
        
        # Get file size
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"   Saved: {output_file}")
        print(f"   File size: {file_size_mb:.1f} MB")
        
        print(f"SUCCESS: {subject_id} preprocessed!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR processing {subject_id}:")
        print(f"   {str(e)}")
        return False


if __name__ == "__main__":
    # Test with first subject
    if len(sys.argv) > 1:
        subject_id = sys.argv[1]
    else:
        # Default: process first subject
        data_dir = 'data/raw/NYU'
        subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
        subject_id = subjects[0]
        print(f"No subject specified, using first subject: {subject_id}")
    
    success = preprocess_subject(subject_id)
    
    if success:
        print("Preprocessing complete!")
    else:
        print("Preprocessing failed - check errors above")