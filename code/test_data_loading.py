import nibabel as nib
import os

print("Testing data access...")

# Path to your data
data_path = "data/raw/NYU"

# List subjects
subjects = [d for d in os.listdir(data_path) if d.startswith('sub-')]
print(f"\nFound {len(subjects)} subjects")
print(f"First 5: {subjects[:5]}")

# Try loading one subject's fMRI file
if len(subjects) > 0:
    subj = subjects[0]
    func_path = f"{data_path}/{subj}/ses-1/func"
    
    if os.path.exists(func_path):
        fmri_files = [f for f in os.listdir(func_path) if f.endswith('.nii.gz')]
        
        if len(fmri_files) > 0:
            test_file = f"{func_path}/{fmri_files[0]}"
            print(f"\nLoading: {test_file}")
            
            img = nib.load(test_file)
            print(f"Successfully loaded!")
            print(f"   Shape: {img.shape}")
            print(f"   Data type: {img.get_data_dtype()}")
            
            # Get dimensions
            x, y, z, t = img.shape
            print(f"\n   Brain dimensions: {x} x {y} x {z} voxels")
            print(f"   Time points: {t} volumes")
            print(f"   File size: {os.path.getsize(test_file) / (1024**2):.1f} MB")
        else:
            print("No fMRI files found")
    else:
        print(f"Path doesn't exist: {func_path}")
else:
    print("No subjects found")