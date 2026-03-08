
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from nilearn.maskers import NiftiLabelsMasker

# AAL 116 region names
AAL_REGIONS = [
    'Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R',
    'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
    'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
    'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
    'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
    'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
    'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R',
    'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
    'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R',
    'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
    'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R',
    'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R',
    'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
    'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
    'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
    'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R',
    'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
    'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R',
    'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',
    'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
    'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R',
    'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R',
    'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R',
    'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R',
    'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6',
    'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10'
]

def extract_timeseries_aal(subject_id,
                           processed_dir='data/processed/NYU',
                           output_dir='data/timeseries_aal',
                           atlas_file='/Users/nidhiveerendra/nilearn_data/aal_atlas.nii.gz'):
    """Extract AAL timeseries for one subject"""

    fmri_file = os.path.join(processed_dir, subject_id,
                             f'{subject_id}_preprocessed.nii.gz')

    if not os.path.exists(fmri_file):
        return False, 'File not found'

    try:
        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize=True,
            resampling_target='labels',
            memory='nilearn_cache'
        )

        ts = masker.fit_transform(fmri_file)

        if ts.shape != (176, 116):
            return False, f'Unexpected shape: {ts.shape}'

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{subject_id}_timeseries_aal.npy')
        np.save(output_file, ts)

        return True, ts.shape

    except Exception as e:
        return False, str(e)


if __name__ == '__main__':
    processed_dir = 'data/processed/NYU'
    output_dir = 'data/timeseries_aal'

    subjects = sorted([s for s in os.listdir(processed_dir)
                      if os.path.isdir(os.path.join(processed_dir, s))])

    print(f'AAL TIMESERIES EXTRACTION')
    print(f'Total subjects: {len(subjects)}')
    print(f'Output: {output_dir}')
    print(f'Expected shape per subject: (176, 116)')
    print()

    success = []
    failed = []

    for i, subj in enumerate(subjects):
        ok, result = extract_timeseries_aal(subj, processed_dir, output_dir)
        if ok:
            success.append(subj)
            if (i+1) % 10 == 0:
                print(f'Progress: {i+1}/{len(subjects)} | Success: {len(success)} | Failed: {len(failed)}')
        else:
            failed.append((subj, result))
            print(f'FAILED: {subj} - {result}')

    print()
    print(f'COMPLETE!')
    print(f'Successful: {len(success)}/{len(subjects)}')
    print(f'Failed: {len(failed)}')

    # Save labels for reference
    labels_file = os.path.join(output_dir, 'AAL_labels.txt')
    with open(labels_file, 'w') as f:
        for i, label in enumerate(AAL_REGIONS):
            f.write(f'{i+1}: {label}\n')
    print(f'Labels saved to: {labels_file}')