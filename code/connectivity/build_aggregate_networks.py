import pandas as pd
import numpy as np
import os

# Harvard-Oxford 45 region names
regions = [
    'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus',
    'Middle Frontal Gyrus', 'Inferior Frontal Gyrus (pars triangularis)',
    'Inferior Frontal Gyrus (pars opercularis)', 'Precentral Gyrus',
    'Temporal Pole', 'Superior Temporal Gyrus (anterior)',
    'Superior Temporal Gyrus (posterior)', 'Middle Temporal Gyrus (anterior)',
    'Middle Temporal Gyrus (posterior)', 'Middle Temporal Gyrus (temporooccipital)',
    'Inferior Temporal Gyrus (anterior)', 'Inferior Temporal Gyrus (posterior)',
    'Inferior Temporal Gyrus (temporooccipital)', 'Postcentral Gyrus',
    'Superior Parietal Lobule', 'Supramarginal Gyrus (anterior)',
    'Supramarginal Gyrus (posterior)', 'Angular Gyrus',
    'Lateral Occipital Cortex (superior)', 'Lateral Occipital Cortex (inferior)',
    'Intracalcarine Cortex', 'Frontal Medial Cortex',
    'Juxtapositional Lobule Cortex', 'Subcallosal Cortex',
    'Paracingulate Gyrus', 'Cingulate Gyrus (anterior)',
    'Cingulate Gyrus (posterior)', 'Precuneous Cortex', 'Cuneal Cortex',
    'Frontal Orbital Cortex', 'Parahippocampal Gyrus (anterior)',
    'Parahippocampal Gyrus (posterior)', 'Lingual Gyrus',
    'Temporal Fusiform Cortex (anterior)', 'Temporal Fusiform Cortex (posterior)',
    'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus',
    'Frontal Operculum Cortex', 'Central Opercular Cortex',
    'Parietal Operculum Cortex', 'Planum Polare', 'Planum Temporale'
]

def load_group_matrices(subject_ids, data_dir):
    """Load all connectivity matrices for a group, only keeping 45x45 ones"""
    matrices = []
    loaded = []
    skipped = []
    
    for subid in subject_ids:
        filepath = os.path.join(data_dir, f'sub-{subid}_connectivity.npy')
        try:
            mat = np.load(filepath)
            if mat.shape == (45, 45):
                matrices.append(mat)
                loaded.append(subid)
            else:
                skipped.append((subid, mat.shape))
        except FileNotFoundError:
            pass
    
    print(f'  Loaded: {len(loaded)} subjects')
    print(f'  Skipped (wrong size): {len(skipped)} subjects')
    return np.array(matrices), loaded

def build_aggregate_network(matrices, regions, group_name):
    """
    Build aggregate network by:
    1. Averaging Fisher z values across all subjects
    2. Back-transforming to r
    """
    print(f'\nBuilding aggregate network for {group_name}...')
    print(f'  Number of subjects: {matrices.shape[0]}')
    
    # Average Fisher z values across subjects
    avg_fisher_z = np.mean(matrices, axis=0)
    
    # Back-transform to r: r = tanh(z)
    avg_r = np.tanh(avg_fisher_z)
    
    # Convert to edge table
    rows = []
    for i in range(len(regions)):
        for j in range(i+1, len(regions)):
            rows.append({
                'Source': regions[i],
                'Target': regions[j],
                'Group': group_name,
                'Avg_Fisher_Z': round(avg_fisher_z[i, j], 4),
                'Avg_r': round(avg_r[i, j], 4)
            })
    
    df = pd.DataFrame(rows)
    df = df.reindex(df['Avg_r'].abs().sort_values(ascending=False).index)
    return df, avg_fisher_z, avg_r

#  LOAD PHENOTYPIC DATA 
pheno = pd.read_csv('data/phenotypic/NYU_phenotypic.csv')
pheno['ADHD Index'] = pd.to_numeric(pheno['ADHD Index'], errors='coerce')
pheno = pheno.dropna(subset=['ADHD Index'])

subthreshold = pheno[(pheno['ADHD Index'] >= 40) & (pheno['ADHD Index'] < 60)]
diagnosed = pheno[pheno['ADHD Index'] >= 60]

sub_ids = [str(int(x)) for x in subthreshold['ScanDir ID'].tolist()]
diag_ids = [str(int(x)) for x in diagnosed['ScanDir ID'].tolist()]

print(f'Subthreshold subjects in phenotypic data: {len(sub_ids)}')
print(f'Diagnosed subjects in phenotypic data: {len(diag_ids)}')

# LOAD MATRICES 
sub_matrices, sub_loaded = load_group_matrices(sub_ids, 'data/connectivity')
diag_matrices, diag_loaded = load_group_matrices(diag_ids, 'data/connectivity')

# BUILD AGGREGATE NETWORKS 
sub_network, sub_z, sub_r = build_aggregate_network(sub_matrices, regions, 'Subthreshold')
diag_network, diag_z, diag_r = build_aggregate_network(diag_matrices, regions, 'Diagnosed')

# SAVE RESULTS
os.makedirs('results', exist_ok=True)

sub_network.to_csv('results/aggregate_subthreshold_network.csv', index=False)
diag_network.to_csv('results/aggregate_diagnosed_network.csv', index=False)

print('\n TOP 10 CONNECTIONS ')
print('\nSUBTHRESHOLD aggregate network:')
print(sub_network.head(10)[['Source', 'Target', 'Avg_Fisher_Z', 'Avg_r']].to_string())

print('\nDIAGNOSED aggregate network:')
print(diag_network.head(10)[['Source', 'Target', 'Avg_Fisher_Z', 'Avg_r']].to_string())

# COMPARE TOP CONNECTIONS 
print('\n COMPARISON ')
print(f'Subthreshold max avg r: {sub_network["Avg_r"].max()}')
print(f'Diagnosed max avg r: {diag_network["Avg_r"].max()}')
print(f'Subthreshold mean avg r: {round(sub_network["Avg_r"].mean(), 4)}')
print(f'Diagnosed mean avg r: {round(diag_network["Avg_r"].mean(), 4)}')

print('\ndone! files saved:')
print('  results/aggregate_subthreshold_network.csv')
print('  results/aggregate_diagnosed_network.csv')