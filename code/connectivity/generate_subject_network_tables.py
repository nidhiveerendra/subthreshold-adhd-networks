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

def matrix_to_table(matrix, subject_id, adhd_score, group):
    rows = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            rows.append({
                'Subject_ID': subject_id,
                'Group': group,
                'ADHD_Index': adhd_score,
                'Region_A': regions[i],
                'Region_B': regions[j],
                'Fisher_Z': round(matrix[i, j], 4)
            })
    df = pd.DataFrame(rows)
    df = df.reindex(df['Fisher_Z'].abs().sort_values(ascending=False).index)
    return df

# Load matrices
sub_data = np.load('data/connectivity/sub-1700637_connectivity.npy')
diag_data = np.load('data/connectivity/sub-1099481_connectivity.npy')

print(f'Subthreshold matrix shape: {sub_data.shape}')
print(f'Diagnosed matrix shape: {diag_data.shape}')

# Generate tables
sub_table = matrix_to_table(sub_data, '1700637', 43, 'Subthreshold')
diag_table = matrix_to_table(diag_data, '1099481', 86, 'Diagnosed')

# Save
os.makedirs('results', exist_ok=True)
sub_table.to_csv('results/sub-1700637_network_table.csv', index=False)
diag_table.to_csv('results/sub-1099481_network_table.csv', index=False)

print()
print('SUBTHRESHOLD (ID: 1700637, ADHD Index: 43)')
print('Top 10 strongest connections:')
print(sub_table.head(10)[['Region_A', 'Region_B', 'Fisher_Z']].to_string())
print()
print('DIAGNOSED (ID: 1099481, ADHD Index: 86)')
print('Top 10 strongest connections:')
print(diag_table.head(10)[['Region_A', 'Region_B', 'Fisher_Z']].to_string())