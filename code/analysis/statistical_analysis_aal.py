import numpy as np
import pandas as pd
import os
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

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

def get_weighted_degree(matrix, threshold=0.3):
    """calculate weighted degree for each node"""
    r_matrix = np.tanh(matrix)
    r_matrix[r_matrix < threshold] = 0
    np.fill_diagonal(r_matrix, 0)
    return r_matrix.sum(axis=1)

# Load phenotypic data
df = pd.read_csv('data/phenotypic/NYU_phenotypic.csv')
df['ADHD Index'] = pd.to_numeric(df['ADHD Index'], errors='coerce')
df = df.dropna(subset=['ADHD Index'])

sub_pheno = df[(df['ADHD Index'] >= 40) & (df['ADHD Index'] < 60)]
diag_pheno = df[df['ADHD Index'] >= 60]

sub_ids = [str(int(x)) for x in sub_pheno['ScanDir ID'].tolist()]
diag_ids = [str(int(x)) for x in diag_pheno['ScanDir ID'].tolist()]

def load_subject_degrees(subject_ids, conn_dir, pheno_df):
    degrees = []
    ages = []
    sexes = []
    loaded_ids = []

    for sid in subject_ids:
        filepath = os.path.join(conn_dir, f'sub-{sid}_connectivity_aal.npy')
        try:
            mat = np.load(filepath)
            if mat.shape == (116, 116):
                mat = np.nan_to_num(mat, nan=0.0)
                deg = get_weighted_degree(mat)
                degrees.append(deg)
                # Get demographics
                row = pheno_df[pheno_df['ScanDir ID'] == int(sid)]
                if len(row) > 0:
                    ages.append(float(row['Age'].values[0]))
                    sexes.append(float(row['Gender'].values[0]))
                else:
                    ages.append(np.nan)
                    sexes.append(np.nan)
                loaded_ids.append(sid)
        except:
            pass

    return np.array(degrees), np.array(ages), np.array(sexes), loaded_ids

sub_degrees, sub_ages, sub_sexes, sub_loaded = load_subject_degrees(
    sub_ids, 'data/connectivity_aal', df)
diag_degrees, diag_ages, diag_sexes, diag_loaded = load_subject_degrees(
    diag_ids, 'data/connectivity_aal', df)

print(f'Subthreshold: {len(sub_loaded)} subjects')
print(f'Diagnosed: {len(diag_loaded)} subjects')
print(f'Sub degrees shape: {sub_degrees.shape}')
print(f'Diag degrees shape: {diag_degrees.shape}')

# Run Mann-Whitney U tests for each region
pvalues = []
ustatistics = []
effect_sizes = []
sub_means = []
diag_means = []

for i in range(116):
    sub_vals = sub_degrees[:, i]
    diag_vals = diag_degrees[:, i]

    u_stat, p_val = stats.mannwhitneyu(sub_vals, diag_vals, alternative='two-sided')
    
    # effect size (rank-biserial correlation)
    n1, n2 = len(sub_vals), len(diag_vals)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)

    pvalues.append(p_val)
    ustatistics.append(u_stat)
    effect_sizes.append(effect_size)
    sub_means.append(sub_vals.mean())
    diag_means.append(diag_vals.mean())

# FDR correction
pvalues = np.array(pvalues)
rejected, pvalues_corrected = fdrcorrection(pvalues, alpha=0.05)

# build results dataframe
results = pd.DataFrame({
    'Region': AAL_REGIONS,
    'Sub_Mean_Degree': sub_means,
    'Diag_Mean_Degree': diag_means,
    'Degree_Diff': np.array(diag_means) - np.array(sub_means),
    'U_Statistic': ustatistics,
    'P_Value': pvalues,
    'P_Value_FDR': pvalues_corrected,
    'Significant_FDR': rejected,
    'Effect_Size': effect_sizes
})

results = results.sort_values('P_Value_FDR')

# save
os.makedirs('results', exist_ok=True)
results.to_csv('results/statistical_analysis_aal.csv', index=False)

print('\nSIGNIFICANT REGIONS (FDR corrected p < 0.05):')
sig = results[results['Significant_FDR']]
if len(sig) > 0:
    print(sig[['Region', 'Sub_Mean_Degree', 'Diag_Mean_Degree',
               'Degree_Diff', 'P_Value_FDR', 'Effect_Size']].to_string(index=False))
else:
    print('No regions survived FDR correction')

print('\nTOP 20 REGIONS BY UNCORRECTED P-VALUE:')
print(results.head(20)[['Region', 'Sub_Mean_Degree', 'Diag_Mean_Degree',
                          'Degree_Diff', 'P_Value', 'P_Value_FDR',
                          'Effect_Size']].to_string(index=False))
