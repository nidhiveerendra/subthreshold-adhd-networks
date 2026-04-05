import numpy as np
import pandas as pd
import os
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.linear_model import LinearRegression

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
    r_matrix = np.tanh(matrix)
    r_matrix[r_matrix < threshold] = 0
    np.fill_diagonal(r_matrix, 0)
    return r_matrix.sum(axis=1)

def regress_out_covariates(data, covariates):
    reg = LinearRegression()
    reg.fit(covariates, data)
    residuals = data - reg.predict(covariates)
    return residuals

# Load phenotypic data
df = pd.read_csv('data/phenotypic/NYU_phenotypic.csv')
df['ADHD Index'] = pd.to_numeric(df['ADHD Index'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
df = df.dropna(subset=['ADHD Index', 'Age', 'Gender'])

sub_pheno = df[(df['ADHD Index'] >= 40) & (df['ADHD Index'] < 60)]
diag_pheno = df[df['ADHD Index'] >= 60]

sub_ids = [str(int(x)) for x in sub_pheno['ScanDir ID'].tolist()]
diag_ids = [str(int(x)) for x in diag_pheno['ScanDir ID'].tolist()]

def load_subject_data(subject_ids, conn_dir, pheno_df):
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
                row = pheno_df[pheno_df['ScanDir ID'] == int(sid)]
                if len(row) > 0 and not pd.isna(row['Age'].values[0]):
                    degrees.append(deg)
                    ages.append(float(row['Age'].values[0]))
                    sexes.append(float(row['Gender'].values[0]))
                    loaded_ids.append(sid)
        except:
            pass

    return np.array(degrees), np.array(ages), np.array(sexes), loaded_ids

sub_deg, sub_age, sub_sex, sub_ids_loaded = load_subject_data(
    sub_ids, 'data/connectivity_aal', df)
diag_deg, diag_age, diag_sex, diag_ids_loaded = load_subject_data(
    diag_ids, 'data/connectivity_aal', df)

print(f'Subthreshold: {len(sub_ids_loaded)} subjects')
print(f'Diagnosed: {len(diag_ids_loaded)} subjects')

# Combine all subjects
all_degrees = np.vstack([sub_deg, diag_deg])
all_ages = np.concatenate([sub_age, diag_age])
all_sexes = np.concatenate([sub_sex, diag_sex])
all_groups = np.array([0]*len(sub_ids_loaded) + [1]*len(diag_ids_loaded))

# Regress out age and sex from each region
covariates = np.column_stack([all_ages, all_sexes])
residual_degrees = np.zeros_like(all_degrees)

for i in range(116):
    residual_degrees[:, i] = regress_out_covariates(
        all_degrees[:, i], covariates)

# Split back into groups
sub_resid = residual_degrees[:len(sub_ids_loaded)]
diag_resid = residual_degrees[len(sub_ids_loaded):]

# Run Mann-Whitney U tests on residuals
pvalues = []
effect_sizes = []
sub_means = []
diag_means = []

for i in range(116):
    u_stat, p_val = stats.mannwhitneyu(
        sub_resid[:, i], diag_resid[:, i], alternative='two-sided')
    n1, n2 = len(sub_resid), len(diag_resid)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)
    pvalues.append(p_val)
    effect_sizes.append(effect_size)
    sub_means.append(sub_resid[:, i].mean())
    diag_means.append(diag_resid[:, i].mean())

pvalues = np.array(pvalues)
rejected, pvalues_corrected = fdrcorrection(pvalues, alpha=0.05)

results = pd.DataFrame({
    'Region': AAL_REGIONS,
    'Sub_Mean_Residual': sub_means,
    'Diag_Mean_Residual': diag_means,
    'Residual_Diff': np.array(diag_means) - np.array(sub_means),
    'P_Value': pvalues,
    'P_Value_FDR': pvalues_corrected,
    'Significant_FDR': rejected,
    'Effect_Size': effect_sizes
})

results = results.sort_values('P_Value')
results.to_csv('results/statistical_analysis_covariates.csv', index=False)

sig = results[results['Significant_FDR']]
if len(sig) > 0:
    print(sig[['Region', 'P_Value_FDR', 'Effect_Size']].to_string(index=False))
else:
    print('No regions survived FDR correction')

print('\nTOP 15 REGIONS BY UNCORRECTED P-VALUE (age/sex corrected):')
print(results.head(15)[['Region', 'P_Value', 'P_Value_FDR',
                          'Effect_Size', 'Residual_Diff']].to_string(index=False))