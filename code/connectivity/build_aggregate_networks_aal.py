import numpy as np
import pandas as pd
import os

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

df = pd.read_csv('data/phenotypic/NYU_phenotypic.csv')
df['ADHD Index'] = pd.to_numeric(df['ADHD Index'], errors='coerce')
df = df.dropna(subset=['ADHD Index'])

subthreshold = df[(df['ADHD Index'] >= 40) & (df['ADHD Index'] < 60)]
diagnosed = df[df['ADHD Index'] >= 60]

sub_ids = [str(int(x)) for x in subthreshold['ScanDir ID'].tolist()]
diag_ids = [str(int(x)) for x in diagnosed['ScanDir ID'].tolist()]

def build_aggregate(subject_ids, conn_dir, group_name):
    matrices = []
    for subid in subject_ids:
        filepath = os.path.join(conn_dir, f'sub-{subid}_connectivity_aal.npy')
        try:
            mat = np.load(filepath)
            if mat.shape == (116, 116):
                mat = np.nan_to_num(mat, nan=0.0)
                matrices.append(mat)
        except:
            pass

    print(f'{group_name}: {len(matrices)} subjects loaded')
    matrices = np.array(matrices)
    avg_z = np.nanmean(matrices, axis=0)
    avg_r = np.tanh(avg_z)

    rows = []
    for i in range(116):
        for j in range(i+1, 116):
            rows.append({
                'source': AAL_REGIONS[i],
                'target': AAL_REGIONS[j],
                'Fisher_Z': round(float(avg_z[i, j]), 4),
                'avg_r': round(float(avg_r[i, j]), 4)
            })
    df_out = pd.DataFrame(rows)
    df_out = df_out.reindex(df_out['Fisher_Z'].abs().sort_values(ascending=False).index)
    return df_out
sub_net = build_aggregate(sub_ids, 'data/connectivity_aal', 'Subthreshold')
diag_net = build_aggregate(diag_ids, 'data/connectivity_aal', 'Diagnosed')

os.makedirs('results', exist_ok=True)
sub_net.to_csv('results/AAL_aggregate_subthreshold_network.csv', index=False)
diag_net.to_csv('results/AAL_aggregate_diagnosed_network.csv', index=False)

sub_cyto = sub_net[sub_net['Fisher_Z'] > 0.3][['source', 'target', 'Fisher_Z']]
diag_cyto = diag_net[diag_net['Fisher_Z'] > 0.3][['source', 'target', 'Fisher_Z']]

sub_cyto.to_csv('results/AAL_aggregate_subthreshold_cytoscape.csv', index=False)
diag_cyto.to_csv('results/AAL_aggregate_diagnosed_cytoscape.csv', index=False)

print()
print('TOP 10 SUBTHRESHOLD CONNECTIONS:')
print(sub_net.head(10)[['source', 'target', 'Fisher_Z']].to_string())
print()
print('TOP 10 DIAGNOSED CONNECTIONS:')
print(diag_net.head(10)[['source', 'target', 'Fisher_Z']].to_string())
print()
print(f'Subthreshold edges above 0.3: {len(sub_cyto)}')
print(f'Diagnosed edges above 0.3: {len(diag_cyto)}')