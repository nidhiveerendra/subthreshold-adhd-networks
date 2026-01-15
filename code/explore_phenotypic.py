import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Create results folder
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("ADHD-200 NYU PHENOTYPIC DATA EXPLORATION")

# Load data
df = pd.read_csv('data/phenotypic/NYU_phenotypic.csv')

print(f"Total subjects: {len(df)}\n")

# Handle missing data codes
index_col = 'ADHD Index'
n_missing = (df[index_col] == -999).sum()
print(f"Subjects with -999 (missing code): {n_missing}")

# Create groups
df['Group'] = 'Unknown'
valid_scores = df[index_col] != -999
df.loc[valid_scores & (df[index_col] < 40), 'Group'] = 'Neurotypical'
df.loc[valid_scores & (df[index_col] >= 40) & (df[index_col] < 60), 'Group'] = 'Subthreshold'
df.loc[valid_scores & (df[index_col] >= 60), 'Group'] = 'Diagnosed'

# Add neurotypical controls from DX=0
if 'DX' in df.columns:
    df.loc[(df[index_col] == -999) & (df['DX'] == 0), 'Group'] = 'Neurotypical'

# Count groups
n_neuro = len(df[df['Group'] == 'Neurotypical'])
n_sub = len(df[df['Group'] == 'Subthreshold'])
n_diag = len(df[df['Group'] == 'Diagnosed'])

print(" FINAL SAMPLE SIZES:")
print(f" Neurotypical:    {n_neuro:3d} subjects")
print(f" SUBTHRESHOLD:   {n_sub:3d} subjects   YOUR NOVEL GROUP!")
print(f" Diagnosed:      {n_diag:3d} subjects")

# Create visualizations

df_plot = df[df['Group'].isin(['Neurotypical', 'Subthreshold', 'Diagnosed'])].copy()
valid_scores_only = df[df[index_col] != -999][index_col]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ADHD Index Distribution
axes[0, 0].hist(valid_scores_only, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(40, color='green', linestyle='--', linewidth=2, label='Subthreshold (T=40)')
axes[0, 0].axvline(60, color='red', linestyle='--', linewidth=2, label='Diagnostic (T=60)')
axes[0, 0].set_xlabel('ADHD Index T-Score', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0, 0].set_title('ADHD Index Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Group Sizes
groups = df_plot['Group'].value_counts()
colors = {'Neurotypical': '#2ecc71', 'Subthreshold': '#f39c12', 'Diagnosed': '#e74c3c'}
group_colors = [colors[g] for g in groups.index]
bars = axes[0, 1].bar(groups.index, groups.values, color=group_colors, edgecolor='black', alpha=0.8)
axes[0, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Sample Size by Group', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

for i, (group, count) in enumerate(groups.items()):
    axes[0, 1].text(i, count + 2, str(count), ha='center', fontsize=12, fontweight='bold')

if 'Subthreshold' in groups.index:
    sub_idx = list(groups.index).index('Subthreshold')
    bars[sub_idx].set_edgecolor('gold')
    bars[sub_idx].set_linewidth(3)

# Plot 3: Age by Group
if 'Age' in df.columns:
    age_data = []
    labels = []
    group_colors_list = []
    for group in ['Subthreshold', 'Diagnosed']:  # Skip neurotypical (too small)
        ages = df_plot[df_plot['Group'] == group]['Age'].dropna()
        if len(ages) > 0:
            age_data.append(ages)
            labels.append(group)
            group_colors_list.append(colors[group])
    
    if len(age_data) > 0:
        bp = axes[1, 0].boxplot(age_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], group_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    axes[1, 0].set_ylabel('Age (years)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Age Distribution by Group', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Sex by Group
sex_col = 'Gender'
if sex_col in df.columns:
    sex_data = pd.crosstab(df_plot['Group'], df_plot[sex_col])
    sex_data.plot(kind='bar', ax=axes[1, 1], color=['#3498db', '#e91e63'], 
                 edgecolor='black', alpha=0.8)
    axes[1, 1].set_xlabel('Group', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Sex Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend(['Female (0)', 'Male (1)'])
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/sample_characteristics.png', dpi=300, bbox_inches='tight')
print(f"Figure saved: results/figures/sample_characteristics.png")

# Save summary
with open('results/sample_summary.txt', 'w') as f:
    f.write("NYU ADHD-200 SAMPLE SUMMARY\n")
    f.write(f"Total subjects: {len(df)}\n")
    f.write(f"Valid ADHD scores: {(df[index_col] != -999).sum()}\n\n")
    f.write("GROUP SIZES:\n")
    f.write(f"  Neurotypical:   {n_neuro:3d}\n")
    f.write(f"  Subthreshold:   {n_sub:3d}  ⭐\n")
    f.write(f"  Diagnosed:      {n_diag:3d}\n\n")
    f.write(f"Age: {df['Age'].mean():.1f} ± {df['Age'].std():.1f} years\n")
    f.write(f"Date: 2026-01-11\n")

print(f"Summary saved: results/sample_summary.txt")
print(f"\nKEY: You have {n_sub} SUBTHRESHOLD subjects! \n")