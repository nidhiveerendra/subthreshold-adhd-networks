"""
Analyze demographics of final sample (201 subjects)
Age, sex, ADHD Index scores by group
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load phenotypic data
df = pd.read_csv('../../data/phenotypic/NYU_phenotypic.csv')

# Load successful subjects from preprocessing
with open('../../data/processed/preprocessing_log.txt', 'r') as f:
    lines = f.readlines()

successful_subjects = []
reading = False
for line in lines:
    if 'SUCCESSFUL SUBJECTS:' in line:
        reading = True
        continue
    if 'FAILED SUBJECTS:' in line:
        break
    if reading and line.strip().startswith('sub-'):
        # Convert back to ScanDir ID format
        subject_id = int(line.strip().replace('sub-', ''))
        successful_subjects.append(subject_id)

print(f"Total successful subjects: {len(successful_subjects)}")

# Filter to only successful subjects
df_success = df[df['ScanDir ID'].isin(successful_subjects)].copy()

# Create groups based on ADHD Index
index_col = 'ADHD Index'
df_success['Group'] = 'Unknown'
valid_scores = df_success[index_col] != -999

df_success.loc[valid_scores & (df_success[index_col] >= 40) & (df_success[index_col] < 60), 'Group'] = 'Subthreshold'
df_success.loc[valid_scores & (df_success[index_col] >= 60), 'Group'] = 'Diagnosed'

# Filter to analysis groups
df_analysis = df_success[df_success['Group'].isin(['Subthreshold', 'Diagnosed'])].copy()

print("FINAL SAMPLE DEMOGRAPHICS")

# Overall statistics
print(f"\nTotal sample size: {len(df_analysis)}")
print(f"  Subthreshold: {(df_analysis['Group'] == 'Subthreshold').sum()}")
print(f"  Diagnosed: {(df_analysis['Group'] == 'Diagnosed').sum()}")

# Age statistics
print("AGE DISTRIBUTION")

print(f"\nOverall:")
print(f"  Mean: {df_analysis['Age'].mean():.2f} ± {df_analysis['Age'].std():.2f}")
print(f"  Range: {df_analysis['Age'].min():.1f} - {df_analysis['Age'].max():.1f}")
print(f"  Median: {df_analysis['Age'].median():.1f}")

print(f"\nBy Group:")
for group in ['Subthreshold', 'Diagnosed']:
    group_data = df_analysis[df_analysis['Group'] == group]['Age']
    print(f"  {group}:")
    print(f"    Mean: {group_data.mean():.2f} ± {group_data.std():.2f}")
    print(f"    Range: {group_data.min():.1f} - {group_data.max():.1f}")
    print(f"    Median: {group_data.median():.1f}")

# Test age difference between groups
sub_age = df_analysis[df_analysis['Group'] == 'Subthreshold']['Age']
diag_age = df_analysis[df_analysis['Group'] == 'Diagnosed']['Age']
t_stat, p_val = stats.ttest_ind(sub_age, diag_age)
print(f"\nAge difference between groups: t = {t_stat:.3f}, p = {p_val:.3f}")
if p_val < 0.05:
    print("  Groups differ significantly in age - MUST control for age in analyses!")
else:
    print("  Groups are age-matched")

# Sex distribution
print("SEX DISTRIBUTION")

print(f"\nOverall:")
sex_counts = df_analysis['Gender'].value_counts()
print(f"  Male: {sex_counts.get(1, 0)} ({sex_counts.get(1, 0)/len(df_analysis)*100:.1f}%)")
print(f"  Female: {sex_counts.get(0, 0)} ({sex_counts.get(0, 0)/len(df_analysis)*100:.1f}%)")

print(f"\nBy Group:")
for group in ['Subthreshold', 'Diagnosed']:
    group_data = df_analysis[df_analysis['Group'] == group]
    male = (group_data['Gender'] == 1).sum()
    female = (group_data['Gender'] == 0).sum()
    total = len(group_data)
    print(f"  {group}:")
    print(f"    Male: {male} ({male/total*100:.1f}%)")
    print(f"    Female: {female} ({female/total*100:.1f}%)")

# Chi-square test for sex differences
ct = pd.crosstab(df_analysis['Group'], df_analysis['Gender'])
chi2, p_val, dof, expected = stats.chi2_contingency(ct)
print(f"\nSex distribution difference: χ² = {chi2:.3f}, p = {p_val:.3f}")
if p_val < 0.05:
    print("  Groups differ in sex distribution - should control for sex!")
else:
    print("  Groups have similar sex distributions")

# ADHD Index scores
print("ADHD INDEX SCORES")

print(f"\nOverall:")
print(f"  Mean: {df_analysis[index_col].mean():.2f} ± {df_analysis[index_col].std():.2f}")
print(f"  Range: {df_analysis[index_col].min():.0f} - {df_analysis[index_col].max():.0f}")

print(f"\nBy Group:")
for group in ['Subthreshold', 'Diagnosed']:
    group_data = df_analysis[df_analysis['Group'] == group][index_col]
    print(f"  {group}:")
    print(f"    Mean: {group_data.mean():.2f} ± {group_data.std():.2f}")
    print(f"    Range: {group_data.min():.0f} - {group_data.max():.0f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
ax = axes[0, 0]
for group in ['Subthreshold', 'Diagnosed']:
    data = df_analysis[df_analysis['Group'] == group]['Age']
    ax.hist(data, alpha=0.6, label=group, bins=15)
ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Age Distribution by Group', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Age boxplot
ax = axes[0, 1]
df_analysis.boxplot(column='Age', by='Group', ax=ax)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Age (years)', fontsize=12)
ax.set_title('Age Comparison', fontsize=14, fontweight='bold')
plt.sca(ax)
plt.xticks(rotation=0)

# Sex distribution
ax = axes[1, 0]
sex_by_group = df_analysis.groupby(['Group', 'Gender']).size().unstack(fill_value=0)
sex_by_group.plot(kind='bar', ax=ax, color=['pink', 'lightblue'])
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Sex Distribution by Group', fontsize=14, fontweight='bold')
ax.legend(['Female', 'Male'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(alpha=0.3)

# ADHD Index scores
ax = axes[1, 1]
df_analysis.boxplot(column=index_col, by='Group', ax=ax)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('ADHD Index Score', fontsize=12)
ax.set_title('ADHD Symptom Severity', fontsize=14, fontweight='bold')
ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Diagnostic threshold')
plt.sca(ax)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('../../results/figures/sample_demographics.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization: results/figures/sample_demographics.png")

# Save summary to file
with open('../../results/demographics_summary.txt', 'w') as f:
    f.write("FINAL SAMPLE DEMOGRAPHICS\n")
    f.write(f"Total N: {len(df_analysis)}\n")
    f.write(f"Subthreshold: {(df_analysis['Group'] == 'Subthreshold').sum()}\n")
    f.write(f"Diagnosed: {(df_analysis['Group'] == 'Diagnosed').sum()}\n\n")
    
    f.write("AGE:\n")
    f.write(f"  Overall: {df_analysis['Age'].mean():.2f} ± {df_analysis['Age'].std():.2f} (range: {df_analysis['Age'].min():.1f}-{df_analysis['Age'].max():.1f})\n")
    f.write(f"  Subthreshold: {sub_age.mean():.2f} ± {sub_age.std():.2f}\n")
    f.write(f"  Diagnosed: {diag_age.mean():.2f} ± {diag_age.std():.2f}\n")
    f.write(f"  Group difference: t={t_stat:.3f}, p={p_val:.3f}\n\n")
    
    f.write("SEX:\n")
    f.write(f"  Male: {sex_counts.get(1, 0)} ({sex_counts.get(1, 0)/len(df_analysis)*100:.1f}%)\n")
    f.write(f"  Female: {sex_counts.get(0, 0)} ({sex_counts.get(0, 0)/len(df_analysis)*100:.1f}%)\n")

print(f"Saved summary: results/demographics_summary.txt")
print("DEMOGRAPHICS ANALYSIS COMPLETE!")