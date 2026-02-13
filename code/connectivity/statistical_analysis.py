"""
Statistical analysis of graph metrics vs ADHD symptoms
Tests the main hypothesis: Do graph metrics correlate with symptom severity?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("STATISTICAL ANALYSIS - TESTING HYPOTHESIS")

# 1. Load graph metrics
print("\n1. Loading graph metrics...")
df_metrics = pd.read_csv('../../data/graph_metrics/all_subjects_graph_metrics.csv')
print(f"   Loaded {len(df_metrics)} subjects with graph metrics")

# 2. Load phenotypic data
print("\n2. Loading phenotypic data...")
df_pheno = pd.read_csv('../../data/phenotypic/NYU_phenotypic.csv')

# Create subject ID column in phenotypic data
df_pheno['subject_id'] = df_pheno['ScanDir ID'].apply(lambda x: f'sub-{x:07d}')

# 3. Merge datasets
print("\n3. Merging graph metrics with phenotypic data...")
df = df_metrics.merge(df_pheno, on='subject_id', how='left')
print(f"   Merged dataset: {len(df)} subjects")

# 4. Filter to analysis groups (subthreshold + diagnosed)
print("\n4. Creating analysis groups...")
index_col = 'ADHD Index'
df['Group'] = 'Unknown'
valid_scores = df[index_col] != -999

df.loc[valid_scores & (df[index_col] >= 40) & (df[index_col] < 60), 'Group'] = 'Subthreshold'
df.loc[valid_scores & (df[index_col] >= 60), 'Group'] = 'Diagnosed'

df_analysis = df[df['Group'].isin(['Subthreshold', 'Diagnosed'])].copy()

print(f"   Analysis sample: {len(df_analysis)} subjects")
print(f"   - Subthreshold: {(df_analysis['Group'] == 'Subthreshold').sum()}")
print(f"   - Diagnosed: {(df_analysis['Group'] == 'Diagnosed').sum()}")

# 5. MAIN ANALYSIS: Correlations with ADHD Index
print("HYPOTHESIS TEST: Do graph metrics correlate with symptom severity?")

# Metrics to analyze
metrics_to_test = [
    'clustering_coefficient',
    'path_length',
    'global_efficiency',
    'modularity',
    'small_worldness',
    'mean_degree'
]

results = []

print("\nCorrelations with ADHD Index Score:")

for metric in metrics_to_test:
    # Remove NaN values
    valid_data = df_analysis[[metric, index_col]].dropna()
    
    if len(valid_data) < 10:
        print(f"\n{metric}: SKIPPED (insufficient data)")
        continue
    
    # Spearman correlation (non-parametric, robust to outliers)
    rho, p_value = stats.spearmanr(valid_data[metric], valid_data[index_col])
    
    # Store results
    results.append({
        'metric': metric,
        'rho': rho,
        'p_value': p_value,
        'n': len(valid_data)
    })
    
    # Determine significance
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    print(f"\n{metric}:")
    print(f"   Spearman rho = {rho:.4f}")
    print(f"   p-value = {p_value:.4f} {sig}")
    print(f"   n = {len(valid_data)}")
    
    if abs(rho) > 0.3:
        print(f"    MEDIUM TO LARGE EFFECT SIZE!")
    elif abs(rho) > 0.1:
        print(f"   → Small to medium effect")

# 6. Multiple comparison correction
print("MULTIPLE COMPARISON CORRECTION (FDR)")

df_results = pd.DataFrame(results)
p_values = df_results['p_value'].values

# FDR correction
rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

df_results['p_adjusted'] = p_adjusted
df_results['significant_fdr'] = rejected

print("\nAfter FDR correction (α = 0.05):")
for _, row in df_results.iterrows():
    sig = " SIGNIFICANT" if row['significant_fdr'] else "  Not significant"
    print(f"{row['metric']:30s} p_adj = {row['p_adjusted']:.4f}  {sig}")

# 7. Group comparisons
print("GROUP COMPARISONS: Subthreshold vs Diagnosed")

group_results = []

for metric in metrics_to_test:
    subthreshold = df_analysis[df_analysis['Group'] == 'Subthreshold'][metric].dropna()
    diagnosed = df_analysis[df_analysis['Group'] == 'Diagnosed'][metric].dropna()
    
    if len(subthreshold) < 5 or len(diagnosed) < 5:
        continue
    
    # Independent t-test
    t_stat, p_value = stats.ttest_ind(subthreshold, diagnosed)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(subthreshold)-1)*subthreshold.std()**2 + 
                          (len(diagnosed)-1)*diagnosed.std()**2) / 
                         (len(subthreshold) + len(diagnosed) - 2))
    cohens_d = (subthreshold.mean() - diagnosed.mean()) / pooled_std
    
    group_results.append({
        'metric': metric,
        'subthreshold_mean': subthreshold.mean(),
        'diagnosed_mean': diagnosed.mean(),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    })
    
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    print(f"\n{metric}:")
    print(f"   Subthreshold: {subthreshold.mean():.4f} ± {subthreshold.std():.4f}")
    print(f"   Diagnosed:    {diagnosed.mean():.4f} ± {diagnosed.std():.4f}")
    print(f"   t = {t_stat:.4f}, p = {p_value:.4f} {sig}")
    print(f"   Cohen's d = {cohens_d:.4f}")

# 8. Save results
print("SAVING RESULTS")

# Save correlation results
df_results.to_csv('../../results/correlation_results.csv', index=False)
print(" Saved: results/correlation_results.csv")

# Save group comparison results
df_group_results = pd.DataFrame(group_results)
df_group_results.to_csv('../../results/group_comparison_results.csv', index=False)
print(" Saved: results/group_comparison_results.csv")

# 9. Create visualizations
print("\n9. Creating visualizations...")

# Figure 1: Correlation plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_test):
    ax = axes[i]
    
    valid_data = df_analysis[[metric, index_col, 'Group']].dropna()
    
    # Scatter plot with regression line
    for group, color in [('Subthreshold', 'gold'), ('Diagnosed', 'red')]:
        group_data = valid_data[valid_data['Group'] == group]
        ax.scatter(group_data[metric], group_data[index_col], 
                  alpha=0.6, color=color, label=group, s=50)
    
    # Overall regression line
    z = np.polyfit(valid_data[metric], valid_data[index_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data[metric].min(), valid_data[metric].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2)
    
    # Add correlation info
    result = df_results[df_results['metric'] == metric].iloc[0]
    ax.text(0.05, 0.95, f"ρ = {result['rho']:.3f}\np = {result['p_value']:.3f}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('ADHD Index Score', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../results/figures/correlation_plots.png', dpi=300, bbox_inches='tight')
print(" Saved: results/figures/correlation_plots.png")

# Figure 2: Group comparisons
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_test):
    ax = axes[i]
    
    data_to_plot = [
        df_analysis[df_analysis['Group'] == 'Subthreshold'][metric].dropna(),
        df_analysis[df_analysis['Group'] == 'Diagnosed'][metric].dropna()
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Subthreshold', 'Diagnosed'],
                    patch_artist=True)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('gold')
    bp['boxes'][1].set_facecolor('red')
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_xlabel('Group', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../../results/figures/group_comparison_plots.png', dpi=300, bbox_inches='tight')
print(" Saved: results/figures/group_comparison_plots.png")

# 10. Summary
print("ANALYSIS COMPLETE!")

print(f"\nSignificant correlations (after FDR correction):")
significant = df_results[df_results['significant_fdr']]
if len(significant) > 0:
    for _, row in significant.iterrows():
        print(f"    {row['metric']}: ρ = {row['rho']:.4f}, p_adj = {row['p_adjusted']:.4f}")
else:
    print("   No metrics survived FDR correction")

print(f"\nSignificant group differences (p < 0.05):")
sig_groups = df_group_results[df_group_results['p_value'] < 0.05]
if len(sig_groups) > 0:
    for _, row in sig_groups.iterrows():
        print(f"    {row['metric']}: t = {row['t_stat']:.3f}, p = {row['p_value']:.4f}, d = {row['cohens_d']:.3f}")
else:
    print("   No significant group differences")

print(" HYPOTHESIS TESTING COMPLETE!")