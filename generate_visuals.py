import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read the CSV file
print("Loading data...")
df = pd.read_csv('raw.csv', low_memory=False)
print(f"Total observations: {len(df):,}")

# Focus on SHROUT (shares outstanding) - the main economic variable
shrout = df['SHROUT'].dropna()

# Define percentiles
percentiles = [0, 0.01, 0.05, 0.125, 0.25, 0.50, 0.875, 0.95, 0.99, 1.0]
percentile_labels = ['Min', '1%', '5%', '12.5%', '25%', '50%', '87.5%', '95%', '99%', 'Max']

# Calculate statistics
stats = {
    'N': len(shrout),
    'Mean': shrout.mean(),
    'Std Dev': shrout.std(),
}
for p, label in zip(percentiles, percentile_labels):
    if p == 0:
        stats[label] = shrout.min()
    elif p == 1.0:
        stats[label] = shrout.max()
    else:
        stats[label] = shrout.quantile(p)

# ============================================================================
# FIGURE 1: Formatted Statistics Table
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.axis('off')

table_data = [
    ['N (observations)', f"{stats['N']:,}"],
    ['Mean', f"{stats['Mean']:,.2f}"],
    ['Standard Deviation', f"{stats['Std Dev']:,.2f}"],
    ['', ''],
    ['Minimum', f"{stats['Min']:,.2f}"],
    ['1st Percentile', f"{stats['1%']:,.2f}"],
    ['5th Percentile', f"{stats['5%']:,.2f}"],
    ['12.5th Percentile', f"{stats['12.5%']:,.2f}"],
    ['25th Percentile (Q1)', f"{stats['25%']:,.2f}"],
    ['50th Percentile (Median)', f"{stats['50%']:,.2f}"],
    ['87.5th Percentile', f"{stats['87.5%']:,.2f}"],
    ['95th Percentile', f"{stats['95%']:,.2f}"],
    ['99th Percentile', f"{stats['99%']:,.2f}"],
    ['Maximum', f"{stats['Max']:,.2f}"],
]

table = ax1.table(cellText=table_data,
                  colLabels=['Statistic', 'SHROUT (thousands)'],
                  cellLoc='left',
                  colColours=['#4472C4', '#4472C4'],
                  loc='center',
                  colWidths=[0.5, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style the header
for i in range(2):
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_facecolor('#4472C4')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#D9E2F3')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

ax1.set_title('Descriptive Statistics: Shares Outstanding (SHROUT)\n', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('table_descriptive_stats.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: table_descriptive_stats.png")

# ============================================================================
# FIGURE 2: Histogram with Log Scale (due to extreme skewness)
# ============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Regular histogram (truncated for visibility)
shrout_trimmed = shrout[shrout <= shrout.quantile(0.95)]
axes[0].hist(shrout_trimmed, bins=50, color='#4472C4', edgecolor='white', alpha=0.8)
axes[0].axvline(shrout.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {shrout.mean():,.0f}')
axes[0].axvline(shrout.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {shrout.median():,.0f}')
axes[0].set_xlabel('Shares Outstanding (thousands)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of SHROUT (≤95th percentile)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].ticklabel_format(style='plain', axis='x')

# Right: Log-transformed histogram
shrout_positive = shrout[shrout > 0]
axes[1].hist(np.log10(shrout_positive), bins=50, color='#ED7D31', edgecolor='white', alpha=0.8)
axes[1].axvline(np.log10(shrout_positive.mean()), color='red', linestyle='--', linewidth=2, label=f'Mean')
axes[1].axvline(np.log10(shrout_positive.median()), color='green', linestyle='--', linewidth=2, label=f'Median')
axes[1].set_xlabel('Log₁₀(Shares Outstanding)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Log Distribution of SHROUT', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('histogram_shrout.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: histogram_shrout.png")

# ============================================================================
# FIGURE 3: Box Plot
# ============================================================================
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Box plot (trimmed)
axes[0].boxplot(shrout_trimmed, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#4472C4', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
axes[0].set_ylabel('Shares Outstanding (thousands)', fontsize=11)
axes[0].set_title('Box Plot of SHROUT (≤95th percentile)', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['SHROUT'])

# Right: Box plot with log scale
axes[1].boxplot(shrout_positive, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#ED7D31', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
axes[1].set_yscale('log')
axes[1].set_ylabel('Shares Outstanding (thousands) - Log Scale', fontsize=11)
axes[1].set_title('Box Plot of SHROUT (Log Scale)', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(['SHROUT'])

plt.tight_layout()
plt.savefig('boxplot_shrout.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: boxplot_shrout.png")

# ============================================================================
# FIGURE 4: Percentile Bar Chart
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(12, 6))

percentile_values = [stats[label] for label in percentile_labels]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(percentile_labels)))

bars = ax4.bar(percentile_labels, percentile_values, color=colors, edgecolor='black')
ax4.set_yscale('log')
ax4.set_xlabel('Percentile', fontsize=12)
ax4.set_ylabel('Shares Outstanding (thousands) - Log Scale', fontsize=12)
ax4.set_title('SHROUT by Percentile', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, val in zip(bars, percentile_values):
    height = bar.get_height()
    ax4.annotate(f'{val:,.0f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, rotation=45)

plt.tight_layout()
plt.savefig('percentiles_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: percentiles_chart.png")

# ============================================================================
# FIGURE 5: Summary Dashboard
# ============================================================================
fig5 = plt.figure(figsize=(16, 10))

# Create grid
gs = fig5.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Top left: Stats table
ax_table = fig5.add_subplot(gs[0, 0])
ax_table.axis('off')
mini_table_data = [
    ['N', f"{stats['N']:,}"],
    ['Mean', f"{stats['Mean']:,.2f}"],
    ['Std Dev', f"{stats['Std Dev']:,.2f}"],
    ['Median', f"{stats['50%']:,.2f}"],
    ['Min', f"{stats['Min']:,.2f}"],
    ['Max', f"{stats['Max']:,.2f}"],
]
mini_table = ax_table.table(cellText=mini_table_data,
                            colLabels=['Statistic', 'Value'],
                            cellLoc='left',
                            colColours=['#4472C4', '#4472C4'],
                            loc='center',
                            colWidths=[0.4, 0.5])
mini_table.auto_set_font_size(False)
mini_table.set_fontsize(10)
mini_table.scale(1.2, 1.5)
for i in range(2):
    mini_table[(0, i)].set_text_props(weight='bold', color='white')
ax_table.set_title('Key Statistics', fontsize=12, fontweight='bold')

# Top middle: Histogram
ax_hist = fig5.add_subplot(gs[0, 1])
ax_hist.hist(shrout_trimmed, bins=40, color='#4472C4', edgecolor='white', alpha=0.8)
ax_hist.axvline(shrout.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax_hist.axvline(shrout.median(), color='green', linestyle='--', linewidth=2, label='Median')
ax_hist.set_xlabel('SHROUT (thousands)')
ax_hist.set_title('Distribution (≤95th pctl)', fontsize=12, fontweight='bold')
ax_hist.legend(fontsize=8)

# Top right: Log histogram
ax_log = fig5.add_subplot(gs[0, 2])
ax_log.hist(np.log10(shrout_positive), bins=40, color='#ED7D31', edgecolor='white', alpha=0.8)
ax_log.set_xlabel('Log₁₀(SHROUT)')
ax_log.set_title('Log Distribution', fontsize=12, fontweight='bold')

# Bottom: Percentile chart (spanning full width)
ax_pct = fig5.add_subplot(gs[1, :])
bars = ax_pct.bar(percentile_labels, percentile_values, color=colors, edgecolor='black')
ax_pct.set_yscale('log')
ax_pct.set_xlabel('Percentile', fontsize=12)
ax_pct.set_ylabel('SHROUT (thousands) - Log Scale', fontsize=12)
ax_pct.set_title('Shares Outstanding by Percentile', fontsize=12, fontweight='bold')
for bar, val in zip(bars, percentile_values):
    height = bar.get_height()
    ax_pct.annotate(f'{val:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

fig5.suptitle('Descriptive Statistics Dashboard: SHROUT (Shares Outstanding)', 
              fontsize=16, fontweight='bold', y=0.98)
plt.savefig('dashboard_shrout.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: dashboard_shrout.png")

print("\n✓ All visualizations generated successfully!")
print("\nFiles created:")
print("  1. table_descriptive_stats.png - Formatted statistics table")
print("  2. histogram_shrout.png - Distribution histograms")
print("  3. boxplot_shrout.png - Box plots")
print("  4. percentiles_chart.png - Percentile bar chart")
print("  5. dashboard_shrout.png - Summary dashboard")
