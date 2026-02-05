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

# Check for missing values
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100

# Create summary dataframe
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': missing_count.values,
    'Missing Percentage': missing_pct.values,
    'Non-Missing Count': (len(df) - missing_count.values),
    'Non-Missing Percentage': (100 - missing_pct.values)
})

# Sort by missing count (descending)
missing_summary = missing_summary.sort_values('Missing Count', ascending=False)

# ============================================================================
# FIGURE 1: Missing Values Count Bar Chart
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

colors = ['#DC143C' if x > 0 else '#2E8B57' for x in missing_summary['Missing Count']]
bars = ax1.barh(missing_summary['Column'], missing_summary['Missing Count'], color=colors, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, missing_summary['Missing Count'])):
    if val > 0:
        ax1.text(val + max(missing_summary['Missing Count']) * 0.01, i, 
                f'{val:,}', va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Number of Missing Values', fontsize=12)
ax1.set_ylabel('Column', fontsize=12)
ax1.set_title('Missing Values Count by Column', fontsize=14, fontweight='bold')
ax1.set_xscale('log')  # Use log scale due to NWPERM's extreme value
ax1.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('missing_values_count.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_values_count.png")

# ============================================================================
# FIGURE 2: Missing Values Percentage Bar Chart
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

colors = ['#DC143C' if x > 0 else '#2E8B57' for x in missing_summary['Missing Percentage']]
bars = ax2.barh(missing_summary['Column'], missing_summary['Missing Percentage'], color=colors, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, missing_summary['Missing Percentage'])):
    if val > 0:
        ax2.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Missing Percentage (%)', fontsize=12)
ax2.set_ylabel('Column', fontsize=12)
ax2.set_title('Missing Values Percentage by Column', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('missing_values_percentage.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_values_percentage.png")

# ============================================================================
# FIGURE 3: Comparison - Missing vs Non-Missing
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(12, 6))

x = np.arange(len(missing_summary['Column']))
width = 0.35

bars1 = ax3.barh(x - width/2, missing_summary['Missing Count'], width, 
                 label='Missing', color='#DC143C', edgecolor='black')
bars2 = ax3.barh(x + width/2, missing_summary['Non-Missing Count'], width, 
                 label='Non-Missing', color='#2E8B57', edgecolor='black')

ax3.set_xlabel('Count', fontsize=12)
ax3.set_ylabel('Column', fontsize=12)
ax3.set_title('Missing vs Non-Missing Values by Column', fontsize=14, fontweight='bold')
ax3.set_yticks(x)
ax3.set_yticklabels(missing_summary['Column'])
ax3.set_xscale('log')
ax3.legend(loc='lower right')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('missing_vs_nonmissing.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_vs_nonmissing.png")

# ============================================================================
# FIGURE 4: Pie Chart for Overall Missing Data
# ============================================================================
fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall missing vs non-missing
total_cells = len(df) * len(df.columns)
total_missing = missing_count.sum()
total_nonmissing = total_cells - total_missing

axes[0].pie([total_missing, total_nonmissing], 
            labels=['Missing', 'Non-Missing'],
            autopct='%1.2f%%',
            colors=['#DC143C', '#2E8B57'],
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title(f'Overall Data Completeness\n(Total Cells: {total_cells:,})', 
                  fontsize=12, fontweight='bold')

# Right: Missing by column (excluding NWPERM for better visibility)
missing_excl_nwperm = missing_summary[missing_summary['Column'] != 'NWPERM']
missing_excl_nwperm = missing_excl_nwperm[missing_excl_nwperm['Missing Count'] > 0]

if len(missing_excl_nwperm) > 0:
    axes[1].pie(missing_excl_nwperm['Missing Count'], 
                labels=missing_excl_nwperm['Column'],
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*missing_excl_nwperm["Missing Count"].sum()):,})',
                startangle=90,
                textprops={'fontsize': 9})
    axes[1].set_title('Missing Values Distribution\n(Excluding NWPERM)', 
                      fontsize=12, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'No missing values\n(excluding NWPERM)', 
                ha='center', va='center', fontsize=12)
    axes[1].set_title('Missing Values Distribution\n(Excluding NWPERM)', 
                      fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('missing_pie_charts.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_pie_charts.png")

# ============================================================================
# FIGURE 5: Summary Table Visualization
# ============================================================================
fig5, ax5 = plt.subplots(figsize=(14, 6))
ax5.axis('off')

# Prepare table data
table_data = []
for _, row in missing_summary.iterrows():
    table_data.append([
        row['Column'],
        f"{row['Missing Count']:,}",
        f"{row['Missing Percentage']:.2f}%",
        f"{row['Non-Missing Count']:,}",
        f"{row['Non-Missing Percentage']:.2f}%"
    ])

table = ax5.table(cellText=table_data,
                  colLabels=['Column', 'Missing Count', 'Missing %', 'Non-Missing Count', 'Non-Missing %'],
                  cellLoc='left',
                  colColours=['#4472C4'] * 5,
                  loc='center',
                  colWidths=[0.2, 0.2, 0.15, 0.2, 0.15])

table.set_fontsize(10)
table.scale(1.2, 2.0)

# Style the header
for i in range(5):
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_facecolor('#4472C4')

# Color code rows based on missing percentage
for i in range(1, len(table_data) + 1):
    missing_pct = float(table_data[i-1][2].replace('%', ''))
    for j in range(5):
        if missing_pct > 50:
            table[(i, j)].set_facecolor('#FFE6E6')  # Light red for high missing
        elif missing_pct > 0:
            table[(i, j)].set_facecolor('#FFF4E6')  # Light orange for some missing
        else:
            table[(i, j)].set_facecolor('#E6F3E6')  # Light green for no missing

ax5.set_title('Missing Values Summary Table', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('missing_values_table.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_values_table.png")

# ============================================================================
# FIGURE 6: Dashboard - All Missing Value Visualizations
# ============================================================================
fig6 = plt.figure(figsize=(16, 10))
gs = fig6.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Top left: Missing count
ax1 = fig6.add_subplot(gs[0, 0])
colors = ['#DC143C' if x > 0 else '#2E8B57' for x in missing_summary['Missing Count']]
bars = ax1.barh(missing_summary['Column'], missing_summary['Missing Count'], color=colors, edgecolor='black')
ax1.set_xlabel('Missing Count (log scale)', fontsize=10)
ax1.set_title('Missing Count', fontsize=11, fontweight='bold')
ax1.set_xscale('log')
ax1.tick_params(labelsize=8)

# Top middle: Missing percentage
ax2 = fig6.add_subplot(gs[0, 1])
colors = ['#DC143C' if x > 0 else '#2E8B57' for x in missing_summary['Missing Percentage']]
bars = ax2.barh(missing_summary['Column'], missing_summary['Missing Percentage'], color=colors, edgecolor='black')
ax2.set_xlabel('Missing %', fontsize=10)
ax2.set_title('Missing Percentage', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.tick_params(labelsize=8)

# Top right: Pie chart
ax3 = fig6.add_subplot(gs[0, 2])
total_missing = missing_count.sum()
total_cells = len(df) * len(df.columns)
total_nonmissing = total_cells - total_missing
ax3.pie([total_missing, total_nonmissing], 
        labels=['Missing', 'Non-Missing'],
        autopct='%1.2f%%',
        colors=['#DC143C', '#2E8B57'],
        startangle=90,
        textprops={'fontsize': 9})
ax3.set_title('Overall Completeness', fontsize=11, fontweight='bold')

# Bottom: Summary table
ax4 = fig6.add_subplot(gs[1, :])
ax4.axis('off')
table_data = []
for _, row in missing_summary.iterrows():
    table_data.append([
        row['Column'],
        f"{row['Missing Count']:,}",
        f"{row['Missing Percentage']:.2f}%",
        f"{row['Non-Missing Count']:,}",
        f"{row['Non-Missing Percentage']:.2f}%"
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['Column', 'Missing Count', 'Missing %', 'Non-Missing Count', 'Non-Missing %'],
                  cellLoc='left',
                  colColours=['#4472C4'] * 5,
                  loc='center',
                  colWidths=[0.15, 0.25, 0.15, 0.25, 0.15])

table.set_fontsize(9)
table.scale(1.2, 1.8)

for i in range(5):
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_facecolor('#4472C4')

for i in range(1, len(table_data) + 1):
    missing_pct = float(table_data[i-1][2].replace('%', ''))
    for j in range(5):
        if missing_pct > 50:
            table[(i, j)].set_facecolor('#FFE6E6')
        elif missing_pct > 0:
            table[(i, j)].set_facecolor('#FFF4E6')
        else:
            table[(i, j)].set_facecolor('#E6F3E6')

fig6.suptitle('Missing Values Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('missing_values_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: missing_values_dashboard.png")

print("\n✓ All missing values visualizations generated successfully!")
print("\nFiles created:")
print("  1. missing_values_count.png - Bar chart of missing counts")
print("  2. missing_values_percentage.png - Bar chart of missing percentages")
print("  3. missing_vs_nonmissing.png - Comparison chart")
print("  4. missing_pie_charts.png - Pie charts")
print("  5. missing_values_table.png - Summary table")
print("  6. missing_values_dashboard.png - Comprehensive dashboard")
