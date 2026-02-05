import pandas as pd
import numpy as np

# Read the CSV file
print("Loading data...")
df = pd.read_csv('raw.csv', low_memory=False)

print(f"Total observations: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Identify numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns: {numeric_cols}")

# Define the percentiles we need
percentiles = [0, 0.01, 0.05, 0.125, 0.25, 0.50, 0.875, 0.95, 0.99, 1.0]
percentile_labels = ['Min', '1%', '5%', '12.5%', '25%', '50%', '87.5%', '95%', '99%', 'Max']

# Calculate descriptive statistics for each numeric column
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

results = {}

for col in numeric_cols:
    # Skip if column is mostly empty
    non_null_count = df[col].notna().sum()
    if non_null_count < 100:
        print(f"\n{col}: Skipped (only {non_null_count} non-null values)")
        continue
    
    data = df[col].dropna()
    
    stats = {
        'N': len(data),
        'Mean': data.mean(),
        'Std Dev': data.std(),
    }
    
    # Calculate percentiles
    for p, label in zip(percentiles, percentile_labels):
        if p == 0:
            stats[label] = data.min()
        elif p == 1.0:
            stats[label] = data.max()
        else:
            stats[label] = data.quantile(p)
    
    results[col] = stats

# Create a summary table
print("\n")
for col, stats in results.items():
    print(f"\n{'='*60}")
    print(f"Variable: {col}")
    print(f"{'='*60}")
    print(f"{'N (observations)':<20}: {stats['N']:>15,}")
    print(f"{'Mean':<20}: {stats['Mean']:>15,.2f}")
    print(f"{'Standard Deviation':<20}: {stats['Std Dev']:>15,.2f}")
    print(f"\nPercentiles:")
    print("-"*40)
    for label in percentile_labels:
        print(f"  {label:<12}: {stats[label]:>20,.2f}")

# Create a formatted table for all variables
print("\n\n" + "="*80)
print("SUMMARY TABLE - ALL NUMERIC VARIABLES")
print("="*80)

# Create DataFrame for nice display
summary_rows = ['N', 'Mean', 'Std Dev'] + percentile_labels
summary_data = {}

for col in results:
    summary_data[col] = [
        f"{results[col]['N']:,}",
        f"{results[col]['Mean']:,.2f}",
        f"{results[col]['Std Dev']:,.2f}",
    ]
    for label in percentile_labels:
        summary_data[col].append(f"{results[col][label]:,.2f}")

summary_df = pd.DataFrame(summary_data, index=summary_rows)
print("\n")
print(summary_df.to_string())

# Save to CSV
summary_df.to_csv('descriptive_statistics.csv')
print("\n\nResults saved to 'descriptive_statistics.csv'")
