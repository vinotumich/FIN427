import pandas as pd
import numpy as np

# Read the CSV file
print("Loading data...")
df = pd.read_csv('raw.csv', low_memory=False)

print(f"Total observations: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\n" + "="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

# Check for missing values
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100

# Create summary table
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': missing_count.values,
    'Missing Percentage': missing_pct.values,
    'Non-Missing Count': (len(df) - missing_count.values),
    'Non-Missing Percentage': (100 - missing_pct.values)
})

# Sort by missing count (descending)
missing_summary = missing_summary.sort_values('Missing Count', ascending=False)

print("\nMissing Values Summary:")
print("-"*80)
print(missing_summary.to_string(index=False))

# Check for empty strings (which might be considered missing)
print("\n" + "="*80)
print("EMPTY STRING ANALYSIS")
print("="*80)

empty_strings = {}
for col in df.columns:
    if df[col].dtype == 'object':  # String columns
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            empty_strings[col] = empty_count

if empty_strings:
    print("\nColumns with empty strings:")
    for col, count in empty_strings.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("\nNo empty strings found.")

# Check for zeros in numeric columns (which might be meaningful or missing)
print("\n" + "="*80)
print("ZERO VALUES IN NUMERIC COLUMNS")
print("="*80)

for col in df.select_dtypes(include=[np.number]).columns:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        pct = (zero_count / len(df)) * 100
        print(f"  {col}: {zero_count:,} zeros ({pct:.2f}%)")

# Overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
total_missing = missing_count.sum()
total_cells = len(df) * len(df.columns)
overall_missing_pct = (total_missing / total_cells) * 100

print(f"Total missing values across all columns: {total_missing:,}")
print(f"Total cells in dataset: {total_cells:,}")
print(f"Overall missing percentage: {overall_missing_pct:.2f}%")

# Save to CSV
missing_summary.to_csv('missing_values_analysis.csv', index=False)
print("\n✓ Results saved to 'missing_values_analysis.csv'")
