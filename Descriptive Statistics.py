# Import libraries and set display options
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Specify a path
path = r"C:\Users\rsila\OneDrive\Desktop\UMich\FIN 427\FIN427"

file_path = os.path.join(path, "CSV_Dataset_CLEANED.csv")

# Import data
returns01 = pd.read_csv(file_path)

# Ensure correct datetime format
returns01['month'] = pd.to_datetime(returns01['month'], errors='coerce')
returns01['next_month'] = pd.to_datetime(returns01['next_month'], errors='coerce')

print(returns01.head(10))
print(returns01.columns)
print(returns01.dtypes)


# =========================
# Descriptive statistics for ln_shrout_change ONLY
# =========================

var = 'ln_shrout_change'

mean_val   = returns01[var].mean()
std_val    = returns01[var].std()
min_val    = returns01[var].min()
p125_val   = returns01[var].quantile(0.125)
median_val = returns01[var].median()
p875_val   = returns01[var].quantile(0.875)
max_val    = returns01[var].max()

# Create formatted table matching your screenshot style
descriptive_table = pd.DataFrame({
    "Name": [var],
    "Mean": [mean_val],
    "Std Dev": [std_val],
    "Min": [min_val],
    "12.5 p": [p125_val],
    "Median": [median_val],
    "87.5 p": [p875_val],
    "Max": [max_val]
})

print("\nDescriptive Statistics:\n")
print(descriptive_table)


# Optional: export to Excel
output_excel = os.path.join(path, "Descriptive_ln_shrout_change.xlsx")

with pd.ExcelWriter(output_excel) as writer:
    descriptive_table.to_excel(writer, sheet_name='stats', index=False)

print(f"\nSaved to:\n{output_excel}")
