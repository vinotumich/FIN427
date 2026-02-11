# Import libraries and set display options
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 10)
pd.options.display.float_format = '{:.10f}'.format

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
s = returns01[var]

mean_val = s.mean()
std_val  = s.std()

# Percentiles you want
percentiles = {
    "Min":    s.min(),
    "1%":     s.quantile(0.01),
    "5%":     s.quantile(0.05),
    "12.5%":  s.quantile(0.125),
    "25%":    s.quantile(0.25),
    "50%":    s.quantile(0.50),
    "87.5%":  s.quantile(0.875),
    "95%":    s.quantile(0.95),
    "99%":    s.quantile(0.99),
    "Max":    s.max()
}

# Create formatted table (one row, like your screenshot)
descriptive_table = pd.DataFrame({
    "Name": [var],
    "Mean": [mean_val],
    "Std Dev": [std_val],
    "Min": [percentiles["Min"]],
    "1%": [percentiles["1%"]],
    "5%": [percentiles["5%"]],
    "12.5%": [percentiles["12.5%"]],
    "25%": [percentiles["25%"]],
    "50%": [percentiles["50%"]],
    "87.5%": [percentiles["87.5%"]],
    "95%": [percentiles["95%"]],
    "99%": [percentiles["99%"]],
    "Max": [percentiles["Max"]]
}).round(10)

print("\nDescriptive Statistics:\n")
print(descriptive_table)

# Optional: export to Excel
output_excel = os.path.join(path, "Descriptive_ln_shrout_change.xlsx")

with pd.ExcelWriter(output_excel) as writer:
    descriptive_table.to_excel(writer, sheet_name='stats', index=False)

print(f"\nSaved to:\n{output_excel}")
