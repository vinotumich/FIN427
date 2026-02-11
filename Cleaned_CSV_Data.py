import os
import numpy as np
import pandas as pd

# CONFIG (path + file)

BASE_DIR = r"C:\Users\rsila\OneDrive\Desktop\UMich\FIN 427\FIN427"
IN_FILE = "CSV_Dataset.csv"
OUT_FILE = "CSV_Dataset_CLEANED.csv"

in_path = os.path.join(BASE_DIR, IN_FILE)
out_path = os.path.join(BASE_DIR, OUT_FILE)

# Memory Settings Safety
CHUNKSIZE = 300_000

# Drop NWPERM by not reading it
usecols = ["PERMNO", "date", "TICKER", "COMNAM", "PERMCO", "CUSIP", "SHROUT"]

dtype = {
    "PERMNO": "int32",
    "TICKER": "string",
    "COMNAM": "string",
    "PERMCO": "int32",
    "CUSIP": "string",
    "SHROUT": "float64",
}

# Remove prior output if exists (important because we append chunks)
if os.path.exists(out_path):
    os.remove(out_path)

# Carryover across chunks
last_shrout = {}      # PERMNO -> last SHROUT seen (for cross-chunk lag)
seen_permno = set()   # PERMNOs already encountered in earlier chunks

header_written = False
chunk_num = 0

for chunk in pd.read_csv(
    in_path,
    usecols=usecols,
    dtype=dtype,
    parse_dates=["date"],
    na_values=["", "NA", "NaN"],
    chunksize=CHUNKSIZE,
):
    chunk_num += 1
    print(f"Processing chunk {chunk_num}...")

    # Drop placeholder rows where TICKER is blank/missing
    t = chunk["TICKER"]
    chunk = chunk[t.notna() & (t.str.len() > 0)].copy()

    # Ensure SHROUT numeric
    chunk["SHROUT"] = pd.to_numeric(chunk["SHROUT"], errors="coerce")

    # Ensure date is datetime (parse_dates already does this, but safe)
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

    # Next month-end date (aligned with month-end convention)
    chunk["next_date"] = chunk["date"] + pd.offsets.MonthEnd(1)

    # Lag within chunk
    chunk["SHROUT_LAG"] = chunk.groupby("PERMNO", sort=False)["SHROUT"].shift(1)

    # Patch first row per PERMNO in this chunk using carryover last_shrout
    first_row_each_permno_in_chunk = ~chunk["PERMNO"].duplicated()
    first_idx = chunk.index[first_row_each_permno_in_chunk]

    first_permnos = chunk.loc[first_idx, "PERMNO"].to_numpy()
    carry_lags = np.array([last_shrout.get(int(p), np.nan) for p in first_permnos], dtype="float64")

    current_lags = chunk.loc[first_idx, "SHROUT_LAG"].to_numpy(dtype="float64")
    patched_lags = np.where(np.isnan(current_lags), carry_lags, current_lags)
    chunk.loc[first_idx, "SHROUT_LAG"] = patched_lags

    # ============================================================
    # Compute raw monthly change
    # ============================================================
    chunk["d_shrout_raw"] = chunk["SHROUT"] - chunk["SHROUT_LAG"]

    # ============================================================
    # ALLOCATE quarterly jump over current + prior 2 months
    # If change happens at t, allocate (change/3) to t, t-1, t-2.
    g = chunk.groupby("PERMNO", sort=False)["d_shrout_raw"]
    chunk["d_shrout"] = (g.shift(0) + g.shift(-1) + g.shift(-2)) / 3.0

    shrout_monthly = chunk["SHROUT_LAG"] + chunk["d_shrout"]
    valid = (shrout_monthly > 0) & (chunk["SHROUT_LAG"] > 0)
    chunk["ln_shrout_change"] = np.where(valid, np.log(shrout_monthly / chunk["SHROUT_LAG"]), np.nan)

    # FIRST EVER observation per PERMNO in the ENTIRE FILE:
    first_ever_mask = first_row_each_permno_in_chunk & (~chunk["PERMNO"].isin(seen_permno))

    # Apply first-observation behavior (same as your original)
    chunk.loc[first_ever_mask, "d_shrout"] = 0.0
    chunk.loc[first_ever_mask, "ln_shrout_change"] = np.nan

    # Update seen_permno AFTER using it
    seen_permno.update(chunk.loc[first_row_each_permno_in_chunk, "PERMNO"].astype(int).tolist())

    # Update last_shrout for cross-chunk lags (use last row per PERMNO in the chunk)
    last_rows = chunk.groupby("PERMNO", sort=False).tail(1)
    for p, s in zip(last_rows["PERMNO"].to_numpy(), last_rows["SHROUT"].to_numpy()):
        last_shrout[int(p)] = s

    # Drop helpers
    chunk = chunk.drop(columns=["SHROUT_LAG", "d_shrout_raw"])

    # RENAME COLUMNS

    chunk = chunk.rename(columns={
        "date": "month",
        "next_date": "next_month",
        "CUSIP": "cusip8",
    })

    # Ensure date columns are still datetime in-memory (CSV will save as YYYY-MM-DD)
    chunk["month"] = pd.to_datetime(chunk["month"], errors="coerce")
    chunk["next_month"] = pd.to_datetime(chunk["next_month"], errors="coerce")

    # Reorder columns in output
    desired_order = [
        "PERMNO", "month", "next_month", "TICKER", "COMNAM", "PERMCO", "cusip8",
        "SHROUT", "d_shrout", "ln_shrout_change"
    ]
    chunk = chunk[desired_order]

    # Write output
    chunk.to_csv(out_path, mode="a", index=False, header=not header_written)
    header_written = True

print("\nDone.")
print(f"Saved cleaned dataset to:\n{out_path}")
