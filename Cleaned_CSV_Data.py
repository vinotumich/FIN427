import os
import numpy as np
import pandas as pd

# =========================
# CONFIG (your path + file)
# =========================
BASE_DIR = r"C:\Users\rsila\OneDrive\Desktop\UMich\FIN 427\FIN427"
IN_FILE = "CSV_Dataset.csv"
OUT_FILE = "CSV_Dataset_CLEANED.csv"

in_path = os.path.join(BASE_DIR, IN_FILE)
out_path = os.path.join(BASE_DIR, OUT_FILE)

# =========================
# SAFE ON MEMORY SETTINGS
# =========================
CHUNKSIZE = 300_000  # conservative; lower if needed (e.g., 150_000)

dtype = {
    "PERMNO": "int32",
    "TICKER": "string",
    "COMNAM": "string",
    "PERMCO": "int32",
    "CUSIP": "string",
    "NWPERM": "float64",
    "SHROUT": "float64",
}

usecols = ["PERMNO", "date", "TICKER", "COMNAM", "PERMCO", "CUSIP", "NWPERM", "SHROUT"]

# Remove prior output if it exists (prevents accidental appends)
if os.path.exists(out_path):
    os.remove(out_path)

# Carryover: last SHROUT per PERMNO across chunk boundaries
last_shrout = {}

header_written = False

for chunk in pd.read_csv(
    in_path,
    usecols=usecols,
    dtype=dtype,
    parse_dates=["date"],
    na_values=["", "NA", "NaN"],
    chunksize=CHUNKSIZE,
):
    # ------------------------------------------------------------
    # 1) Drop the placeholder lines with an empty ticker
    # (These show up at the "start marker" for each PERMNO block)
    # ------------------------------------------------------------
    t = chunk["TICKER"]
    chunk = chunk[t.notna() & (t.str.len() > 0)].copy()

    # Ensure SHROUT is numeric
    chunk["SHROUT"] = pd.to_numeric(chunk["SHROUT"], errors="coerce")

    # ------------------------------------------------------------
    # 2) Compute lag(SHROUT) within chunk
    # ------------------------------------------------------------
    chunk["SHROUT_LAG"] = chunk.groupby("PERMNO", sort=False)["SHROUT"].shift(1)

    # ------------------------------------------------------------
    # 3) Patch the first row per PERMNO in this chunk using carryover
    # ------------------------------------------------------------
    first_mask = ~chunk["PERMNO"].duplicated()
    first_idx = chunk.index[first_mask]

    first_permnos = chunk.loc[first_idx, "PERMNO"].to_numpy()
    carry_lags = np.array([last_shrout.get(int(p), np.nan) for p in first_permnos], dtype="float64")

    current_lags = chunk.loc[first_idx, "SHROUT_LAG"].to_numpy(dtype="float64")
    patched_lags = np.where(np.isnan(current_lags), carry_lags, current_lags)

    chunk.loc[first_idx, "SHROUT_LAG"] = patched_lags

    # ------------------------------------------------------------
    # 4) New columns: absolute change + log change (log growth)
    # ------------------------------------------------------------
    chunk["d_shrout"] = chunk["SHROUT"] - chunk["SHROUT_LAG"]

    valid = (chunk["SHROUT"] > 0) & (chunk["SHROUT_LAG"] > 0)
    chunk["ln_shrout_change"] = np.where(
        valid,
        np.log(chunk["SHROUT"] / chunk["SHROUT_LAG"]),
        np.nan,
    )

    # ------------------------------------------------------------
    # 5) Update carryover dict with last SHROUT per PERMNO in chunk
    # ------------------------------------------------------------
    last_rows = chunk.groupby("PERMNO", sort=False).tail(1)
    for p, s in zip(last_rows["PERMNO"].to_numpy(), last_rows["SHROUT"].to_numpy()):
        last_shrout[int(p)] = s

    # Drop helper
    chunk = chunk.drop(columns=["SHROUT_LAG"])

    # ------------------------------------------------------------
    # 6) Write chunk to disk
    # ------------------------------------------------------------
    chunk.to_csv(out_path, mode="a", index=False, header=not header_written)
    header_written = True

print(f"Done. Cleaned file saved to:\n{out_path}")
