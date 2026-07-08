#!/usr/bin/env python3
"""Fetch the real Walmart e-commerce dataset and shorten it to exactly 787 rows,
guaranteeing the products referenced by the sample queries survive.
Output: product_data.csv (the 9 columns from the assignment spec).

NO Kaggle API key required. `kagglehub` downloads this public dataset anonymously.
If Kaggle ever rate-limits anonymous downloads, create a free Kaggle account,
Account -> Create New API Token (downloads kaggle.json -> ~/.kaggle/kaggle.json,
chmod 600), then re-run. See README.md.
"""
import os, sys
import pandas as pd

TARGET_ROWS = 787
SLUG = "promptcloud/walmart-product-data-2019"
COLUMNS = ["Uniq Id", "Product Name", "Description", "List Price",
           "Sale Price", "Brand", "Gtin", "Category", "Available"]

# 1) EXACT named products/brands the 10 sample queries reference — always kept.
#    Substrings chosen to be specific (not accidental matches). All confirmed
#    present in the source on 2026-06-13 (Task 1).
MUST_KEEP_NAME = [
    "Littmann",                # q1 stethoscope
    "01-415EA",                # q2 thermometer #1 (item number, unique)
    "HEALTHMAX",               # q2 thermometer #2
    "Fermometer",              # q2 thermometer #3 (misspelling in source, unique)
    "Wrist Weight",            # q10 wrist weights
]
MUST_KEEP_BRAND = ["LHCER"]    # q8 comparable-brand (thermometers etc.)

# 2) Category coverage the queries exercise — guarantee a quota of each so RAG
#    and filters have real matches. (token-in-name, quota) — broad but bounded.
COVERAGE_QUOTAS = [
    ("scale", 40),      # q3 body-weight scales (incl. lb/kg ones)
    ("soap", 40),       # q7 hand/skin soaps
    ("snow", 30),       # q9 snow / winter / snowboarding gear
    ("weight", 40),     # q4 weight-loss / q10 weights
    ("bike", 25),       # general fitness spread
    ("thermometer", 20),# q2/q8 thermometers
    ("vitamin", 20),    # interests category spread
]

def _download_csv() -> str:
    """Return a local path to the source CSV via kagglehub (anonymous)."""
    import kagglehub
    path = kagglehub.dataset_download(SLUG)
    for root, _dirs, files in os.walk(path):  # CSV is nested under home/sdf/
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError("No CSV found in the kagglehub download.")

def main():
    src = _download_csv()
    df = pd.read_csv(src)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"Source CSV missing expected columns: {missing}")
    df = df[COLUMNS].copy()
    df = df.dropna(subset=["Product Name"]).drop_duplicates(subset=["Uniq Id"])
    df = df.reset_index(drop=True)
    name = df["Product Name"].astype(str)
    brand = df["Brand"].astype(str)

    keep_idx: list[int] = []
    def add(idx_iterable):
        for i in idx_iterable:
            if i not in keep_idx:
                keep_idx.append(i)

    # (1) exact named products — always in
    named = pd.Series(False, index=df.index)
    for tok in MUST_KEEP_NAME:
        named |= name.str.contains(tok, case=False, na=False, regex=False)
    for tok in MUST_KEEP_BRAND:
        named |= brand.str.contains(tok, case=False, na=False, regex=False)
    add(df.index[named].tolist())

    # (2) category quotas — deterministic head() of each token's matches
    for tok, quota in COVERAGE_QUOTAS:
        matches = df.index[name.str.contains(tok, case=False, na=False, regex=False)]
        add(matches[:quota].tolist())

    # (3) fill the remainder with a deterministic spread of the rest
    if len(keep_idx) < TARGET_ROWS:
        rest = [i for i in df.index if i not in set(keep_idx)]
        add(rest[: TARGET_ROWS - len(keep_idx)])

    guaranteed = int(named.sum())
    out = df.loc[keep_idx[:TARGET_ROWS]].reset_index(drop=True)
    out.to_csv("product_data.csv", index=False)
    print(f"Wrote product_data.csv with {len(out)} rows "
          f"({guaranteed} named-product rows guaranteed-kept).")

if __name__ == "__main__":
    main()
