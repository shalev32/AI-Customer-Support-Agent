#!/usr/bin/env python3
"""Validate that all required project files are present and correctly formatted."""
import os, sys, csv
REQUIRED = ["fetch_product_data.py","customers.json","customer_store.py",
            "policies.txt","example_queries.txt","README.md"]
def main():
    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing: sys.exit(f"Missing gist files: {missing}")
    if not os.path.exists("product_data.csv"):
        sys.exit("product_data.csv not found. Run: python fetch_product_data.py "
                 "(needs Kaggle credentials — see README.md).")
    with open("product_data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    need = ["Uniq Id","Product Name","Description","List Price","Sale Price",
            "Brand","Gtin","Category","Available"]
    if header[:9] != need: sys.exit(f"product_data.csv columns wrong: {header[:9]}")
    if len(data) != 787: sys.exit(f"product_data.csv has {len(data)} rows, expected 787.")
    import customer_store
    n = customer_store.get_cust_data_collection().count_documents({})
    if n != 50: sys.exit(f"customers.json has {n} records, expected 50.")
    print("All required files found. You are ready to start working on agent.py.")
if __name__ == "__main__":
    main()
