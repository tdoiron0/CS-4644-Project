#!/usr/bin/env python3
"""
Stratified 70/15/15 train/val/test split of labels.csv,
stratified on variant so all hierarchy levels stay proportional.

Produces: train.csv, val.csv, test.csv
Removes the split column from the original labels.csv (keeps it as the master).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")
RANDOM_SEED = 42

COLUMNS = ["image_id", "manufacturer", "family", "variant"]


def stratified_split(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["variant"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["variant"],
    )

    return train_df, val_df, test_df


def main():
    print("=" * 50)
    print("Stratified split")
    print("=" * 50)

    df = pd.read_csv(LABELS_PATH, dtype={"image_id": str})
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    print(f"Loaded {len(df)} rows from labels.csv")

    train_df, val_df, test_df = stratified_split(df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(DATA_DIR, f"{name}.csv")
        split_df[COLUMNS].sort_values("image_id").to_csv(out_path, index=False)
        print(f"Written: {name}.csv ({len(split_df):,} rows)")

    df[COLUMNS].to_csv(LABELS_PATH, index=False)

    print(f"\nPer-variant distribution (first 5 variants):")
    combined = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    check = combined.groupby("variant")["split"].value_counts().unstack(fill_value=0)
    print(check.head().to_string(col_space=8))

    print(f"\nPer-manufacturer distribution:")
    check_mfr = combined.groupby("manufacturer")["split"].value_counts().unstack(fill_value=0)
    print(check_mfr.to_string(col_space=8))

    print("\nDone.")


if __name__ == "__main__":
    main()
