"""
augment_dos_samples.py

Purpose
-------
Generates additional synthetic DoS samples targeting the specific failure
pattern found during model diagnosis: DoS traffic with dbytes > 0 (a partial
response present), which the classifier tends to misread as Normal.

This is NOT blanket oversampling (the classes are already 50/50). It is
targeted augmentation of the exact sub-pattern the model struggles with,
so the classifier sees more examples of "this looks like it has a response,
but it's still DoS" during training.

How it works
------------
1. Loads your existing filtered CSV.
2. Finds real DoS rows where dbytes > 0 (the hard pattern).
3. For each one, generates N synthetic variants by jittering the numeric
   columns (dur, sbytes, dbytes) with small multiplicative noise, while
   keeping the categorical columns (proto, state, service) identical to
   a real row — this avoids inventing unrealistic protocol/state
   combinations that don't exist in real traffic.
4. Appends the synthetic rows to the original dataset and writes a new CSV.

Usage
-----
    python3 augment_dos_samples.py unsw_nb15_filtered.csv unsw_nb15_augmented.csv --n-per-row 3

Then re-upload unsw_nb15_augmented.csv to Edge Impulse via the CSV Wizard,
using the same label column as your original upload.
"""

import argparse
import csv
import random


NUMERIC_COLS = ["dur", "sbytes", "dbytes"]
LABEL_COL = "attack_cat"  # adjust if your CSV uses a different column name
DOS_LABEL = "DoS"
JITTER_PCT = 0.28  # +/- 28% random variation on numeric columns (widened after
                   # iteration 6 showed +/-15% with 3x repeats produced near-duplicate
                   # synthetic rows tight enough to destabilize training)


def jitter(value: float, pct: float) -> float:
    """Apply random multiplicative jitter, floored at zero."""
    factor = 1.0 + random.uniform(-pct, pct)
    return max(0.0, value * factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to existing filtered CSV")
    parser.add_argument("output_csv", help="Path to write augmented CSV")
    parser.add_argument("--n-per-row", type=int, default=1,
                         help="Synthetic variants to generate per hard DoS row (default 1 - "
                              "iteration 6 found n=3 at +/-15%% jitter produced tightly-clustered "
                              "near-duplicates that destabilized training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if LABEL_COL not in fieldnames:
        raise SystemExit(
            f"Column '{LABEL_COL}' not found. Available columns: {fieldnames}\n"
            f"Edit LABEL_COL at the top of this script to match your CSV."
        )

    hard_rows = [
        r for r in rows
        if r[LABEL_COL].strip() == DOS_LABEL and float(r.get("dbytes", 0) or 0) > 0
    ]

    print(f"Loaded {len(rows)} total rows")
    print(f"Found {len(hard_rows)} hard-pattern DoS rows (label=DoS, dbytes>0)")

    if not hard_rows:
        raise SystemExit("No matching hard-pattern rows found — check LABEL_COL and DOS_LABEL values.")

    synthetic_rows = []
    next_timestamp = max(int(float(r["timestamp"])) for r in rows if r.get("timestamp") not in (None, "")) + 1
    for base_row in hard_rows:
        for _ in range(args.n_per_row):
            new_row = dict(base_row)
            new_row["timestamp"] = next_timestamp
            next_timestamp += 1
            for col in NUMERIC_COLS:
                if col in new_row and new_row[col] not in ("", None):
                    new_row[col] = round(jitter(float(new_row[col]), JITTER_PCT), 6)
            synthetic_rows.append(new_row)

    all_rows = rows + synthetic_rows

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Generated {len(synthetic_rows)} synthetic DoS rows")
    print(f"Wrote {len(all_rows)} total rows to {args.output_csv}")
    print()
    print("Next steps:")
    print("1. Re-upload the output CSV to Edge Impulse via Data acquisition > CSV Wizard")
    print("2. Retrain manually (not EON Tuner) using the same architecture as Impulse #1")
    print("3. Check the confusion matrix specifically for DoS recall vs Impulse #1's baseline")


if __name__ == "__main__":
    main()