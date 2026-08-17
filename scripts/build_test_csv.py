#!/usr/bin/env python3
"""
build_test_csv.py

Converts Edge Impulse's exported "testing/" JSON files (from the
"Original files (uploader compatible)" export, Test tab selected)
into a single CSV of the 121 held-out test records.

Reads info.labels (Edge Impulse's manifest of every sample: path,
name, category, label) to identify exactly which files are the
genuine "testing" category records, then pulls the 6 feature values
out of each corresponding JSON file.

Usage:
    python3 build_test_csv.py <export_folder> <output_csv>

Example:
    python3 build_test_csv.py \
        unsw-nb15-network-anomaly-detection-export \
        unsw_nb15_test_holdout_121.csv
"""

import json
import csv
import sys
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 build_test_csv.py <export_folder> <output_csv>")
        sys.exit(1)

    export_folder = sys.argv[1]
    output_csv = sys.argv[2]

    labels_path = os.path.join(export_folder, "info.labels")
    if not os.path.isfile(labels_path):
        print(f"ERROR: could not find info.labels at {labels_path}")
        sys.exit(1)

    with open(labels_path, "r") as f:
        manifest = json.load(f)

    files = manifest.get("files", [])
    test_entries = [f for f in files if f.get("category") == "testing"]

    print(f"Manifest lists {len(files)} total files, "
          f"{len(test_entries)} tagged 'testing'.")

    rows = []
    sensor_names = None
    skipped = 0

    for entry in test_entries:
        rel_path = entry["path"]
        full_path = os.path.join(export_folder, rel_path)

        if not os.path.isfile(full_path):
            print(f"  WARNING: file listed in manifest but not found on disk: {full_path}")
            skipped += 1
            continue

        with open(full_path, "r") as jf:
            data = json.load(jf)

        payload = data.get("payload", {})
        sensors = payload.get("sensors", [])
        values = payload.get("values", [])

        if not values or not values[0]:
            print(f"  WARNING: no values found in {rel_path}, skipping.")
            skipped += 1
            continue

        current_sensor_names = [s["name"] for s in sensors]
        if sensor_names is None:
            sensor_names = current_sensor_names
        elif sensor_names != current_sensor_names:
            print(f"  WARNING: sensor order mismatch in {rel_path} "
                  f"({current_sensor_names} vs expected {sensor_names})")

        feature_row = values[0]  # single sample per file, non-time-series
        label = entry["label"]["label"]
        sample_name = entry["name"]

        rows.append({
            "sample_name": sample_name,
            **dict(zip(current_sensor_names, feature_row)),
            "label": label,
        })

    if not rows:
        print("ERROR: no rows extracted, nothing to write.")
        sys.exit(1)

    fieldnames = ["sample_name"] + sensor_names + ["label"]

    with open(output_csv, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {output_csv}")
    if skipped:
        print(f"({skipped} entries skipped due to missing files or bad data — see warnings above)")

    dos_count = sum(1 for r in rows if r["label"] == "DoS")
    normal_count = sum(1 for r in rows if r["label"] == "Normal")
    print(f"Label breakdown: DoS={dos_count}, Normal={normal_count}")


if __name__ == "__main__":
    main()