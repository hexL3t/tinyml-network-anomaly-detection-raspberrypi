#!/usr/bin/env python3
"""
run_inference.py

Performs batch or single-row inference on an evaluation dataset CSV
using the trained Edge Impulse UNSW-NB15 model.
"""

import argparse
import csv
import os
from edge_impulse_linux.runner import ImpulseRunner

MODEL_PATH = '/home/pi/unsw-nb15-network-anomaly-detection-linux-armv7-v3-impulse-eon-9e6.eim'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Path to evaluation CSV dataset')
    args = parser.parse_args()

    print("=" * 50)
    print("UNSW-NB15 Batch Dataset Inference")
    print(f"Dataset: {args.data}")
    print("=" * 50)

    runner = ImpulseRunner(MODEL_PATH)
    model_info = runner.init()
    labels = model_info['model_parameters']['labels']
    print(f"Model Labels: {labels}\n")

    total = 0
    correct = 0

    with open(args.data, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Expecting features: dur, proto, sbytes, dbytes, state, service
            try:
                features = [
                    float(row['dur']),
                    float(row['proto']),
                    float(row['sbytes']),
                    float(row['dbytes']),
                    float(row['state']),
                    float(row['service'])
                ]
            except KeyError as e:
                print(f"Error: Missing expected column {e} in CSV row {i+1}")
                continue

            result = runner.classify(features)
            predicted = result.get('result', {}).get('classification', {})
            top_label = max(predicted, key=predicted.get) if predicted else 'unknown'

            total += 1
            print(f"Row {i+1}: Predicted => {top_label} ({predicted})")

    print(f"\nFinished processing {total} rows.")
    runner.stop()

if __name__ == '__main__':
    main()