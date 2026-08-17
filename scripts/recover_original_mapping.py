"""
recover_original_mapping.py

Recovers the proto/state/service string->integer mapping that
filter_dataset.py used, WITHOUT retraining the model or changing the
existing unsw_nb15_filtered.csv.

Why this works: pd.factorize() is deterministic given the same input
row order. Since filter_dataset.py's original steps (load -> filter to
Normal/DoS -> select 6 features -> factorize) are fully reproducible
from the same source CSV, re-running those exact steps and capturing
factorize()'s second return value (the category list it discarded)
recovers the exact mapping that was actually used to produce your
current unsw_nb15_filtered.csv and the model trained on it.

This is the SAFE option: it does not change any existing data,
results, or the trained Impulse-EON-9e6 model. It only recovers
documentation of a mapping that already implicitly exists.

Usage:
    python3 recover_original_mapping.py
"""

import pandas as pd
import json
import os

SOURCE_CSV = os.path.expanduser('~/Documents/University/MLP301/dataset/UNSW_NB15_training-set.csv')
MAPPING_JSON = os.path.expanduser('~/Documents/University/MLP301/dataset/category_mappings.json')

# --- Reproduce filter_dataset.py's exact steps, in the exact same order ---
df = pd.read_csv(SOURCE_CSV)
df_filtered = df[df['attack_cat'].isin(['Normal', 'DoS'])]

features = ['dur', 'proto', 'sbytes', 'dbytes', 'state', 'service', 'attack_cat']
df_filtered = df_filtered[features].copy()

# factorize() returns (codes, uniques) - uniques[i] is the original
# string that was mapped to integer i. The original script only kept
# codes (df_filtered['proto'] = ...[0]) and threw away uniques ([1]).
# We capture both here.
proto_codes, proto_uniques = pd.factorize(df_filtered['proto'])
state_codes, state_uniques = pd.factorize(df_filtered['state'])
service_codes, service_uniques = pd.factorize(df_filtered['service'])

proto_map = {str(v): int(i) for i, v in enumerate(proto_uniques)}
state_map = {str(v): int(i) for i, v in enumerate(state_uniques)}
service_map = {str(v): int(i) for i, v in enumerate(service_uniques)}

with open(MAPPING_JSON, 'w') as f:
    json.dump({
        'proto': proto_map,
        'state': state_map,
        'service': service_map,
    }, f, indent=2)

print("Recovered category mappings (matches your EXISTING trained model - no retraining needed):")
print()
print(f"proto ({len(proto_map)} categories):")
for k, v in sorted(proto_map.items(), key=lambda x: x[1]):
    print(f"  {k!r:20s} -> {v}")
print()
print(f"state ({len(state_map)} categories):")
for k, v in sorted(state_map.items(), key=lambda x: x[1]):
    print(f"  {k!r:20s} -> {v}")
print()
print(f"service ({len(service_map)} categories):")
for k, v in sorted(service_map.items(), key=lambda x: x[1]):
    print(f"  {k!r:20s} -> {v}")
print()
print(f"Saved to: {MAPPING_JSON}")
print()
print("IMPORTANT: this recovery only works if UNSW_NB15_training-set.csv on")
print("disk is byte-identical to what filter_dataset.py originally read")
print("(same file, unmodified, unsorted). If the recovered mapping's")
print("category COUNTS below don't match what you'd expect from the")
print("UNSW-NB15 documentation (proto should have ~130+ categories,")
print("state ~10-16, service ~13), the source file may have changed -")
print("flag this before trusting the recovered mapping.")