"""
filter_dataset_v2.py

Fixed version of filter_dataset.py. The original script used
pd.factorize() to convert proto/state/service strings into integers,
but never saved WHICH string became WHICH number — meaning that
mapping only ever existed in memory during that one script run.

This is a critical gap for the live packet-capture pipeline (A3 gap #1):
without knowing the exact string->integer mapping used during training,
there is no way to correctly encode a live-captured packet's protocol
(e.g. "tcp") into the number the model expects (e.g. 3), since
factorize() assigns numbers based on first-appearance order, which is
not reproducible without saving it.

This version does two things differently:
1. Uses a FIXED, explicit mapping dict (not factorize) so the encoding
   is deterministic and known in advance, rather than depending on
   row order.
2. Saves that mapping to a JSON file so the live capture script can
   import and reuse the exact same encoding.

If retraining is not desired (i.e. you want to keep using the existing
trained model), see the note at the bottom on recovering the original
factorize() mapping instead of changing it.
"""

import pandas as pd
import json

SOURCE_CSV = r'C:\Users\Tia Darvell\Downloads\MLP301\dataset\UNSW_NB15_training-set.csv'
OUTPUT_CSV = r'C:\Users\Tia Darvell\Downloads\MLP301\dataset\unsw_nb15_filtered_v2.csv'
MAPPING_JSON = r'C:\Users\Tia Darvell\Downloads\MLP301\dataset\category_mappings.json'

df = pd.read_csv(SOURCE_CSV)

df_filtered = df[df['attack_cat'].isin(['Normal', 'DoS'])]

features = ['dur', 'proto', 'sbytes', 'dbytes', 'state', 'service', 'attack_cat']
df_filtered = df_filtered[features].copy()

# --- Fixed, explicit mappings (deterministic, independent of row order) ---
# Built from the full set of unique values seen in the UNSW-NB15 dataset.
# Sorted alphabetically so the mapping is reproducible and easy to
# regenerate if the dataset changes.

proto_categories = sorted(df_filtered['proto'].unique())
state_categories = sorted(df_filtered['state'].unique())
service_categories = sorted(df_filtered['service'].unique())

proto_map = {v: i for i, v in enumerate(proto_categories)}
state_map = {v: i for i, v in enumerate(state_categories)}
service_map = {v: i for i, v in enumerate(service_categories)}

df_filtered['proto'] = df_filtered['proto'].map(proto_map)
df_filtered['state'] = df_filtered['state'].map(state_map)
df_filtered['service'] = df_filtered['service'].map(service_map)

# Balance the classes - 400 of each
normal = df_filtered[df_filtered['attack_cat'] == 'Normal'].sample(400, random_state=42)
dos = df_filtered[df_filtered['attack_cat'] == 'DoS'].sample(400, random_state=42)

df_final = pd.concat([normal, dos]).sample(frac=1, random_state=42).reset_index(drop=True)
df_final.insert(0, 'timestamp', range(0, len(df_final)))

df_final.to_csv(OUTPUT_CSV, index=False)

with open(MAPPING_JSON, 'w') as f:
    json.dump({
        'proto': proto_map,
        'state': state_map,
        'service': service_map,
    }, f, indent=2)

print(f"Done. Total records: {len(df_final)}")
print(df_final['attack_cat'].value_counts())
print()
print(f"Category mappings saved to: {MAPPING_JSON}")
print(f"  proto:   {len(proto_map)} categories")
print(f"  state:   {len(state_map)} categories")
print(f"  service: {len(service_map)} categories")

# ---------------------------------------------------------------------
# NOTE: this produces a DIFFERENT encoding than the original
# filter_dataset.py (alphabetical vs first-appearance order), so if you
# use this version, the model needs to be RETRAINED on unsw_nb15_filtered_v2.csv
# — you cannot mix this encoding with the already-trained Impulse-EON-9e6 model.
#
# If you'd rather NOT retrain, the alternative is recovering the exact
# factorize() mapping the original script produced. That requires
# re-running the exact same pandas version/row order and capturing the
# factorize() output's second return value (the categories array),
# which the original script discarded. See recover_original_mapping.py
# for that approach if needed.
# ---------------------------------------------------------------------