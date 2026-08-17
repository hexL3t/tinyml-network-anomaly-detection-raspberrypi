import pandas as pd

# Load the training set
df = pd.read_csv(r'C:\Users\Tia Darvell\Downloads\MLP301\dataset\UNSW_NB15_training-set.csv')

# Keep only Normal and DoS records
df_filtered = df[df['attack_cat'].isin(['Normal', 'DoS'])]

# Keep only the 6 selected features plus the label
features = ['dur', 'proto', 'sbytes', 'dbytes', 'state', 'service', 'attack_cat']
df_filtered = df_filtered[features].copy()

# Convert text columns to numeric using label encoding
df_filtered['proto'] = pd.factorize(df_filtered['proto'])[0]
df_filtered['state'] = pd.factorize(df_filtered['state'])[0]
df_filtered['service'] = pd.factorize(df_filtered['service'])[0]

# Balance the classes - 400 of each
normal = df_filtered[df_filtered['attack_cat'] == 'Normal'].sample(400, random_state=42)
dos = df_filtered[df_filtered['attack_cat'] == 'DoS'].sample(400, random_state=42)

# Combine and shuffle
df_final = pd.concat([normal, dos]).sample(frac=1, random_state=42).reset_index(drop=True)

# Add timestamp column required by Edge Impulse
df_final.insert(0, 'timestamp', range(0, len(df_final)))

# Save to new CSV
df_final.to_csv(r'C:\Users\Tia Darvell\Downloads\MLP301\dataset\unsw_nb15_filtered.csv', index=False)

print(f"Done. Total records: {len(df_final)}")
print(df_final['attack_cat'].value_counts())
print(df_final.head())