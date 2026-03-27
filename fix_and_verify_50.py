import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('forward_model.joblib')
model.n_jobs = 1
sub_df = pd.read_csv('inverse_design/submission_50.csv')

features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
bounds = {
    'porosity': (1.3e-05, 0.3497), 'atmosphere': (0.0501, 0.8498),
    'gravity': (1.62, 9.81), 'coupling': (0.4005, 1.5998),
    'strength': (0.8103, 3.7999), 'shape_factor': (0.7500, 1.3499),
    'energy': (2.6015, 4.5965), 'angle_rad': (0.5240, 1.3083)
}

invalid_indices = []

for idx, row in sub_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    preds = model.predict(pd.DataFrame(inputs, columns=features))[0]
    if not (96 <= preds[0] <= 101 and preds[3] <= 175):
        invalid_indices.append(idx)

if len(invalid_indices) > 0:
    print(f"Sweeping 500,000 random configurations to fix {len(invalid_indices)} rows...")
    np.random.seed(45)
    n_samples = 500000
    random_data = {}
    for f in features:
        random_data[f] = np.random.uniform(bounds[f][0], bounds[f][1], n_samples)
    
    random_data['energy'] = np.random.uniform(bounds['energy'][0], bounds['energy'][0] + 0.3, n_samples)
    
    random_df = pd.DataFrame(random_data)
    preds = model.predict(random_df)
    
    p80 = preds[:, 0]
    r95 = preds[:, 3]
    valid_mask = (p80 >= 96) & (p80 <= 101) & (r95 <= 175)
    
    valid_candidates = random_df[valid_mask]
    best_candidates = valid_candidates.sort_values('energy').head(len(invalid_indices))
    
    for i, idx in enumerate(invalid_indices):
        for f in features:
            sub_df.loc[idx, f] = best_candidates.iloc[i][f]

    sub_df.to_csv('inverse_design/submission_50.csv', index=False)
    print("Fixed 50-row submission successfully!")

# VERIFY
print(f"\n--- VERIFYING {len(sub_df)} IMPACT SCENARIOS ---")
all_pass = True
for idx, row in sub_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    preds = model.predict(pd.DataFrame(inputs, columns=features))[0]
    if not (96 <= preds[0] <= 101 and preds[3] <= 175):
        all_pass = False
        print(f"Row {idx+1} FAIL: P80={preds[0]:.2f}, R95={preds[3]:.2f}")

if all_pass:
    print(f"SUCCESS: All {len(sub_df)} scenarios strictly pass the physics constraints!")
else:
    print("WARNING: Some rows failed.")
