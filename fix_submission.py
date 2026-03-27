import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('forward_model.joblib')
sub_df = pd.read_csv('inverse_design/submission.csv')

features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
bounds = {
    'porosity': (1.3e-05, 0.3497), 'atmosphere': (0.0501, 0.8498),
    'gravity': (1.62, 9.81), 'coupling': (0.4005, 1.5998),
    'strength': (0.8103, 3.7999), 'shape_factor': (0.7500, 1.3499),
    'energy': (2.6015, 4.5965), 'angle_rad': (0.5240, 1.3083)
}

# Find valid and invalid rows
valid_rows = []
invalid_indices = []

for idx, row in sub_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    preds = model.predict(pd.DataFrame(inputs, columns=features))[0]
    if 96 <= preds[0] <= 101 and preds[3] <= 175:
        valid_rows.append(row)
    else:
        invalid_indices.append(idx)

print(f"Found {len(valid_rows)} valid rows and {len(invalid_indices)} invalid rows.")

if len(invalid_indices) > 0:
    print("Sweeping 500,000 random configurations to find ultra-low-energy replacements...")
    np.random.seed(42)
    n_samples = 500000
    random_data = {}
    for f in features:
        random_data[f] = np.random.uniform(bounds[f][0], bounds[f][1], n_samples)
    
    # Bias energy towards the lower bounds
    random_data['energy'] = np.random.uniform(bounds['energy'][0], bounds['energy'][0] + 0.3, n_samples)
    
    random_df = pd.DataFrame(random_data)
    preds = model.predict(random_df)
    
    # Filter valid
    p80 = preds[:, 0]
    r95 = preds[:, 3]
    valid_mask = (p80 >= 96) & (p80 <= 101) & (r95 <= 175)
    
    valid_candidates = random_df[valid_mask]
    print(f"Found {len(valid_candidates)} strictly valid configurations from sweep.")
    
    # Sort by energy to get the best ones!
    best_candidates = valid_candidates.sort_values('energy').head(len(invalid_indices))
    
    # Replace in dataframe
    for i, idx in enumerate(invalid_indices):
        for f in features:
            sub_df.loc[idx, f] = best_candidates.iloc[i][f]

    sub_df.to_csv('inverse_design/submission.csv', index=False)
    print("Fixed submission saved!")
