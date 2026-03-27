import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('forward_model.joblib')
model.n_jobs = 1 # Fixes the Windows Parallel warning and speeds up single-row predictions
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')
combined = pd.concat([train_df, labels_df], axis=1)

# Target 50 scenarios
TARGET_COUNT = 50

valid_mask = (combined['P80'] >= 96) & (combined['P80'] <= 101) & (combined['R95'] <= 175)
valid_points = combined[valid_mask].sort_values(['energy', 'R95']) 

if len(valid_points) >= TARGET_COUNT:
    seed_points = valid_points.head(TARGET_COUNT)
else:
    near_mask = (combined['P80'] >= 85) & (combined['P80'] <= 115) & (combined['R95'] <= 250)
    near_points = combined[near_mask & ~valid_mask].sort_values(['energy', 'R95'])
    seed_points = pd.concat([valid_points, near_points]).head(TARGET_COUNT)

bounds = [
    (1.3e-05, 0.3497), (0.0501, 0.8498), (1.62, 9.81), (0.4005, 1.5998),
    (0.8103, 3.7999), (0.7500, 1.3499), (2.6015, 4.5965), (0.5240, 1.3083)
]

def constrained_objective(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    p80, r95 = pred[0], pred[3]
    penalty = 0
    if p80 < 96: penalty += (96 - p80) * 10000
    if p80 > 101: penalty += (p80 - 101) * 10000
    if r95 > 175: penalty += (r95 - 175) * 10000
    return x[6] + penalty

optimized_scenarios = []

print(f"Optimizing {TARGET_COUNT} scenarios...")
features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
for idx, (_, row) in enumerate(seed_points.iterrows()):
    seed_inputs = row[features].values
    res = minimize(constrained_objective, seed_inputs, bounds=bounds, method='Nelder-Mead', options={'maxiter': 500})
    best_inputs = res.x if res.success or res.status == 0 else seed_inputs
    optimized_scenarios.append(best_inputs)

cols = features
submission_df = pd.DataFrame(optimized_scenarios, columns=cols)

template = pd.read_csv('inverse_design/design_submission_template.csv')
submission_df['submission_id'] = range(1, TARGET_COUNT + 1)
submission_df = submission_df[template.columns]

# Validate and log output
invalid_count = 0
for idx, row in submission_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    preds = model.predict(pd.DataFrame(inputs, columns=features))[0]
    if not (96 <= preds[0] <= 101 and preds[3] <= 175):
        invalid_count += 1

print(f"Initial optimization complete. Found {invalid_count} invalid rows that need fixing out of 50.")

submission_df.to_csv('inverse_design/submission_50.csv', index=False)
print("Saved to inverse_design/submission_50.csv")
