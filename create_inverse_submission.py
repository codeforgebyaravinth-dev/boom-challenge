import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('forward_model.joblib')
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')
combined = pd.concat([train_df, labels_df], axis=1)

# Find top valid points to use as seeds for optimization
valid_mask = (combined['P80'] >= 96) & (combined['P80'] <= 101) & (combined['R95'] <= 175)
valid_points = combined[valid_mask].sort_values(['energy', 'R95']) # Optimize for small-impact score

# We need 20 scenarios. If there are fewer than 20 valid in train, we add near-matches
if len(valid_points) >= 20:
    seed_points = valid_points.head(20)
else:
    near_mask = (combined['P80'] >= 90) & (combined['P80'] <= 110) & (combined['R95'] <= 200)
    near_points = combined[near_mask & ~valid_mask].sort_values(['energy', 'R95'])
    seed_points = pd.concat([valid_points, near_points]).head(20)

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

print("Optimizing 20 scenarios...")
features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
for idx, (_, row) in enumerate(seed_points.iterrows()):
    seed_inputs = row[features].values
    res = minimize(constrained_objective, seed_inputs, bounds=bounds, method='Nelder-Mead', options={'maxiter': 500})
    best_inputs = res.x if res.success or res.status == 0 else seed_inputs
    optimized_scenarios.append(best_inputs)

cols = features
submission_df = pd.DataFrame(optimized_scenarios, columns=cols)

# Reorder columns to match template: submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor
template = pd.read_csv('inverse_design/design_submission_template.csv')
submission_df['submission_id'] = range(1, 21)
submission_df = submission_df[template.columns]

submission_df.to_csv('inverse_design/submission.csv', index=False)
print("Inverse design submission saved to inverse_design/submission.csv with 20 scenarios.")
