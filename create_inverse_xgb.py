import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# Load XGBoost model instead of Random Forest
model = joblib.load('forward_model_xgb.joblib')

# Load training data to find seed points
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')
combined = pd.concat([train_df, labels_df], axis=1)

features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
bounds = [
    (1.3e-05, 0.3497), (0.0501, 0.8498), (1.62, 9.81), (0.4005, 1.5998),
    (0.8103, 3.7999), (0.7500, 1.3499), (2.6015, 4.5965), (0.5240, 1.3083)
]

# Find seed points from training data
valid_mask = (combined['P80'] >= 96) & (combined['P80'] <= 101) & (combined['R95'] <= 175)
seed_points = combined[valid_mask].sort_values(['energy', 'R95']).head(20)

def constrained_objective(x):
    x_df = pd.DataFrame([x], columns=features)
    pred = model.predict(x_df)[0]
    p80, r95 = pred[0], pred[3]
    penalty = 0
    if p80 < 96: penalty += (96 - p80) * 10000
    if p80 > 101: penalty += (p80 - 101) * 10000
    if r95 > 175: penalty += (r95 - 175) * 10000
    return x[6] + penalty

optimized = []
print("Optimizing 20 scenarios using XGBoost surrogate...")
for _, row in seed_points.iterrows():
    seed = row[features].values
    res = minimize(constrained_objective, seed, bounds=bounds, method='Nelder-Mead', options={'maxiter': 500})
    optimized.append(res.x if res.fun < 1000 else seed)

# Build submission
template = pd.read_csv('inverse_design/design_submission_template.csv')
sub_df = pd.DataFrame(optimized, columns=features)
sub_df.insert(0, 'submission_id', range(1, 21))
sub_df = sub_df[template.columns]
sub_df.to_csv('inverse_design/submission_xgb.csv', index=False)

# Verify and print
print(f"\n{'ID':<4} | {'P80 (96-101)':<15} | {'R95 (<=175)':<15} | {'Energy':<12} | Status")
print("-" * 65)
all_pass = True
for _, row in sub_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    preds = model.predict(pd.DataFrame(inputs, columns=features))[0]
    p80, r95 = preds[0], preds[3]
    ok = 96 <= p80 <= 101 and r95 <= 175
    if not ok: all_pass = False
    print(f"{int(row['submission_id']):<4} | {p80:<15.4f} | {r95:<15.4f} | {row['energy']:<12.4f} | {'✅' if ok else '❌'}")

print("-" * 65)
print("ALL PASS ✅" if all_pass else "SOME FAILED ❌")

# Now compare with RF submission
print("\n\n" + "=" * 70)
print("COMPARISON: Random Forest vs XGBoost Inverse Design")
print("=" * 70)

rf_model = joblib.load('forward_model.joblib')
rf_model.n_jobs = 1
rf_sub = pd.read_csv('inverse_design/submission.csv')
xgb_sub = sub_df

rf_energies = rf_sub['energy'].values
xgb_energies = xgb_sub['energy'].values

print(f"\n{'Metric':<30} | {'Random Forest':<18} | {'XGBoost':<18}")
print("-" * 70)
print(f"{'Avg Energy':<30} | {rf_energies.mean():<18.4f} | {xgb_energies.mean():<18.4f}")
print(f"{'Min Energy':<30} | {rf_energies.min():<18.4f} | {xgb_energies.min():<18.4f}")
print(f"{'Max Energy':<30} | {rf_energies.max():<18.4f} | {xgb_energies.max():<18.4f}")

# Verify RF rows through RF model
rf_pass = 0
for _, row in rf_sub.iterrows():
    preds = rf_model.predict(pd.DataFrame([row[features].values], columns=features))[0]
    if 96 <= preds[0] <= 101 and preds[3] <= 175: rf_pass += 1

# Verify XGB rows through XGB model
xgb_pass = 0
for _, row in xgb_sub.iterrows():
    preds = model.predict(pd.DataFrame([row[features].values], columns=features))[0]
    if 96 <= preds[0] <= 101 and preds[3] <= 175: xgb_pass += 1

print(f"{'Rows Passing Constraints':<30} | {rf_pass}/20{'':>12} | {xgb_pass}/20")
print(f"{'Winner':<30} | {'🏆' if rf_energies.mean() < xgb_energies.mean() else '':<18} | {'🏆' if xgb_energies.mean() < rf_energies.mean() else ''}")
print("=" * 70)
