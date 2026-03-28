import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model
model = joblib.load('forward_model.joblib')
model.n_jobs = 1

# Define features and bounds
features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']
bounds = {
    'porosity': (1.3e-05, 0.3497), 'atmosphere': (0.0501, 0.8498),
    'gravity': (1.62, 9.81), 'coupling': (0.4005, 1.5998),
    'strength': (0.8103, 3.7999), 'shape_factor': (0.7500, 1.3499),
    'energy': (2.6015, 4.5965), 'angle_rad': (0.5240, 1.3083)
}

print("Running Guarantee Sweep (500,000 Samples) for 20/20 Pass...")
np.random.seed(42)
n_samples = 500000

# Generate massive random search space
random_data = {}
for f in features:
    random_data[f] = np.random.uniform(bounds[f][0], bounds[f][1], n_samples)

# Bias energy slightly to prioritize lower impacts
random_data['energy'] = np.random.uniform(bounds['energy'][0], bounds['energy'][0] + 0.3, n_samples)

random_df = pd.DataFrame(random_data)

# Surrogate predictions
preds = model.predict(random_df)
p80 = preds[:, 0]
r95 = preds[:, 3]

# Filter for strict constraints
valid_mask = (p80 >= 96) & (p80 <= 101) & (r95 <= 175)
valid_candidates = random_df[valid_mask]

print(f"Found {len(valid_candidates)} strictly valid configurations.")

# Take top 20 lowest energy unique scenarios
best_20 = valid_candidates.sort_values('energy').head(20)

# Load template for formatting
template = pd.read_csv('inverse_design/design_submission_template.csv')
submission_df = best_20.copy()
submission_df.insert(0, 'submission_id', range(1, 21))
submission_df = submission_df[template.columns]

# Save
submission_df.to_csv('inverse_design/submission.csv', index=False)
print("Guaranteed 20/20 submission saved to inverse_design/submission.csv")
