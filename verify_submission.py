import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model and submission
model = joblib.load('forward_model.joblib')
sub_df = pd.read_csv('inverse_design/submission.csv')

print("--- VERIFYING 20 IMPACT SCENARIOS ---\n")
print(f"{'ID':<4} | {'P80 (96-101)':<15} | {'R95 (<=175)':<15} | {'Energy (Min)':<15} | {'Status'}")
print("-" * 70)

all_pass = True
features = ['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']

for _, row in sub_df.iterrows():
    inputs = row[features].values.reshape(1, -1)
    inputs_df = pd.DataFrame(inputs, columns=features)
    
    preds = model.predict(inputs_df)[0]
    p80 = preds[0]
    r95 = preds[3]
    energy = row['energy']
    id_val = int(row['submission_id'])
    
    # Check constraints
    p80_pass = 96 <= p80 <= 101
    r95_pass = r95 <= 175
    status = "✅ PASS" if (p80_pass and r95_pass) else "❌ FAIL"
    
    if not (p80_pass and r95_pass):
        all_pass = False
        
    print(f"{id_val:<4} | {p80:<15.4f} | {r95:<15.4f} | {energy:<15.4f} | {status}")

print("-" * 70)
if all_pass:
    print("SUCCESS: All 20 scenarios strictly satisfy the physical constraints!")
else:
    print("WARNING: Some scenarios failed the constraints.")
