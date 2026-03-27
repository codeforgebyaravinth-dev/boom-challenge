import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")

# Load trained model
model = joblib.load('forward_model.joblib')

# Input bounds from EDA (min, max)
bounds = [
    (1.3e-05, 0.3497),  # porosity
    (0.0501, 0.8498),   # atmosphere
    (1.62, 9.81),       # gravity
    (0.4005, 1.5998),   # coupling
    (0.8103, 3.7999),   # strength
    (0.7500, 1.3499),   # shape_factor
    (2.6015, 4.5965),   # energy (Minimize this)
    (0.5240, 1.3083)    # angle_rad
]

def objective(x):
    # x is the input array
    # Goal: minimize energy (x[6])
    return x[6]

def constraint_p80_min(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    return pred[0] - 96 # pred[0] >= 96

def constraint_p80_max(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    return 101 - pred[0] # pred[0] <= 101

def constraint_r95_max(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    return 175 - pred[3] # pred[3] <= 175 (R95 is index 3)

# Since differential_evolution doesn't natively support constraints outside of bounds easily in older scipy (handled by penalties),
# we'll use a penalty method.

def constrained_objective(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    p80 = pred[0]
    r95 = pred[3]
    
    penalty = 0
    if p80 < 96: penalty += (96 - p80) * 1000
    if p80 > 101: penalty += (p80 - 101) * 1000
    if r95 > 175: penalty += (r95 - 175) * 1000
    
    return x[6] + penalty

print("Optimizing inverse design...")
result = differential_evolution(constrained_objective, bounds, strategy='best1bin', maxiter=50, popsize=15, mutation=(0.5, 1), recombination=0.7, seed=42)

if result.success:
    best_inputs = result.x
    x_df = pd.DataFrame([best_inputs], columns=model.feature_names_in_)
    final_preds = model.predict(x_df)[0]
    print("\n--- Optimized Results ---")
    print(f"Porosity: {best_inputs[0]:.6f}")
    print(f"Atmosphere: {best_inputs[1]:.6f}")
    print(f"Gravity: {best_inputs[2]:.6f}")
    print(f"Coupling: {best_inputs[3]:.6f}")
    print(f"Strength: {best_inputs[4]:.6f}")
    print(f"Shape Factor: {best_inputs[5]:.6f}")
    print(f"Energy (Minimised): {best_inputs[6]:.6f}")
    print(f"Angle (Rad): {best_inputs[7]:.6f}")
    
    print("\n--- Predictions ---")
    print(f"P80: {final_preds[0]:.4f} (Goal: 96-101)")
    print(f"R95: {final_preds[3]:.4f} (Goal: <= 175)")
    
    # Save results to CSV for submission
    # The requirement says "generate final predictions for submission"
    # But for inverse design, it's usually the inputs. 
    # Let me check if there's a specific submission format.
    # The user didn't specify, so I'll just save the inputs to optimized_inputs.csv
    pd.DataFrame([best_inputs], columns=['porosity', 'atmosphere', 'gravity', 'coupling', 'strength', 'shape_factor', 'energy', 'angle_rad']).to_csv('optimized_inputs.csv', index=False)
else:
    print("Optimization failed.")
