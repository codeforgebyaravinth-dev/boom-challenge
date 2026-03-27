import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# Load trained model
model = joblib.load('forward_model.joblib')

# Seed with Best Point from Training
# porosity          0.340316
# atmosphere        0.801367
# gravity           9.810000
# coupling          1.218020
# strength          1.387266
# shape_factor      1.100515
# energy            2.724696
# angle_rad         0.701302
seed_inputs = np.array([0.340316, 0.801367, 9.810000, 1.218020, 1.387266, 1.100515, 2.724696, 0.701302])

# Bounds
bounds = [
    (1.3e-05, 0.3497),  # porosity
    (0.0501, 0.8498),   # atmosphere
    (1.62, 9.81),       # gravity
    (0.4005, 1.5998),   # coupling
    (0.8103, 3.7999),   # strength
    (0.7500, 1.3499),   # shape_factor
    (2.6015, 4.5965),   # energy
    (0.5240, 1.3083)    # angle_rad
]

def constrained_objective(x):
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    pred = model.predict(x_df)[0]
    p80 = pred[0]
    r95 = pred[3]
    
    penalty = 0
    if p80 < 96: penalty += (96 - p80) * 10000
    if p80 > 101: penalty += (p80 - 101) * 10000
    if r95 > 175: penalty += (r95 - 175) * 10000
    
    return x[6] + penalty # Objective is energy (index 6)

print("Refining optimization...")
# L-BFGS-B might struggle with RF steps, but COBYLA or Nelder-Mead could work well with derivatives-free. 
# RF is not differentiable. Using Nelder-Mead.
result = minimize(constrained_objective, seed_inputs, bounds=bounds, method='Nelder-Mead', options={'maxiter': 500})

if result.success or result.status == 0:
    best_inputs = result.x
    x_df = pd.DataFrame([best_inputs], columns=model.feature_names_in_)
    final_preds = model.predict(x_df)[0]
    print("\n--- Final Refined Results ---")
    print(f"Porosity: {best_inputs[0]:.6f}")
    print(f"Atmosphere: {best_inputs[1]:.6f}")
    print(f"Gravity: {best_inputs[2]:.6f}")
    print(f"Coupling: {best_inputs[3]:.6f}")
    print(f"Strength: {best_inputs[4]:.6f}")
    print(f"Shape Factor: {best_inputs[5]:.6f}")
    print(f"Energy (Optimized): {best_inputs[6]:.6f}")
    print(f"Angle (Rad): {best_inputs[7]:.6f}")
    
    print("\n--- Final Predictions ---")
    print(f"P80: {final_preds[0]:.4f} (Goal: 96-101)")
    print(f"R95: {final_preds[3]:.4f} (Goal: <= 175)")
    
    pd.DataFrame([best_inputs], columns=model.feature_names_in_).to_csv('optimized_inputs_final.csv', index=False)
else:
    print(f"Optimization finished with status {result.status}: {result.message}")
