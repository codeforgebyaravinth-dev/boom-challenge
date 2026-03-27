import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')

# Split data exactly as trained to evaluate on the hold-out test set
X_train, X_test, y_train, y_test = train_test_split(train_df, labels_df, test_size=0.2, random_state=42)

# 2. Load Models
rf_model = joblib.load('forward_model.joblib')
xgb_model = joblib.load('forward_model_xgb.joblib')

# 3. Predict
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# 4. Compare Metrics
targets = labels_df.columns

print("\n" + "="*85)
print(f"{'Target Variable':<18} | {'Random Forest R²':<18} | {'XGBoost R²':<18} | {'Absolute Winner'}")
print("="*85)

for i, col in enumerate(targets):
    rf_r2 = r2_score(y_test.iloc[:, i], rf_preds[:, i])
    xgb_r2 = r2_score(y_test.iloc[:, i], xgb_preds[:, i])
    
    # Calculate advantage percentage based on remaining error margin
    # Using simple "greater is better" for R2 string
    if xgb_r2 > rf_r2:
        winner = "XGBoost 🏆"
    else:
        winner = "Random Forest 🏆"
    
    print(f"{col:<18} | {rf_r2:<18.4f} | {xgb_r2:<18.4f} | {winner}")

print("="*85)
