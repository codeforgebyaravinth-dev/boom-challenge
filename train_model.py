import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(train_df, labels_df, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
for i, col in enumerate(labels_df.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col} - MSE: {mse:.4f}, R2: {r2:.4f}")

# Save model
joblib.dump(model, 'forward_model.joblib')
print("\nModel saved to forward_model.joblib")
