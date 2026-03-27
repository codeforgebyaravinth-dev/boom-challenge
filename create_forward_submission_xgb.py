import pandas as pd
import joblib

# Load test data and template
test_df = pd.read_csv('forward_prediction/test.csv')
template_df = pd.read_csv('forward_prediction/prediction_submission_template.csv')

# Load trained XGBoost model
model = joblib.load('forward_model_xgb.joblib')

# Predict
preds = model.predict(test_df)

# Create submission df
cols = model.feature_names_out_ if hasattr(model, 'feature_names_out_') else ['P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize']
submission_df = pd.DataFrame(preds, columns=cols)
submission_df.insert(0, 'scenario_id', test_df.index + 1)

# Ensure columns match template exactly
submission_df = submission_df[template_df.columns]

submission_df.to_csv('forward_prediction/prediction_submission_xgb.csv', index=False)
print("Ultra-robust XGBoost forward prediction saved to forward_prediction/prediction_submission_xgb.csv")
