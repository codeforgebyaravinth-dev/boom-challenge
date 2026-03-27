import pandas as pd
import joblib

# Load test data and template
test_df = pd.read_csv('forward_prediction/test.csv')
template_df = pd.read_csv('forward_prediction/prediction_submission_template.csv')

# Load model
model = joblib.load('forward_model.joblib')

# Predict
preds = model.predict(test_df)

# Create submission df
# The template likely has submission_id and the 6 target columns
submission_df = pd.DataFrame(preds, columns=model.feature_names_out_ if hasattr(model, 'feature_names_out_') else ['P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize'])
submission_df.insert(0, 'scenario_id', test_df.index + 1)
# make sure columns match template
submission_df = submission_df[template_df.columns]

submission_df.to_csv('forward_prediction/prediction_submission.csv', index=False)
print("Forward prediction submission saved to forward_prediction/prediction_submission.csv")
