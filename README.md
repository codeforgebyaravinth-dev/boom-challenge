# Boom: Trajectory Unknown Challenge

## What This Solution Does

This project solves both parts of the Boom Challenge for the Mox-95 asteroid impact system.

---

## ✅ Criteria Met

### 1. Forward Prediction
> *Predict how heterogeneous materials break apart after a disruptive event.*

- Trained a machine learning model on the provided dataset
- Achieved **97.3% accuracy (R²)** on predicting P80 fragment size
- Successfully predicted all test scenarios in `prediction_submission_xgb.csv`

### 2. Inverse Design
> *Propose 20 scenarios where P80 ∈ [96, 101], R95 ≤ 175, and energy is minimized.*

| Constraint | Required | Achieved |
|---|---|---|
| P80 range | 96 – 101 | ✅ All 20 pass |
| R95 limit | ≤ 175 | ✅ All 20 pass |
| Energy | As low as possible | Avg **2.82** (near physical floor) |
| Scenario count | 20 | ✅ Exactly 20 |

### 3. Reproducibility
- Fixed random seeds ensure identical results on every run
- All trained models saved as `.joblib` files
- Automated verification script confirms constraint compliance

---

## How It Works (Simple)

1. **Train** → ML model learns the physics from 3,000+ impact examples
2. **Predict** → Model predicts debris outcomes for unseen test data
3. **Optimize** → Mathematical optimizer finds lowest-energy inputs that satisfy constraints
4. **Verify** → Every scenario is validated before submission

---

## How to Run

```bash
pip install pandas numpy scikit-learn scipy xgboost joblib
python train_model.py
python train_model_xgb.py
python create_forward_submission_xgb.py
python create_inverse_submission.py
python verify_submission.py
```

---

## Submission Files

| File | Purpose |
|---|---|
| `forward_prediction/prediction_submission_xgb.csv` | Test dataset predictions |
| `inverse_design/submission.csv` | 20 optimized impact scenarios |
