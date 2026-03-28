# Boom: Trajectory Unknown Challenge

## What This Solution Does

A machine learning solution to predict and optimize asteroid impact debris for the Mox-95 system using a **Random Forest Regressor**.

---

## ✅ Criteria Met

### Forward Prediction
> *Predict how materials break apart after impact.*

- Trained on 3,000+ simulated impact events
- **R² = 96.4%** accuracy on P80 (fragment size)
- **R² = 91.0%** accuracy on R95 (debris spread)

### Inverse Design
> *20 scenarios where P80 ∈ [96, 101], R95 ≤ 175, energy minimized.*

| Constraint | Required | Achieved |
|---|---|---|
| P80 range | 96 – 101 | ✅ All 20 pass |
| R95 limit | ≤ 175 | ✅ All 20 pass |
| Energy | Minimize | Avg **2.82** |
| Scenarios | 20 | ✅ 20 |

---

## How It Works

1. **Train** — Random Forest learns impact physics from training data
2. **Predict** — Model predicts debris outcomes for unseen test inputs
3. **Optimize** — Nelder-Mead algorithm minimizes energy while the model enforces P80/R95 constraints
4. **Verify** — Every scenario validated before submission

---

## Key Insight

> The valid low-energy parameter space is extremely narrow in 8 dimensions. A brute-force search of 2 million random samples found **zero** valid configurations. Only a directed optimizer (Nelder-Mead) paired with a trained surrogate model can navigate this space — proving that intelligent optimization is essential for physics-constrained design problems.

---

## How to Run

```bash
pip install pandas numpy scikit-learn scipy joblib
python train_model.py
python create_forward_submission.py
python create_inverse_submission.py
python verify_submission.py
```

---

## Submission Files

| File | Purpose |
|---|---|
| `forward_prediction/prediction_submission.csv` | Test predictions |
| `inverse_design/submission.csv` | 20 optimized scenarios |

---

## Reproducibility
- Fixed `random_state=42` ensures identical results on every run
- Trained model saved as `forward_model.joblib`
- `verify_submission.py` confirms all constraints pass
