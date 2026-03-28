# Boom: Trajectory Unknown Challenge 🚀

## Executive Summary
A physics-informed Machine Learning solution to predict asteroid impact fragment distributions and optimize impact parameters for the Mox-95 system. This project uses a **deliberate hybrid strategy** — deploying two different ML architectures (XGBoost + Random Forest), each where it performs strongest.

---

## The Challenge
| Part | Goal |
|---|---|
| **Forward Prediction** | Predict debris outcomes (P80, R95, etc.) from 8 impact parameters |
| **Inverse Design** | Propose 20 impact scenarios where `96 ≤ P80 ≤ 101`, `R95 ≤ 175`, and **energy is minimized** |

---

## Approach & Methodology

### Forward Prediction — XGBoost Surrogate Engine
Trained an **XGBoost Multi-Output Regressor** (300 estimators, learning rate 0.05) on the provided training dataset to serve as a high-fidelity surrogate physics engine.

| Target | R² Accuracy |
|---|---|
| **P80** | **97.33%** |
| oversize_frac | 98.81% |
| fines_frac | 94.28% |
| R95 | 90.58% |

### Inverse Design — Random Forest + Nelder-Mead Optimization
A 3-stage pipeline:

1. **Baseline Discovery**: Exhaustively searched 3,000+ training events to find 20 physically viable seeds approaching the P80 target range.
2. **Surrogate Optimization**: Used a **Random Forest Regressor** as the surrogate model, paired with **SciPy's Nelder-Mead** derivative-free local search optimizer to systematically minimize energy.
3. **Constraint Enforcement**: Applied heavy penalty functions during optimization to guarantee `P80 ∈ [96, 101]` and `R95 ≤ 175` across all 20 scenarios.

**Result**: Average energy of **2.82** with **20/20 rows passing constraints** ✅

### Why Two Models? (Hybrid Strategy)

| Metric | Random Forest | XGBoost |
|---|---|---|
| **P80 Accuracy (R²)** | 96.37% | **97.33%** 🏆 |
| **Avg Inverse Energy** | **2.82** 🏆 | 3.13 |
| **Constraint Pass Rate** | **20/20** 🏆 | 19/20 |

- **XGBoost** excels at raw prediction accuracy → used for **Forward Prediction**
- **Random Forest** has smoother decision boundaries → used for **Inverse Design** (allows Nelder-Mead to glide efficiently toward energy minimums)

---

## Repository Structure
```
├── train_model.py                  # Train Random Forest surrogate
├── train_model_xgb.py              # Train XGBoost surrogate
├── create_forward_submission_xgb.py # Generate test predictions (XGBoost)
├── create_inverse_submission.py     # Nelder-Mead optimization (RF)
├── create_inverse_xgb.py           # Nelder-Mead optimization (XGBoost)
├── compare_models.py               # Side-by-side model comparison
├── verify_submission.py            # Automated constraint verification
├── fix_submission.py               # Stochastic sweep constraint fixer
├── forward_model.joblib            # Trained Random Forest model
├── forward_model_xgb.joblib        # Trained XGBoost model
├── forward_prediction/
│   ├── train.csv                   # Training dataset
│   ├── train_labels.csv            # Training labels
│   ├── test.csv                    # Hidden test inputs
│   └── prediction_submission_xgb.csv # Final forward predictions
└── inverse_design/
    └── submission.csv              # Final 20 optimized scenarios
```

## Tech Stack
- **Python 3.13** | **scikit-learn** | **XGBoost** | **SciPy** | **pandas** | **NumPy** | **joblib**

## How to Run
```bash
pip install pandas numpy scikit-learn scipy xgboost joblib
python train_model.py
python train_model_xgb.py
python create_forward_submission_xgb.py
python create_inverse_submission.py
python verify_submission.py
python compare_models.py
```
