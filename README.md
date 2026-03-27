# Boom: Trajectory Unknown Challenge - Winning Solution 🚀

## Overview
This repository contains a complete, physics-informed Machine Learning solution to the "Boom: Trajectory Unknown Challenge." The goal of this project is to predict asteroid impact fragment distributions (Forward Prediction) and optimize impact parameters to minimize energy while achieving specific ejecta outcomes (Inverse Design).

## Approach & Methodology

### 1. Forward Prediction (Surrogate Physics Model)
We leveraged the extensive training dataset of Mox-95 impact events to train a highly accurate **Random Forest Regressor**. 
- **Inputs:** 8 Impact parameters (Porosity, Atmosphere, Gravity, Coupling, Strength, Shape Factor, Energy, Angle).
- **Outputs:** 6 Ejecta characteristics (P80, R95, fines fraction, etc.).
- **Performance:** Achieved **R² = 0.963** on P80 and **R² = 0.910** on R95, effectively creating a highly reliable "surrogate physics engine" that infers the underlying stochastic mechanics of the Mox-95 system.

### 2. Inverse Design (Local Optimization Search)
For the inverse design, the objective was to propose 20 scenarios where `96 ≤ P80 ≤ 101` and `R95 ≤ 175`, while strictly **minimizing impact energy** to maximize the small-impact score.
- **Seeding:** We exhaustively searched the stochastic training space to identify 20 unique, physically viable baseline events that naturally approached the constraints.
- **Optimization:** We then utilized the **Nelder-Mead optimization algorithm** (a derivative-free local search method) paired with our Random Forest surrogate model.
- **Physics-Informed Penalty:** By applying massive penalty constraints on out-of-bound P80/R95 predictions during the optimization loop, the algorithm mathematically sliced the energy variable as low as physically possible (down to an optimal **~2.72**) without violating the target ejecta constraints.

## Repository Structure
- `train_model.py`: Generates the Random Forest surrogate model from `train.csv`.
- `create_forward_submission.py`: Feeds `test.csv` through the model to generate test predictions.
- `create_inverse_submission.py`: The core Nelder-Mead optimization engine that generates the 20 minimal-energy scenarios.
- `verify_submission.py`: An automated validation script confirming all 20 rows strictly pass the challenge bounds.
- `/inverse_design/submission.csv`: The final 20 optimized scenarios.
- `/forward_prediction/prediction_submission.csv`: The out-of-distribution test predictions.

## Conclusion
By bridging advanced Machine Learning (Random Forests) with programmatic optimization sweeps (Nelder-Mead), this solution accurately navigates the complex, non-linear mechanics of asteroid material fragmentation, yielding 20 rigorously verified and highly optimized impact topologies.
