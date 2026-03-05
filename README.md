# Boom: Trajectory Unknown Challenge
For more information and scoring metrics, see the Challenge Website: https://www.freelancer.com/boom

The challenge has two parts: 
- **Forward prediction**: Mandatory for your submission to be accepted; build a physics-based, data-driven predictive model to predict ejecta outcomes given impact scenarios.
- **Inverse design** Optional but provides extra points in your score; using your trained model and an efficient search algorithm, propose 20 impact scenarios with ejecta outcomes that satisfy a given set of constraints.

Please read the following challenge descriptions carefully. 

---

## 1) Forward Prediction

### Data description
**Impact parameters:**
Each impact scenario is **partially** described by 8 parameters:
- energy - Impact energy 
- angle_rad - Impact angle from horizon (in radians)
- coupling - Energy transfer efficiency between asteroid and surface
- strength - Material strength
- porosity - Material porosity
- gravity - Surface gravity
- atmosphere - Atmospheric density at the impact altitude
- shape_factor - Fragment irregularity (a higher value indicates that the material tends to fracture into highly irregular shards) 

**Ejecta Outcomes:**
The aftermath of each impact event is described by 6 statistical measures: 

- **P80** - Fragment diameter (mm) below which 80% of the total ejected mass lies
  - `P80 = d` such that `Σ(m_i | d_i ≤ d) = 0.8 × M_total`

- **fines_frac** - Fraction of total ejecta mass contributed by fragments smaller than 40mm diameter
  - `fines_frac = Σ(m_i | d_i < 40mm) / M_total`

- **oversize_frac** - Fraction of total ejecta mass contributed by fragments larger than 120mm diameter
  - `oversize_frac = Σ(m_i | d_i > 120mm) / M_total`

- **R95** - Landing distance (m) below which 95% of the total ejected mass lies
  - `R95 = r` such that `Σ(m_i | r_i ≤ r) = 0.95 × M_total`

- **R50_fines** - Median landing distance (m) for fragments smaller than 40mm (mass-weighted)
  - `R50_fines = r` such that `Σ(m_i | d_i < 40mm, r_i ≤ r) = 0.5 × Σ(m_i | d_i < 40mm)`

- **R50_oversize** - Median landing distance (m) for fragments larger than 120mm (mass-weighted)
  - `R50_oversize = r` such that `Σ(m_i | d_i > 120mm, r_i ≤ r) = 0.5 × Σ(m_i | d_i > 120mm)`

Notation:
- `d_i` = diameter of fragment i
- `r_i` = landing distance of fragment i  
- `m_i` = mass of fragment i (∝ d_i³)
- `M_total` = Σm_i (total ejected mass)

### Files provided
- `forward_prediction/train.csv` (impact scenarios for training)
- `forward_prediction/train_labels.csv` (ejecta outcomes for training)
- `forward_prediction/test.csv` (impact scenarios for scoring)
- `forward_prediction/prediction_submission_template.csv` (submission template)

### Submission format
Submit `prediction_submission.csv` to your repository with the exact columns:
- `scenario_id`
- `P80`
- `fines_frac`
- `oversize_frac`
- `R95`
- `R50_fines`
- `R50_oversize`

Note:
- `scenario_id` must match the row index in `forward_prediction/test.csv` (0-based).
- One row per test scenario.

---

## 2) Inverse Design

Constraints:
- Ejecta fragment diameter: `96 ≤ P80 ≤ 101 mm`
- Ejecta range: `R95 ≤ 175 m`
- Input parameters must be within specified bounds (see `inverse_design/contraints.json` for input bounds)

### Files Provided
- `inverse_design/constraints.json` (constraints on input and output parameters)
- `inverse_design/design_submission_template.csv` (submission template)

### Submission format
Submit `design_submission.csv` to your repository with the exact columns:
- `submission_id`
- `energy`
- `angle_rad`
- `coupling`
- `strength`
- `porosity`
- `gravity`
- `atmosphere`
- `shape_factor`
