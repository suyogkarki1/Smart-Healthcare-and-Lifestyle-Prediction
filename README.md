# Smart Healthcare & Lifestyle-Based Heart Disease Prediction

A machine learning project that predicts the likelihood of **heart disease** from a patient's lifestyle and clinical profile. The notebook runs the full workflow end to end: exploratory data analysis, statistical significance testing, feature engineering, a leakage-safe preprocessing pipeline, multi-model comparison, hyperparameter tuning, evaluation, and single-patient inference.

---

## Overview

Given a patient's demographic, lifestyle, and clinical indicators, the model estimates the **probability of heart disease** and classifies the patient as *Heart Disease* or *No Heart Disease*. All preprocessing is wrapped in a scikit-learn `Pipeline` so that scaling and encoding are fit only on training folds — preventing data leakage.

*Note on Data Leakage:* In the initial model iteration, the `age` feature was found to perfectly divide the target class mechanically due to synthetic data rules. To build a robust, generalizable model that evaluates genuine physiological indicators, **the `age` feature was dropped from the training process.**

## Dataset

- **File:** `smart_healthcare_dataset.csv`
- **Rows:** 5,000 patients · **Target:** `heart_disease` (0/1), class balance ~70% / 30%
- **Note:** synthetic dataset — see *Limitations*.

### Original features

| Group | Columns | Status |
|-------|---------|--------|
| Demographic | `gender` | Kept |
| Demographic | `age` | **Dropped** (Resolved data leakage/linear separability) |
| Clinical (numeric) | `bmi`, `blood_pressure`, `cholesterol`, `glucose` | Kept |
| Lifestyle | `exercise_level`, `smoking`, `alcohol` | Kept |
| Symptoms / conditions | `fatigue`, `chest_pain`, `dizziness`, `diabetes`, `stroke` | Kept |
| Dropped | `health_risk_score` | Dropped (Constant value = 100 for every row) |

### Engineered features

| Feature | Definition |
|---------|-----------|
| `comorbidity_count` | Sum of `diabetes`, `stroke`, `chest_pain`, `fatigue`, `dizziness` |
| `bmi_category` | Clinical BMI bins (Underweight / Normal / Overweight / Obese) |
| `bp_category` | Blood-pressure stages (Normal / Elevated / Stage1 / Stage2) |
| `glucose_category` | Glucose bins (Normal / Prediabetes / Diabetic) |
| `age_x_diabetes` | *Removed* (Dropped along with the core `age` feature) |

## Workflow

1. **EDA** — distributions, box plots, correlation heatmap, categorical comparisons.
2. **Data Leakage Diagnosis** — Analyzed the `age` variable using boxplots to confirm artificial linear separability between the target classes.
3. **Feature Engineering** — Created a comorbidity count and clinical category bins. 
4. **Preprocessing** — A `ColumnTransformer` pipeline: numeric → median impute + `StandardScaler`; categorical → most-frequent impute + `OneHotEncoder`; binary → most-frequent impute.
5. **Model Comparison** — 5-fold `StratifiedKFold` CV across Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVC.
6. **Hyperparameter Tuning** — `GridSearchCV` on Logistic Regression optimizing for average precision across L1, L2, and Elastic Net penalties.
7. **Evaluation** — Evaluated on a held-out test set using a classification report, confusion matrix, and coefficient breakdown.

## Model Selection

**Logistic Regression with Elastic Net Regularization** is the final chosen setup. The grid search selected an optimal parameter set of **`C=1`** and **`l1_ratio=0.75`** via the `saga` solver. 

* **Balanced Penalty Strength ($C=1$):** Prevents individual features from dominating the calculations.
* **Hybrid Penalty Strategy (`l1_ratio=0.75`):** Automatically assigns 75% of its optimization to L1 (Lasso) to drop redundant binned features, and 25% to L2 (Ridge) to safely shrink down the weights of the remaining active features.

## Results (test set)

After dropping the `age` shortcut, the model achieved a highly robust, realistic, and generalizable performance profile:

| Metric | Score |
|--------|-------|
| Accuracy | 0.9650 |
| Balanced Accuracy | 0.9531 |
| ROC-AUC | 0.9950 |
| PR-AUC | 0.9892 |

### Confusion Matrix Breakdown
* **True Negatives:** 688 patients correctly predicted as having *No Heart Disease*.
* **True Positives:** 277 patients correctly predicted as having *Heart Disease*.
* **False Positives:** 12 patients with *No Heart Disease* accidentally predicted to have it.
* **False Negatives:** 23 patients with *Heart Disease* accidentally missed by the model.

The model captures **over 92%** of active heart disease cases primarily relying on clinical metrics like `bin__diabetes`, `bin__stroke`, and `num__bmi`.

## Limitations & Honest Notes

- **Synthetic Data Shortcuts:** The original `smart_healthcare_dataset.csv` contained a near-deterministic rule where `age` perfectly split the outcome (no heart disease under 70, nearly 100% heart disease over 65). Leaving `age` in resulted in an inflated coefficient weight of `60.33` and an unrealistic `0.9996` PR-AUC. Removing `age` was mandatory to force the model to learn real health metrics.
- **Multicollinearity:** Keeping continuous numeric features (`bmi`, `glucose`, `blood_pressure`) alongside their engineered binned counterparts (`bmi_category`, etc.) creates high correlation. The model natively handled this during tuning; the L1 component of the Elastic Net penalty assigned a coefficient of exactly `0.0` to the redundant categorical slices, keeping only the raw continuous scales.

## How to Run

```bash
git clone https://github.com/suyogkarki1/Smart-Healthcare-and-Lifestyle-Prediction.git
cd Smart-Healthcare-and-Lifestyle-Prediction
pip install -r requirements.txt
jupyter notebook smart_healthcare.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All).

## Tech Stack

Python 3.12 · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn

## Project Structure

