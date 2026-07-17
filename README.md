# Smart Healthcare & Lifestyle-Based Heart Disease Prediction

A machine learning project that predicts the likelihood of **heart disease** from a patient's
lifestyle and clinical profile. The notebook runs the full workflow end to end: exploratory
data analysis, statistical significance testing, feature engineering, a leakage-safe
preprocessing pipeline, multi-model comparison, hyperparameter tuning, evaluation, and
single-patient inference.

---

## Overview

Given a patient's demographic, lifestyle, and clinical indicators, the model estimates the
**probability of heart disease** and classifies the patient as *Heart Disease* or
*No Heart Disease*. All preprocessing is wrapped in a scikit-learn `Pipeline` so that scaling
and encoding are fit only on training folds — preventing data leakage.

## Dataset

- **File:** `smart_healthcare_dataset.csv`
- **Rows:** 5,000 patients · **Target:** `heart_disease` (0/1), class balance ~70% / 30%
- **Note:** synthetic dataset — see *Limitations*.

### Original features

| Group | Columns |
|-------|---------|
| Demographic | `age`, `gender` |
| Clinical (numeric) | `bmi`, `blood_pressure`, `cholesterol`, `glucose` |
| Lifestyle | `exercise_level`, `smoking`, `alcohol` |
| Symptoms / conditions (binary) | `fatigue`, `chest_pain`, `dizziness`, `diabetes`, `stroke` |
| Dropped | `health_risk_score` (constant = 100 for every row — no information) |

### Engineered features

| Feature | Definition |
|---------|-----------|
| `comorbidity_count` | Sum of `diabetes`, `stroke`, `chest_pain`, `fatigue`, `dizziness` |
| `bmi_category` | Clinical BMI bins (Underweight / Normal / Overweight / Obese) |
| `bp_category` | Blood-pressure stages (Normal / Elevated / Stage1 / Stage2) |
| `glucose_category` | Glucose bins (Normal / Prediabetes / Diabetic) |
| `age_x_diabetes` | Interaction term (`age` × `diabetes`) |

## Workflow

1. **EDA** — distributions, box plots, correlation heatmap, categorical comparisons.
2. **Statistical testing** — Welch's t-test + Cohen's *d* (continuous); chi-square + Cramér's *V*
   (categorical). Distinguishes statistical significance from practical effect size.
3. **Feature engineering** — comorbidity count, clinical category bins, interaction term.
4. **Preprocessing** — a `ColumnTransformer` pipeline: numeric → median impute + `StandardScaler`;
   categorical → most-frequent impute + `OneHotEncoder`; binary → most-frequent impute.
5. **Model comparison** — 5-fold `StratifiedKFold` CV across Logistic Regression, Random Forest,
   Gradient Boosting, AdaBoost, SVC; scored on accuracy, balanced accuracy, precision, recall,
   F1, ROC-AUC, PR-AUC.
6. **Hyperparameter tuning** — `GridSearchCV` on the selected model (optimizing average precision).
7. **Evaluation** — held-out test set: classification report, confusion matrix, ROC & PR curves,
   logistic-regression coefficient interpretation.
8. **Inference** — predict on a single patient, returning the probability of disease.

## Model Selection

**Logistic Regression** is the final model — chosen for its highest recall (most important in a
screening context), full interpretability (coefficients / odds ratios), and simplicity. AdaBoost
scored marginally higher on PR-AUC but was statistically indistinguishable.

## Results (test set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.996 |
| Balanced Accuracy | 0.996 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9997 |

Only ~5 misclassifications out of 1,000 test patients, with very few false negatives — the
preferred error profile for a clinical screening tool.

## Limitations & Honest Notes

- **Synthetic, near-deterministic dataset.** `age` alone almost fully separates the target
  (≈0% disease under age 50; ≈98% over 65); `diabetes` and `stroke` are similarly decisive. This
  is why metrics approach 1.000 — it will **not** transfer to real clinical data (expect ~0.75–0.90
  AUC there).
- **No methodological leakage.** Preprocessing is fit inside CV folds and the split is stratified;
  the perfect scores come from the data, not the code.
- **Collinearity from engineered features.** Raw and derived versions of the same variable are
  kept together (`bmi`/`bmi_category`, `blood_pressure`/`bp_category`, `glucose`/`glucose_category`,
  `age`/`age_x_diabetes`), and `comorbidity_count` is the sum of binary features already present.
  This can distort logistic-regression coefficients. Engineered features did not improve metrics,
  so a simpler raw-feature model is equally valid.
- `stroke` / `chest_pain` are arguably co-morbid outcomes rather than clean upstream predictors.

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

```
Smart-Healthcare-and-Lifestyle-Prediction/
├── smart_healthcare.ipynb        # Full analysis notebook
├── smart_healthcare_dataset.csv  # Dataset (5,000 patients)
├── requirements.txt              # Dependencies
└── README.md
```

## Author

Suyog Karki — [GitHub](https://github.com/suyogkarki1)
