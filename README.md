# Titanic Survival Analysis

Exploratory data analysis, statistical analysis, and machine learning
prediction of passenger survival using the Kaggle Titanic dataset
("Titanic: Machine Learning from Disaster").

## Project Overview

This project performs a complete data science workflow on the Titanic
dataset:

1. **Data Acquisition** — loading and inspecting `train.csv`, `test.csv`,
   and `gender_submission.csv`.
2. **Data Cleaning** — detecting and handling missing values (`Age`,
   `Cabin`, `Embarked`), and checking for duplicate records.
3. **Data Visualisation** — histogram of ages, passenger class bar chart,
   boxplot of age by class, scatter plot of age vs fare, correlation
   heatmap, and pairplot.
4. **Statistical Analysis** — descriptive statistics, frequency
   distributions, and correlation analysis with survival.
5. **Machine Learning** — a Logistic Regression classifier trained to
   predict passenger survival, evaluated with accuracy, a confusion
   matrix, and a classification report.
6. **Discussion & Conclusion** — findings, limitations, and
   recommendations, written up in a full report.

## Repository Structure

```
├── data/
│   ├── train.csv                # Training data (891 records, with Survived label)
│   ├── test.csv                 # Test data (418 records, no label)
│   └── gender_submission.csv    # Kaggle sample submission file
├── notebooks/
│   └── Titanic_Analysis.ipynb   # Full Jupyter Notebook (executed, with outputs)
├── src/
│   └── analysis.py              # Same analysis as a plain Python script
├── figures/
│   ├── 01_age_histogram.png
│   ├── 02_pclass_bar.png
│   ├── 03_age_by_class_boxplot.png
│   ├── 04_age_vs_fare_scatter.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_pairplot.png
│   └── 07_confusion_matrix.png
├── report/
│   └── Titanic_Analysis_Report.pdf
└── README.md
```

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Analysis

**Option 1 — Jupyter Notebook (recommended):**

```bash
jupyter notebook notebooks/Titanic_Analysis.ipynb
```

**Option 2 — Plain Python script:**

```bash
cd src
python analysis.py
```

Both produce the same output: printed statistics/tables in the console
and PNG figures saved to a `figures/` folder relative to where the
script/notebook is run.

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 891 training records, 12 columns |
| Missing data | `Age` 19.9%, `Cabin` 77.1%, `Embarked` 0.2% |
| Duplicate rows | 0 |
| Overall survival rate | 38.4% |
| Strongest positive correlation with survival | `Has_Cabin` (r ≈ 0.317) |
| Strongest negative correlation with survival | `Pclass` (r ≈ −0.338) |
| Model | Logistic Regression |
| Test accuracy | 81.0% |

## Methodology Summary

- **Missing values:** `Age` was imputed with the median age within each
  `Pclass`/`Sex` group; `Cabin` was converted into a `Has_Cabin` binary
  indicator (too sparse, ~77% missing, to impute directly); `Embarked`
  was imputed with its mode (2 missing values).
- **Features used for modelling:** `Pclass`, `Sex`, `Age`, `SibSp`,
  `Parch`, `Fare`, `Embarked`, `Has_Cabin` (categoricals one-hot encoded).
- **Model:** Logistic Regression (scikit-learn), trained on an 80/20
  stratified train/test split of `train.csv`, features standardised with
  `StandardScaler`.

## Report

The full write-up — including introduction, dataset description,
methodology, results, discussion, limitations, recommendations, and a
Python code appendix — is available in [`report/Titanic_Analysis_Report.pdf`](report/Titanic_Analysis_Report.pdf)

## Author

Essilfie Bernard
01256139B, BTech Computer Science Top-Up
Accra Technical University, Computer Science Department


## Dataset Source

Kaggle. *Titanic: Machine Learning from Disaster.*
https://www.kaggle.com/competitions/titanic


