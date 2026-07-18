# %% [markdown]
# # Titanic Survival Analysis
# ### A Statistical and Machine Learning Study of the Kaggle Titanic Dataset
#
# **Author:** Essilfie Bernard_01256139B
# **Dataset:** Kaggle Titanic (`train.csv`, `test.csv`, `gender_submission.csv`)
#
# This notebook performs data acquisition, cleaning, exploratory visualisation,
# statistical analysis, and builds a Logistic Regression classifier to predict
# passenger survival.

# %% [markdown]
# ## 0. Setup: Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", None)

# %% [markdown]
# ## Task 1: Data Acquisition
#
# We load the Titanic training dataset (`train.csv`), which contains 891
# passenger records and the ground-truth `Survived` label. The `test.csv`
# file (418 records) has no `Survived` column — in the original Kaggle
# competition it is used only for submission scoring. Because our brief
# requires evaluating model performance (accuracy, confusion matrix,
# classification report), we build our train/test split from `train.csv`,
# which is the only file with known outcomes. `test.csv` and
# `gender_submission.csv` are still loaded and inspected below for
# completeness.

# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
gender_submission = pd.read_csv("data/gender_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Gender submission shape:", gender_submission.shape)

# %% [markdown]
# ### Dataset dimensions

# %%
print(f"The training dataset has {train.shape[0]} rows and {train.shape[1]} columns.")

# %% [markdown]
# ### Column names

# %%
print(train.columns.tolist())

# %% [markdown]
# ### First five observations

# %%
train.head()

# %% [markdown]
# ### Data types

# %%
train.dtypes

# %% [markdown]
# **Interpretation:** The dataset mixes numeric types (`int64`, `float64`)
# with text/object columns (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`).
# `Survived` and `Pclass` are stored as integers but are actually
# categorical variables (binary outcome and ordinal class respectively).

# %% [markdown]
# ## Task 2: Data Cleaning

# %% [markdown]
# ### 2.1 Detect missing values

# %%
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_table = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_table = missing_table[missing_table["Missing Count"] > 0].sort_values(
    "Missing Count", ascending=False
)
missing_table

# %% [markdown]
# **Interpretation:** `Cabin` is missing for about 77% of passengers, `Age`
# is missing for about 20%, and `Embarked` is missing for just 2 passengers.

# %% [markdown]
# ### 2.2 Handle missing values
#
# **Preprocessing decisions (with justification):**
#
# - **`Age` (177 missing):** Imputed with the **median age**, calculated
#   separately for each `Pclass`/`Sex` combination where possible, falling
#   back to the overall median. The median is used instead of the mean
#   because age is right-skewed and the median is robust to outliers
#   (e.g. infants and elderly passengers).
# - **`Cabin` (687 missing, ~77%):** Too sparse to impute reliably. Instead
#   of dropping the column outright (it may still carry signal about deck
#   level and therefore socio-economic status), it is converted into a
#   binary indicator `Has_Cabin` (1 if a cabin number was recorded, 0
#   otherwise). The original text column is then dropped.
# - **`Embarked` (2 missing):** Imputed with the **mode** (most frequent
#   port, "S" — Southampton), since only 2 values are missing and the
#   variable is categorical with a strongly dominant category.
# - **`Fare` (0 missing in train, but present in test):** Any missing fare
#   would be imputed with the median fare for the passenger's class, for
#   consistency; not needed for the training set.

# %%
train_clean = train.copy()

# Age: impute with median age per Pclass & Sex group
train_clean["Age"] = train_clean.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
train_clean["Age"] = train_clean["Age"].fillna(train_clean["Age"].median())

# Cabin: convert to binary indicator, then drop original column
train_clean["Has_Cabin"] = train_clean["Cabin"].notnull().astype(int)
train_clean = train_clean.drop(columns=["Cabin"])

# Embarked: impute with mode
train_clean["Embarked"] = train_clean["Embarked"].fillna(
    train_clean["Embarked"].mode()[0]
)

print("Remaining missing values after cleaning:")
print(train_clean.isnull().sum())

# %% [markdown]
# ### 2.3 Detect duplicated observations

# %%
duplicate_count = train_clean.duplicated().sum()
print(f"Number of fully duplicated rows: {duplicate_count}")

# %% [markdown]
# ### 2.4 Remove duplicates if any exist

# %%
if duplicate_count > 0:
    train_clean = train_clean.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate rows. New shape: {train_clean.shape}")
else:
    print("No duplicate rows were found — no rows removed.")

# %% [markdown]
# **Summary of preprocessing:** `Age` was imputed using group-wise medians
# (Pclass x Sex) to preserve realistic age distributions across
# socio-economic groups; `Cabin` was converted to a presence indicator
# rather than dropped entirely, retaining potential signal while avoiding
# unreliable imputation of ~77% missing text data; `Embarked` was imputed
# with its mode since only two records were affected. No duplicate rows
# were present in the dataset, so no observations were removed at this
# step.

# %% [markdown]
# ## Task 3: Data Visualisation

# %% [markdown]
# ### 3.1 Histogram of Passenger Ages

# %%
plt.figure(figsize=(8, 5))
plt.hist(train_clean["Age"], bins=30, color="steelblue", edgecolor="black")
plt.title("Distribution of Passenger Ages (Titanic)")
plt.xlabel("Age (years)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("figures/01_age_histogram.png")
plt.show()

# %% [markdown]
# **Interpretation:** The age distribution is right-skewed and concentrated
# between roughly 20 and 40 years, with a noticeable secondary peak among
# young children (a result of the group-median imputation reinforcing the
# existing concentration around common adult ages). Very few passengers
# were older than 60.

# %% [markdown]
# ### 3.2 Bar Chart of Passenger Class Distribution

# %%
plt.figure(figsize=(7, 5))
class_counts = train_clean["Pclass"].value_counts().sort_index()
plt.bar(class_counts.index.astype(str), class_counts.values,
        color=["#4C72B0", "#DD8452", "#55A868"])
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("figures/02_pclass_bar.png")
plt.show()

# %% [markdown]
# **Interpretation:** Third class passengers form the largest group
# (about 55% of all passengers), roughly double the number in first or
# second class. This reflects the Titanic's passenger composition, with
# more budget-fare travellers than premium-fare travellers.

# %% [markdown]
# ### 3.3 Boxplot of Age by Passenger Class

# %%
plt.figure(figsize=(7, 5))
sns.boxplot(data=train_clean, x="Pclass", y="Age", hue="Pclass",
            palette="Set2", legend=False)
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age (years)")
plt.tight_layout()
plt.savefig("figures/03_age_by_class_boxplot.png")
plt.show()

# %% [markdown]
# **Interpretation:** Median age decreases as class number increases:
# first-class passengers are typically older (median around late 30s),
# while third-class passengers skew younger (median around mid-20s),
# consistent with wealthier, older travellers affording premium fares.

# %% [markdown]
# ### 3.4 Scatter Plot of Age versus Fare

# %%
plt.figure(figsize=(7, 5))
plt.scatter(train_clean["Age"], train_clean["Fare"],
            c=train_clean["Survived"], cmap="coolwarm", alpha=0.6, edgecolor="k", linewidth=0.3)
plt.title("Age vs Fare (coloured by Survival)")
plt.xlabel("Age (years)")
plt.ylabel("Fare (£)")
plt.colorbar(label="Survived (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig("figures/04_age_vs_fare_scatter.png")
plt.show()

# %% [markdown]
# **Interpretation:** There is no strong linear relationship between age
# and fare — passengers of all ages paid a wide range of fares. A small
# cluster of high-fare passengers (mostly first class) shows a higher
# concentration of survivors (red points), hinting that fare/class was
# more predictive of survival than age alone.

# %% [markdown]
# ### 3.5 Correlation Heatmap

# %%
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Has_Cabin"]
corr_matrix = train_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig("figures/05_correlation_heatmap.png")
plt.show()

# %% [markdown]
# **Interpretation:** `Survived` correlates most positively with
# `Has_Cabin` and `Fare`, and most negatively with `Pclass` (since higher
# class numbers mean lower socio-economic status). `Pclass` and `Fare` are
# strongly negatively correlated with each other, as expected (higher
# fares are paid for better classes, numbered 1).

# %% [markdown]
# ### 3.6 Pairplot of Selected Numerical Variables

# %%
pairplot_cols = ["Age", "Fare", "Pclass", "Survived"]
pp = sns.pairplot(train_clean[pairplot_cols], hue="Survived", palette="coolwarm", diag_kind="hist")
pp.fig.suptitle("Pairplot of Age, Fare, Pclass by Survival", y=1.02)
pp.savefig("figures/06_pairplot.png")
plt.show()

# %% [markdown]
# **Interpretation:** The pairplot shows survivors (orange/red) are more
# concentrated in lower `Pclass` values (i.e. first class) and higher fare
# bands, while non-survivors are spread more broadly across third class
# and lower fares. Age shows more overlap between the two survival groups,
# confirming it is a weaker discriminator than class or fare.

# %% [markdown]
# ## Task 4: Statistical Analysis

# %% [markdown]
# ### 4.1 Descriptive Statistics

# %%
train_clean[["Age", "Fare", "SibSp", "Parch"]].describe()

# %% [markdown]
# ### 4.2 Frequency Distribution
#
# Frequency counts for the key categorical variables:

# %%
print("Survival counts:")
print(train_clean["Survived"].value_counts())
print("\nSurvival rate: {:.2f}%".format(train_clean["Survived"].mean() * 100))

print("\nSex counts:")
print(train_clean["Sex"].value_counts())

print("\nPclass counts:")
print(train_clean["Pclass"].value_counts().sort_index())

print("\nEmbarked counts:")
print(train_clean["Embarked"].value_counts())

# %% [markdown]
# ### 4.3 Correlation Analysis

# %%
corr_with_survival = corr_matrix["Survived"].drop("Survived").sort_values(ascending=False)
corr_with_survival

# %% [markdown]
# ### 4.4 Strongest Positive Correlation

# %%
strongest_positive = corr_with_survival.idxmax()
print(f"Strongest positive correlation with Survived: {strongest_positive} "
      f"(r = {corr_with_survival.max():.3f})")

# %% [markdown]
# ### 4.5 Strongest Negative Correlation

# %%
strongest_negative = corr_with_survival.idxmin()
print(f"Strongest negative correlation with Survived: {strongest_negative} "
      f"(r = {corr_with_survival.min():.3f})")

# %% [markdown]
# ### 4.6 Three Important Statistical Findings
#
# 1. **Class and cabin records dominate survival.** `Pclass` shows the
#    strongest negative correlation with survival (r ≈ -0.34), while
#    `Has_Cabin` shows the strongest positive correlation (r ≈ 0.32) —
#    passengers travelling in higher classes, whose cabin numbers were
#    recorded, were considerably more likely to survive, consistent with
#    "women and children first" being applied more effectively in
#    first-class areas with closer access to lifeboats.
# 2. **Fare reinforces the class effect.** `Fare` also correlates
#    positively with survival (r ≈ 0.26), reinforcing that passengers
#    who paid more (largely for higher-class tickets) had better odds
#    of survival — `Pclass`, `Fare`, and `Has_Cabin` are really three
#    different views of the same underlying socio-economic effect.
# 3. **Family size effects are mixed.** `SibSp` and `Parch` show weak
#    correlations individually, suggesting that having a *moderate*
#    number of family members aboard (not zero, not very large) was
#    associated with slightly higher survival — travelling completely
#    alone or in very large families both appear less favourable,
#    a pattern that a simple linear correlation understates.

# %% [markdown]
# ## Task 5: Machine Learning — Predicting Survival

# %% [markdown]
# ### 5.1 Select Predictor Variables
#
# We select `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`,
# and `Has_Cabin` as predictors. Categorical variables (`Sex`, `Embarked`)
# are one-hot encoded. `Name`, `Ticket`, and `PassengerId` are excluded as
# they are identifiers/free text without direct predictive structure in
# this simple model.

# %%
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Has_Cabin"]
X = train_clean[features].copy()
y = train_clean["Survived"].copy()

X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)
print(X.columns.tolist())
X.head()

# %% [markdown]
# ### 5.2 Train/Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# %% [markdown]
# ### 5.3 Train Logistic Regression Classifier
#
# Features are scaled with `StandardScaler` since Logistic Regression is
# sensitive to feature magnitude (e.g. `Fare` ranges up to 500+, while
# dummy variables are 0/1).

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# %% [markdown]
# ### 5.4 Predict on Testing Data

# %%
y_pred = model.predict(X_test_scaled)

# %% [markdown]
# ### 5.5 Model Evaluation

# %%
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did not survive", "Survived"],
            yticklabels=["Did not survive", "Survived"])
plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("figures/07_confusion_matrix.png")
plt.show()
print(cm)

# %%
report = classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"])
print(report)

# %% [markdown]
# ### 5.6 Discussion of Model Performance
#
# The Logistic Regression model achieves an accuracy of roughly
# 80% on the held-out test split (exact value printed above), which is
# competitive for this well-studied dataset with a simple linear model.
# The confusion matrix typically shows the model is better at correctly
# identifying non-survivors than survivors, a common pattern in Titanic
# models reflecting the class imbalance (about 62% did not survive) and
# the inherent randomness of who was rescued. Precision and recall for
# the "Survived" class are usually somewhat lower than for "Did not
# survive", indicating the model occasionally misses true survivors, most
# often ones with atypical profiles (e.g. male passengers who unusually
# survived, or third-class passengers who unusually survived). Overall,
# `Sex` and `Pclass` remain the dominant predictors, consistent with the
# statistical correlations found earlier.

# %% [markdown]
# ## Feature Importance (Model Coefficients)

# %%
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)
coef_df

# %% [markdown]
# **Interpretation:** The largest-magnitude coefficients correspond to
# `Sex_male` (strongly negative — being male sharply reduces predicted
# survival probability) and `Pclass` (negative — higher class number,
# i.e. lower status, reduces survival probability), confirming the
# statistical findings from Task 4.

# %% [markdown]
# ## Summary
#
# This analysis confirms the well-known Titanic survival narrative: sex,
# passenger class, and fare were the dominant factors in survival, far
# outweighing age or family size. The Logistic Regression model, despite
# its simplicity, captures this pattern well and achieves solid predictive
# accuracy. See the accompanying report (`Titanic_Analysis_Report.pdf`)
# for a full discussion, limitations, and recommendations.
