# Bagging (Bootstrap Aggregation) — Diabetes Prediction

An implementation of **Bagging (Bootstrap Aggregation)** to predict diabetes outcomes, comparing a single Decision Tree against a Bagging ensemble and Random Forest. This project highlights how ensemble techniques reduce variance and improve generalization over a single high-variance model.

## 📌 Overview

A single Decision Tree tends to overfit and has high variance — small changes in training data can produce very different trees. **Bagging** tackles this by training many trees on different bootstrapped (random, with-replacement) samples of the data and aggregating their predictions. This notebook builds that comparison step by step: baseline tree → bagged trees → Random Forest (which is itself an extension of bagging).

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset** (`diabetes.csv`), which contains:
- **768 samples** of female patients of Pima Indian heritage
- **8 features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target (`Outcome`)**: binary — 1 = diabetic, 0 = non-diabetic
- Class distribution: 500 non-diabetic vs. 268 diabetic (moderately imbalanced)

> **Note:** The notebook loads the file as `diabetes.csv`. Make sure the dataset in this repo is named/saved as `diabetes.csv` (or update the `pd.read_csv(...)` path) so it runs without a `FileNotFoundError`.

## 🛠️ Tech Stack

- Python 3
- pandas
- scikit-learn

## 🔍 Workflow

1. **Load the data** — Read the diabetes dataset into a pandas DataFrame.
2. **Explore & clean** — Check for missing values (`isnull().sum()`), review summary statistics (`describe()`), and inspect class balance (`Outcome.value_counts()`).
3. **Prepare features/target** — Split into `x` (features) and `y` (`Outcome`).
4. **Scale features** — Standardize all features using `StandardScaler`.
5. **Train/test split** — Split the scaled data using `train_test_split` with `stratify=y` to preserve class balance across train and test sets.
6. **Baseline model** — Evaluate a plain `DecisionTreeClassifier` using 5-fold cross-validation (`cross_val_score`) to establish a baseline accuracy.
7. **Bagging Classifier** — Train a `BaggingClassifier` (100 decision tree estimators, `max_samples=0.8`, with out-of-bag scoring enabled) and check its OOB score.
8. **Cross-validate the bagged model** — Run 5-fold cross-validation on the `BaggingClassifier` and compare its mean accuracy to the single tree.
9. **Compare with Random Forest** — Run 5-fold cross-validation on a `RandomForestClassifier` to see how it stacks up against manual bagging.

## 📈 Results

The notebook compares mean cross-validation accuracy across three models:
- **Single Decision Tree** — baseline accuracy, expected to be the most variable/overfit-prone
- **Bagging Classifier** (100 bagged decision trees) — improved, more stable accuracy, plus an out-of-bag (OOB) score as an internal validation estimate
- **Random Forest Classifier** — accuracy for comparison, since Random Forest builds on the same bagging idea with added feature randomness

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn jupyter
```

### Running the notebook
```bash
git clone https://github.com/avrajyoti07/20Bagging.git
cd 20Bagging
jupyter notebook 20Bagging.ipynb
```

## 📁 Repository Structure

```
20Bagging/
│
├── 20Bagging.ipynb   # Main notebook: Decision Tree vs Bagging vs Random Forest
├── diabetes.csv       # Pima Indians Diabetes dataset
└── README.md          # Project documentation
```

## 🧠 About Bagging

**Bagging (Bootstrap Aggregating)** trains multiple instances of the same base model (here, Decision Trees) on different random bootstrap samples of the training data, then aggregates their predictions (majority vote for classification). This reduces variance and overfitting compared to a single model. **Random Forest** extends this idea further by also randomizing the subset of features considered at each split, which is why it's often described as "bagging + extra randomness."

## 📄 License

This project is open source and available for learning purposes.

## 👤 Author

**avrajyoti07**
