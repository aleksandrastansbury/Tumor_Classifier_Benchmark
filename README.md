# Charlie's (Algorithm) Angels: Breast Cancer Classification

**CSCE 476/876 Introduction to Artificial Intelligence, Final Project**  
University of Nebraska-Lincoln, Fall 2026

- **Instructor:** Prof. Leen-Kiat Soh
- **Team:** Yin Po Po Aung, Ella Goodrich, Eleanor Krause, Aleksandra Stansbury

## What this project does

We compare machine learning classifiers on the Wisconsin Diagnostic Breast Cancer dataset (569 tumors, 30 measurements each) to reduce missed malignant tumors (false negatives).

**Models:** KNN (Euclidean, Manhattan, and Chebyshev distance), decision trees (ID3, C4.5, CART), SVM, and logistic regression.

## Hypotheses and experiments

| Experiment | Tests | Models |
|---|---|---|
| 1. Feature count | H1: all 30 measurements beat the top 5 | KNN (3 distances), CART, SVM, logistic regression |
| 2. KNN distances | H2: Chebyshev has lower malignant recall | KNN (3 distances) |
| 3. Tree designs | H3: unpruned C4.5 and CART do at least as well as unpruned ID3 | ID3, C4.5, CART |
| 4. Probability cutoffs | Additional analysis | KNN (3 distances), SVM, logistic regression |

All experiments use an 80/20 train/test split, 10-fold cross-validation on the training data only, and 10 random seeds (0 to 9). Metrics: accuracy, precision, recall, F1, type I errors (false positives), type II errors (false negatives), and the confusion matrix. Malignant recall is our main metric.

## Setup

```bash
pip install scikit-learn numpy scipy pandas matplotlib
```

Requirements: scikit-learn, numpy, scipy, pandas, matplotlib.

## How to run

```bash
python data.py                 # checks the data: (569, 30) malignant: 212 benign: 357
python model.py                # quick test of the models on one split
python run_experiments.py      # runs all experiments, saves CSVs in results/
python run_experiments.py 2    # runs only Experiment 2
python analysis.py             # tables, t-tests, and charts from the CSVs
```

Run the experiments before the analysis, because `analysis.py` reads the CSV files in `results/`. The `results/` folder is not stored in Git, so it is created on your computer when you run the experiments.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads the data, flips the labels so malignant = 1, and makes the 80/20 splits |
| `metrics.py` | Accuracy, precision, recall, F1, type I and II errors, confusion matrix |
| `model.py` | Builds and tunes the models: KNN and CART, SVM and logistic regression |
| `id3.py` | Adapted implementation of the ID3 decision tree |
| `c45.py` | Adapted implementation of the C4.5 decision tree |
| `run_experiments.py` | Runs the experiments and saves one CSV each |
| `analysis.py` | Tables, paired t-tests, and charts |
| `results/` | CSVs and charts made by the experiments (not in Git) |

## How the code works

- Every model function in `model.py` returns `(pipeline, grid)`. Scaling and ANOVA F-score feature selection are included in the model pipelines. Feature selection is applied when selecting the top 5, 10, or 15 features; all 30 features are used in the 30-feature setting. These steps are fitted on training data only.
- `tune()` runs 10-fold cross-validation on the training data and picks the best settings by F1.
- Seeds fix every split, so results repeat exactly with the same code and package versions.

## Team rules

- Labels are flipped once, in `data.py` (scikit-learn has 0 = malignant). Never flip them again.
- Always load data with `load_data()` and splits with `get_split(seed)`.
- Score every model with `model_eval` from `metrics.py`.
- Test your code with `N_SEEDS = 2` in `data.py` before a full run, then set it back to 10.

## Credits

- Data: scikit-learn (`load_breast_cancer`, [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html))
- ID3: Adapted from [Decision-Tree-ID3-Algorithm](https://github.com/vidhikhatwani/Decision-Tree-ID3-Algorithm)
- C4.5: Adapted from [C4.5](https://github.com/Valdecy/C4.5)