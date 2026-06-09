# Linear Regression Models Comparison

## Project Overview

This project compares different linear regression implementations and regularization techniques on a real-world rental price prediction task. The dataset comes from the Kaggle competition [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data).

The goal is to evaluate how different model types (analytical solution, SGD, Lasso, Ridge, ElasticNet) and preprocessing techniques (scaling, polynomial features, outlier removal) affect regression metrics. Both scikit-learn and custom implementations are compared.

## Dataset

The dataset contains rental listings with features such as:

- `bathrooms`, `bedrooms`
- `price` (target variable)
- `latitude`, `longitude`
- `features` (amenities list)
- `description`, `photos`, etc.

Only a subset of features was used: 20 most common amenities (one-hot encoded) plus bathroom and bedroom counts.

## Project Structure

```
├── linear_models_research.ipynb   # Main Jupyter Notebook
├── modules.py                      # Custom implementations
├── README.md                       # This file
└── data/                           # train.json, test.json (not included)
```

## Methodology

1. **Preprocessing**
   - Extracted 20 most common amenities from the `features` column
   - Applied one-hot encoding for each amenity
   - Used `bathrooms` and `bedrooms` as numerical features

2. **Models Tested**
   - Linear Regression (analytical and SGD)
   - Lasso (L1 regularization)
   - Ridge (L2 regularization)
   - ElasticNet (L1 + L2)

3. **Experiments**
   - Comparison of sklearn vs custom implementations
   - Scaling methods: MinMaxScaler vs StandardScaler
   - Polynomial features (degree 10) with `bathrooms` and `bedrooms`
   - Outlier removal (1st-99th percentile of price)

## Key Findings

### 1. Regularization Impact

L1 and L2 regularization alone showed almost no improvement over vanilla Linear Regression, achieving test R² around 0.02. ElasticNet with StandardScaler performed best (test RMSE = 9609), but improvements remained modest. Without scaling, regularization had virtually no effect.

### 2. Outlier Removal Effect

Removing outliers dramatically improved train R² from near zero to 0.45 for Lasso, but test R² stayed at 0.01-0.02, unchanged from full data. This indicates distribution shift between train and test sets, where high-price properties contain real signal that removal destroyed.

### 3. Custom vs Scikit-learn

The analytical custom Linear Regression matched sklearn perfectly. SGD-based custom models failed without scaling (negative R²), while sklearn handled unscaled data robustly. With scaling, custom models approached but remained slightly behind sklearn, particularly for ElasticNet. Sklearn's coordinate descent proved faster and more stable than custom SGD.

### 4. Polynomial Features

Polynomial features (degree 10) without scaling caused numerical overflow, producing test R² values in the negative billions. This demonstrates why feature scaling is mandatory before polynomial expansion.

## Conclusion

StandardScaler consistently outperformed MinMaxScaler for regularized models. Sklearn's ElasticNet with StandardScaler was the only model that consistently outperformed the mean baseline.

## License

This project is for educational purposes only. The dataset is subject to Kaggle's terms of use.