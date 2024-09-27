# Fraud Detection Model Using XGBoost

This repository contains a machine learning model that detects fraudulent transactions using an XGBoost classifier. The model analyzes transaction data to identify patterns that distinguish fraudulent transactions from genuine ones. The dataset used for this project is a synthetic dataset of financial transactions.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Installation

To run the project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost
```

## Dataset

The dataset used in this project is the `Fraud.csv` file, which contains various attributes of financial transactions. Each row represents a transaction, and the key columns include:

- `type`: Type of transaction (e.g., TRANSFER, CASH_OUT).
- `amount`: The transaction amount.
- `oldbalanceOrig`, `newbalanceOrig`: Balance before and after the transaction for the origin account.
- `oldbalanceDest`, `newbalanceDest`: Balance before and after the transaction for the destination account.
- `isFraud`: Indicator of whether the transaction is fraudulent (1) or not (0).
- `isFlaggedFraud`: Indicator if the transaction was flagged as suspicious by the system.

## Features

Before training the model, several data processing steps were performed, including:

1. **Feature Engineering**:
   - Mapped transaction types to numerical values.
   - Calculated error balance features for both origin and destination accounts.
   - Handled zero and NaN balances.

2. **VIF Calculation**:
   - Checked for multicollinearity between features using Variance Inflation Factor (VIF).

3. **Handling Missing and Infinite Values**:
   - Replaced missing and infinite values and dropped unnecessary columns for model training.

## Model

An XGBoost classifier was used to detect fraudulent transactions. Key model parameters:

- `max_depth=3`: Depth of the trees.
- `scale_pos_weight`: Used to handle class imbalance by adjusting the weight of fraudulent transactions.
- `n_jobs=4`: Parallel processing to speed up model training.

The data was split into training and testing sets, with 80% for training and 20% for testing.

## Results

The model was evaluated using:

- **Average Precision Score (AUPRC)**: Measures the area under the precision-recall curve.
- **Accuracy Score**: Measures the proportion of correctly classified transactions.

Model Performance:
- AUPRC = 0.9822018505137896
- Accuracy = 0.9993819124801884

## Visualization

A 3D scatter plot was created to visualize how error-based features help separate genuine and fraudulent transactions.

![Fraud vs Genuine Transactions](visualization.png)

## License

This project is licensed under the MIT License.
