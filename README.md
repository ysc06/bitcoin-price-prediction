# Bitcoin Price Prediction
**CS598: Practical Statistical Learning**  
**Author:** Naomi (Yu-Shan) Cheng  

---

## Overview
This project explores how well different regression models can predict **Bitcoin’s daily market price** using blockchain activity data. The goal was to test how traditional statistical learning methods — **Linear Regression**, **Ridge Regression**, and **Lasso Regression** — perform on a highly volatile financial time series.

The dataset comes from a Kaggle competition and Coursera resources. Each observation represents one day of Bitcoin’s network activity. Because the data are sequential, I kept the **time order intact** during train/test splitting to preserve the temporal dependencies.

---

## Project Goals
- Clean and prepare real-world blockchain data for analysis.  
- Build predictive models with Linear, Ridge, and Lasso regression.  
- Compare their performance using **Root Mean Squared Error (RMSE)**.  
- Evaluate how the number of lagged predictors (3-day vs. 7-day history) affects results.

---

## Data and Feature Engineering

### Dataset
- **Total records:** 2,920 daily entries  
- **Columns:** 23 variables + date  
- **Target:** `btc_market_price`

To keep the model interpretable and avoid overfitting, only the most relevant blockchain features were used:

| Feature | Why It Matters |
|----------|----------------|
| `btc_trade_volume` | Indicates market liquidity — higher volume often means higher price movement. |
| `btc_n_transactions` | Reflects daily transaction counts and overall demand. |
| `btc_estimated_transaction_volume_usd` | Shows the USD value of total transactions, a proxy for economic activity. |
| `btc_output_volume` | Captures how much Bitcoin is being moved across the network. |

### Lagged Features
Since Bitcoin’s price depends on past behavior, **lagged versions** of each feature were created:  
- **L = 3:** previous 3 days → 12 lagged predictors  
- **L = 7:** previous 7 days → 28 lagged predictors  

This allowed the model to learn short- and medium-term temporal trends.

### Handling Missing Data
Creating lagged features left the first few rows empty (21 rows for L=3, 28 for L=7).  
Used **forward-fill imputation** to fill missing values and then dropped the leading rows to align predictors with their targets.

### Train/Test Split
Data were split **chronologically (80% train / 20% test)** to maintain time order:  
- **L=3:** 2,333 training rows, 584 testing rows  
- **L=7:** 2,330 training rows, 583 testing rows  

---

## Methods

### Linear Regression
Used as the baseline model.  
**Sequential Forward Selection (SFS)** was applied to identify lagged features that minimized mean squared error.

### Ridge Regression
Added **L2 regularization** using `RidgeCV` to handle multicollinearity.  
Predictors were standardized with `StandardScaler` before fitting.

### Lasso Regression
Introduced **L1 regularization** using `LassoCV` with 5-fold cross-validation.  
This method performs embedded feature selection but risked underfitting for this dataset.

### Baseline Model
A constant mean predictor (training-set average) served as a baseline benchmark.  
All regression models were expected to outperform this.

---

## Evaluation Metric
The models were evaluated using **Root Mean Squared Error (RMSE)** — a sensitive metric for large prediction errors, ideal for volatile data like Bitcoin prices.

---

## Results

| Model | L=3 RMSE | L=7 RMSE | Notes |
|--------|-----------|-----------|-------|
| **Linear Regression** | **1384.6** | **978.4** | Best overall; OLS captured the signal effectively. |
| **Ridge Regression** | 1443.4 | 1150.7 | α=100 (L=3), α=1000 (L=7); regularization didn’t improve performance. |
| **Lasso Regression** | 2227.7 | 2191.4 | Over-penalized and underfit the data. |
| **Baseline (Mean)** | ~5570 | ~5570 | Couldn’t capture volatility at all. |

---

## Discussion
The results were surprising: **simple Linear Regression outperformed Ridge and Lasso**, even with the risk of multicollinearity.  
Ridge added stability but slightly worsened accuracy, while Lasso’s strong regularization removed too many useful signals.  
Adding more lagged features (from 3 to 7 days) improved results only modestly, suggesting **diminishing returns** from highly correlated predictors.

---

## Conclusion
This project showed that when forecasting Bitcoin prices, **simpler models sometimes win**.  
Despite its volatility, Bitcoin’s short-term trends were well captured by an ordinary least-squares regression using a small set of meaningful lagged features. Regularization helped less than expected — a good reminder that model assumptions should always be tested against the data.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/bitcoin-price-prediction.git
   cd bitcoin-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main analysis script:
   ```bash
   python main.py
   ```

4. View the results and plots in the output folder.

---

## Dependencies
- Python 3.9+  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- mlxtend

---

## License
This project is for educational purposes as part of **CS598: Practical Statistical Learning** at the University of Illinois Urbana-Champaign.
