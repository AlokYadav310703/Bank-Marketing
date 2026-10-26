# Bank Marketing Campaign: Who Subscribes to a Term Deposit?

Exploratory analysis and predictive modelling on the UCI Bank Marketing dataset (45,211 phone calls made by a Portuguese bank), built to help a bank decide **who to call, when to call, and how many times**.

## Business Problem

Only about 1 in 9 calls (11.7%) ends in a term deposit subscription. The goal is to find which customer segments convert, how timing and contact history affect results, and whether a model can rank customers so agents call the most promising ones first.

## Key Results

- **Previous campaign success** converts at **64.7%**, about 5.5x the 11.7% baseline.
- **Age is U-shaped:** clients aged 66+ convert at 42.6% and ages 18-25 at 24.0%, while ages 36-55 sit below baseline (9.4%).
- **Clients without a housing loan** convert at 16.7% vs 7.7% for those with one.
- **Conversion falls as calls increase:** 14.6% on the first call, below 8% after five calls. A cap of 3 to 5 contacts is suggested.
- **May has 30% of all calls but the lowest conversion (6.7%).**
- **Best model:** boosted tree models reach a ROC-AUC of about 0.80 (baseline Logistic Regression: 0.775).
- **Lift:** the top 10% of customers ranked by the model convert at **51.7% (4.4x lift)**, and the top 30% contain about 71% of all subscribers.

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── data/
│   └── bank-full.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_and_training.ipynb
├── images/
└── docs/
    └── project_documentation.md
```

## Approach

1. **Data quality audit:** checked nulls, "unknown" placeholders, the `pdays = -1` sentinel and outliers. No rows were removed.
2. **EDA and hypothesis testing:** chi-square tests with Cramer's V, Mann-Whitney U, two-proportion z-tests, Wilson confidence intervals and Spearman correlation.
3. **Feature engineering:** dropped `duration` (data leakage), created `was_previously_contacted` and `age_band`, applied a log transform to `balance`.
4. **Modelling:** Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost, with class weights to handle the 7.5 : 1 imbalance and a stratified 80/20 split.
5. **Evaluation:** ROC-AUC, PR-AUC and F1 on the "yes" class (accuracy is misleading at 88% "no").
6. **Business translation:** decile lift and cumulative gains charts, plus threshold tuning (recall-targeted and cost-based).

## Model Comparison (test set, threshold 0.5)

| Model | ROC-AUC | PR-AUC | F1 (yes) |
|---|---|---|---|
| Logistic Regression | 0.7751 | 0.4140 | 0.3865 |
| Random Forest | 0.7966 | 0.4343 | 0.4426 |
| XGBoost | 0.8026 | 0.4549 | 0.4553 |
| LightGBM | 0.8038 | 0.4581 | 0.4547 |
| CatBoost | 0.8062 | 0.4563 | 0.4481 |

## Limitations

- The dataset has no year, so seasonality cannot be separated from time trends.
- Results are associations, not causal effects.
- The cost ratio used for threshold selection (20:1) is an assumption.

## How to Run

```bash
git clone <https://github.com/AlokYadav310703/Bank-Marketing>
cd <Bank-Marketing>
pip install -r requirements.txt
jupyter notebook
```

Download `bank-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) into the `data/` folder, then run the notebooks in order (`01_eda`, then `02_modeling_and_training`).

## Tools

Python, pandas, NumPy, matplotlib, seaborn, SciPy, statsmodels, scikit-learn, XGBoost, LightGBM, CatBoost.

## Full Documentation

See [docs/project_documentation.md](docs/project_documentation.md) for detailed findings, tables and recommendations.

