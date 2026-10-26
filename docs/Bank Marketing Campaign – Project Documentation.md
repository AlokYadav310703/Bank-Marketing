# Bank Marketing Campaign: Project Documentation

Data analyst portfolio project | Dataset: UCI Bank Marketing (bank-full.csv) | Notebooks: 01\_eda.ipynb, 02\_modeling\_and\_training.ipynb

## 1. Project Overview

A Portuguese bank ran direct phone-call campaigns to sell term deposits. This project answers three questions:

1. Which customer segments are most likely to subscribe?
2. When should the bank call, and how many times?
3. Can a model rank customers so that agents call the most promising ones first?

The work is split into two notebooks: **01\_eda** (exploration and statistical testing) and **02\_modeling\_and\_training** (preprocessing, models, evaluation and business translation).

## 2. Dataset

| Item | Detail |
| --- | --- |
| Source | UCI Bank Marketing dataset |
| Size | 45,211 rows, 17 original columns |
| Target | `y` (subscribed to a term deposit: yes / no) |
| Baseline conversion | 11.70% (5,289 yes, 39,922 no) |
| Class imbalance | 7.5 : 1 (No : Yes) |

**Feature groups**

- **Customer:** age, job, marital, education, default, balance, housing, loan
- **Contact:** contact, day, month, duration
- **Campaign history:** campaign, pdays, previous, poutcome

**Special columns**

- `duration` is only known after the call ends, so it is a data leakage risk.
- `pdays = -1` is a code meaning "never contacted before", not a number of days.
- `month` has no year attached.

## 3. Data Quality Audit

| Check | Result | Decision |
| --- | --- | --- |
| Null values | None | No action needed |
| "unknown" placeholders | poutcome 81.75%, contact 28.80%, education 4.11%, job 0.64% | Keep "unknown" as its own category (dropping rows would lose 82% of the data) |
| pdays = -1 | 36,954 rows (81.7%); consistent with `previous` | Create flag `was_previously_contacted` |
| Negative balance | 3,766 clients (8.3%) | Valid, kept |
| Extreme balance | Max 102,127 EUR vs median 448 EUR | Real customers, kept |
| duration = 0 | 3 calls | Harmless, kept |
| previous max | 275 (99th percentile is about 9) | Likely a data-entry error; handled by method, not deletion |

No rows were removed. Outliers are handled through rank-based tests, capping and log transforms.

## 4. Exploratory Data Analysis

### 4.1 Target distribution

- About 1 in 9 calls succeeds (baseline 11.70%, 95% CI 11.41% to 12.00%).
- A model that always predicts "no" would be 88.3% accurate while finding zero buyers, so accuracy is not used. ROC-AUC, PR-AUC and F1 on the "yes" class are used instead.

### 4.2 Numerical features

| Feature | Key observation |
| --- | --- |
| age | Mean 40.9, median 39, range 18 to 95. Moderately right-skewed (0.69). Not normal (Shapiro-Wilk p about 2e-34). 487 clients (1.1%) above 70 are real values |
| balance | Extremely right-skewed (8.4). Mean 1,362 EUR vs median 448 EUR. Log transform makes it near-symmetric |
| duration | Excluded from modelling because it is only known after the call |
| campaign | Median 2, max 63. 9.6% contacted more than 5 times, 2.6% more than 10 times. Capped at 10 for analysis |

### 4.3 Categorical features

- Management, blue-collar and technician make up most clients; students (938) and housemaids (1,240) are small groups with wide confidence intervals.
- 60% of clients are married; 51% have secondary education.
- Only 815 clients (1.8%) have credit in default, so the effect of this column is small.

### 4.4 Feature vs target (hypothesis tests)

Each categorical feature was tested with a chi-square test of independence (H0: feature and target are independent; reject if p < 0.05), with Cramér's V as the effect size. Numeric features were tested with Mann-Whitney U. All tests listed below were significant; the effect size shows how much each one matters.

| Feature | Conversion pattern | Cramér's V |
| --- | --- | --- |
| poutcome | success 64.7%, other 16.7%, failure 12.6%, unknown 9.2% | 0.31 (largest) |
| month | Mar 52.0%, Dec 46.7%, Sep 46.5%, Oct 43.8%, May 6.7% | 0.26 |
| age band | 18-25: 24.0%, 26-35: 12.0%, 36-45: 9.4%, 46-55: 9.4%, 56-65: 14.1%, 66+: 42.6% (U-shape) | 0.15 |
| contact | cellular 14.9%, telephone 13.4%, unknown 4.1% | 0.15 |
| job | student 28.7%, retired 22.8%, blue-collar 7.3%, entrepreneur 8.3% | 0.14 |
| housing loan | no 16.7%, yes 7.7% | 0.14 |
| pdays band (previously contacted only) | 1-90 days 35.0%, 91-180: 27.4%, 181-365: 17.8%, 366+: 29.2% | 0.14 |
| balance (quartiles) | Q1 7.2% rising to Q4 16.2% | 0.10 |
| campaign (capped at 10) | 1 call 14.6% falling to 4.2% at 10+ | 0.09 |
| education | primary 8.6%, secondary 10.6%, tertiary 15.0% | 0.07 |
| marital | single 14.9%, divorced 11.9%, married 10.1% | 0.07 |
| personal loan | no 12.7%, yes 6.7% | 0.07 |
| default | no 11.8%, yes 6.4% | 0.02 (negligible) |

For balance, the median of subscribers is 733 EUR vs 417 EUR for non-subscribers (Mann-Whitney p about 7e-101, rank-biserial r = 0.18).

### 4.4.1 Points that need careful reading

- **Month:** the best four months have very few calls (214 to 738 each, 4.4% of all calls combined). The bank probably called a hand-picked audience in those months, so a 52% rate in March will not hold if volume is scaled up. May has the most calls (13,766, 30% of the total) and the lowest rate.
- **Contact type:** the difference comes almost entirely from "unknown" (concentrated in June and May). Cellular vs telephone is only 1.5 points apart (p = 0.030, not significant after Bonferroni correction).
- **Campaign calls:** high call counts are filled with "no" clients because "yes" clients stop being called. This does not prove extra calls reduce interest.
- **pdays:** the pattern is not a straight line (clients last contacted over a year ago convert better than those contacted 6-12 months ago), so no simple "call within 90 days" rule is drawn from it.
- **Job and marital vs age:** student and retired sit at the two ends of the age U-shape, and singles are younger, so part of these effects is likely an age effect.

### 4.5 Multivariate analysis

- **Spearman correlation with target:** duration +0.34 (leakage), previous +0.17, pdays +0.15, balance +0.10, campaign -0.08, day -0.03, age -0.01 (not significant). All are weak apart from duration.
- `pdays` and `previous` are almost identical (Spearman about 0.99), so both should not be used together in a linear model.
- **Job x education:** the best groups (at least 100 clients) are students with secondary education (29.7%), retired with tertiary (27.6%) and students with tertiary (26.5%). Education matters inside most jobs (for example blue-collar 5.8% to 16.1% from primary to tertiary).
- **Age vs balance scatter:** no simple two-variable rule separates yes from no, which is why a multivariate model was built.

## 5. Preprocessing & Feature Engineering

Based on the EDA:

| Step | Reason |
| --- | --- |
| Drop `duration` | Data leakage |
| Create `was_previously_contacted` and drop raw `pdays` | Fix the -1 sentinel |
| Create `age_band` | Capture the U-shaped age effect |
| Log transform `balance` (signed log1p) | Reduce extreme skew |
| Encode target as 0/1 | Modelling |
| Keep "unknown" as a category | Informative; avoids losing rows |
| Scale numeric features (StandardScaler) | Fitted on training data only |
| One-hot encode categorical and binary features | Fitted on training data only |

**Final feature set (16 features):** 5 numeric (age, balance, day, campaign, previous), 1 binary (was\_previously\_contacted) and 10 categorical (job, marital, education, default, housing, loan, contact, month, poutcome, age\_band).

**Train/test split:** 80/20, stratified on the target, random seed 42. Train: 36,168 rows. Test: 9,043 rows. Positive rate is 11.7% in both.

**Class imbalance:** handled with class weights (`class_weight='balanced'`, `scale_pos_weight` for XGBoost, `auto_class_weights='Balanced'` for CatBoost) rather than synthetic oversampling.

## 6. Modelling & Evaluation

Five models were trained on the same pipeline and evaluated on the test set at the default 0.5 threshold.

| Model | ROC-AUC | PR-AUC | F1 (yes) | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.7751 | 0.4140 | 0.3865 | 0.28 | 0.64 |
| Random Forest | 0.7966 | 0.4343 | 0.4426 | 0.34 | 0.62 |
| XGBoost | 0.8026 | 0.4549 | 0.4553 | 0.3533 | 0.6399 |
| LightGBM | 0.8038 | 0.4581 | 0.4547 | 0.3538 | 0.6361 |
| CatBoost | 0.8062 | 0.4563 | 0.4481 | 0.3429 | 0.6465 |

**Key points**

- All tree-based models beat the Logistic Regression baseline, which suggests real non-linear interactions in the data.
- The three boosting models are very close. CatBoost has the highest ROC-AUC (0.8062); LightGBM has the highest PR-AUC (0.4581); XGBoost has the highest F1.
- Random Forest feature importance was plotted to check that the model agrees with the EDA (poutcome, month, age, contact).

## 7. Business Translation: Lift and Gains

Customers in the test set were ranked by predicted score (XGBoost) and split into 10 equal groups.

| Decile | Customers | Converted | Conversion rate | Lift |
| --- | --- | --- | --- | --- |
| 1 (top 10%) | 905 | 468 | 51.7% | 4.42x |
| 2 | 904 | 191 | 21.1% | 1.81x |
| 3 | 904 | 95 | 10.5% | 0.90x |
| 4 | 904 | 75 | 8.3% | 0.71x |
| 5 | 905 | 51 | 5.6% | 0.48x |
| 6 | 904 | 41 | 4.5% | 0.39x |
| 7 | 904 | 43 | 4.8% | 0.41x |
| 8 | 904 | 50 | 5.5% | 0.47x |
| 9 | 904 | 23 | 2.5% | 0.22x |
| 10 | 905 | 21 | 2.3% | 0.20x |

- The top decile converts at 51.7%, which is 4.4 times the 11.7% base rate.
- Deciles 1 to 3 contain 754 of the 1,058 subscribers in the test set (about 71%), so calling the top 30% of the list captures roughly 7 in 10 buyers.
- From decile 3 onward, conversion is at or below average.

## 8. Threshold Tuning

A missed buyer (false negative) is considered costlier than a wasted call (false positive), so the 0.5 cutoff was reviewed.

| Option | Threshold | Recall | Precision | Calls made (of 9,043) | TP / FP / FN |
| --- | --- | --- | --- | --- | --- |
| Default | 0.500 | 64.0% | 35.3% | 1,916 | 677 / 1,239 / 381 |
| Recall target of 0.66 (chosen) | 0.475 | 66.8% | 32.6% | 2,167 | 707 / 1,460 / 351 |
| Cost-based (FN costs 20x FP) | 0.200 | 94.9% | 14.2% | 7,071 | 1,004 / 6,067 / 54 |

- Moving from 0.5 to 0.475 gains 30 extra subscribers for 221 extra calls, a modest trade.
- The 20:1 cost ratio is an assumption. At that ratio the model says "call almost everyone" (78% of the list), so the ratio must be replaced with real profit per conversion and cost per call before use.

## 9. Key Findings and Recommendations

| # | Finding | Recommended action |
| --- | --- | --- |
| 1 | Previous campaign success converts at 64.7% (about 5.5x baseline) | Prioritise these clients first; the segment is small (3.3% of data) |
| 2 | Clients aged 66+ (42.6%) and 18-25 (24.0%), students and retirees convert best | Include in priority targeting |
| 3 | Clients with no housing loan convert at 16.7% vs 7.7% | Prioritise clients without loans; adjust messaging for those with loans |
| 4 | Higher balance goes with higher conversion (7.2% to 16.2% across quartiles) | Use balance as a ranking signal |
| 5 | Conversion drops as calls increase (14.6% at 1 call, below 8% after 5) | Set a contact cap of about 3 to 5 calls per client |
| 6 | May carries 30% of calls with the lowest conversion (6.7%) | Reduce May volume; test whether Mar, Sep, Oct, Dec results hold at larger scale |
| 7 | Unknown contact type converts at only 4.1% | Improve contact data capture; prefer cellular |
| 8 | Model ranks customers well (4.4x lift in top decile) | Use score-based call lists instead of random calling |
| 9 | `duration` is leakage | Never use it as a predictor |

## 10. Limitations and Next Steps

**Limitations**

- The dataset has no year, so time trends and seasonality cannot be separated.
- The bank may have called selected clients, so 11.7% may not equal the conversion rate of the whole customer base.
- Associations are not causal (for example call count vs conversion).
- Model performance is measured on a single train/test split with no cross-validation and no hyperparameter tuning.
- Threshold tables and the cost analysis were calculated on the test set, so the chosen threshold is slightly optimistic. A separate validation set would be cleaner.
- Model comparison charts, the confusion matrix and the lift analysis use XGBoost, because LightGBM and CatBoost were not added to the comparison dictionary. The selection "best by PR-AUC" is therefore only among three models.
- The CatBoost PR-AUC is 0.4563 in the notebook output, but the written note says 0.4617. This document uses the printed output (0.4563).
- The EDA summary suggests a 3-call cap, while the detailed section suggests 3 to 5. This document uses 3 to 5.

**Next steps**

- Add LightGBM and CatBoost to the ROC/PR curves and re-select the best model.
- Use cross-validation and hyperparameter tuning; keep a separate validation set for threshold selection.
- Add SHAP values to explain individual predictions.
- Replace the 20:1 cost ratio with real business economics.
- Try dropping `previous` or `pdays` for the linear model to avoid multicollinearity.
- Build a dashboard summarising segments and the lift chart for business users.

## Appendix: Tools and Reproducibility

- **Language and libraries:** Python, pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn, XGBoost, LightGBM, CatBoost
- **Statistical methods:** chi-square test, Cramér's V, Mann-Whitney U, Welch t-test, two-proportion z-test, Wilson confidence intervals, Spearman correlation, Shapiro-Wilk
- **Random seed:** 42 for all splits and models
- **Data path used in notebooks:** `../data/bank-full.csv` (semicolon separated)
