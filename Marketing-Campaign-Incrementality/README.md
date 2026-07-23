## Business Question

Did the Men's E-Mail or Women's E-Mail campaign increase customer visits, conversions, and spending compared with sending no email? Which campaign and customer segments should be prioritized?

## Dataset

The analysis uses Kevin Hillstrom's MineThatData E-Mail Analytics dataset with 64,000 customers randomly assigned to one of three groups:

- Men's E-Mail
- Women's E-Mail
- No E-Mail control

The dataset includes pre-campaign customer characteristics and post-campaign outcomes such as website visits, conversions, and spending.

## Project Workflow

1. Data-quality checks and funnel analysis
2. Exploratory campaign comparison
3. Randomization and balance validation
4. Statistical hypothesis tests and confidence intervals
5. Incremental conversions and revenue estimation
6. Customer-segment analysis
7. Conversion prediction with Logistic Regression, Random Forest, and XGBoost
8. Hyperparameter and threshold analysis
9. Exploratory two-model uplift analysis

## Key Experiment Results

### Men's E-Mail vs No E-Mail

- Visit-rate lift: **+7.659 percentage points**
- 95% confidence interval for visit lift: **6.995 to 8.323 percentage points**
- Conversion-rate lift: **+0.681 percentage points**
- 95% confidence interval for conversion lift: **0.500 to 0.861 percentage points**
- Relative conversion lift: **118.84%**
- Estimated incremental conversions: **145**
- Estimated incremental revenue: **$16,402.71**

### Women's E-Mail vs No E-Mail

- Conversion-rate lift: **+0.311 percentage points**
- 95% confidence interval: **0.150 to 0.472 percentage points**
- Relative conversion lift: **54.33%**
- Estimated incremental conversions: **66.5**
- Estimated incremental revenue: **$9,076.90**

The Men's E-Mail campaign significantly outperformed the Women's E-Mail campaign on conversion rate and average spending per customer.

## Segment Findings

- **Multichannel customers** produced the largest conversion lift for both campaigns.
- The Men's E-Mail campaign increased conversion by **1.02 percentage points** among Multichannel customers.
- New customers had a lower baseline conversion rate but showed stronger incremental response to email outreach.

## Machine-Learning Results

The conversion target was highly imbalanced: only 578 of 64,000 customers converted.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6449 | 0.0181 |
| XGBoost | 0.5778 | 0.0128 |
| Random Forest | 0.4607 | 0.0087 |

A tuned class-weighted Logistic Regression achieved:

- Recall: **57.8%**
- Precision: **1.3%**
- ROC-AUC: **0.6420**
- PR-AUC: **0.0183**

The model was more useful for ranking broad customer groups than for making hard individual conversion predictions.

## Uplift-Modeling Conclusion

The exploratory two-model uplift approach did not reliably rank customers by incremental campaign response. Observed uplift was not monotonic across predicted-uplift deciles, so the customer-level uplift scores should not be used for targeting decisions.

This limitation is reported intentionally: the randomized experiment provides credible campaign-level causal evidence, while the available features and low number of conversions do not support reliable individual-level treatment-effect estimation.

## Business Recommendation

Prioritize the **Men's E-Mail campaign**, particularly for **Multichannel** and **new customers**, while continuing to evaluate targeting strategies through randomized experiments.

Do not operationalize the exploratory uplift scores without richer behavioral data and additional validation.

## Repository Structure

```text
Marketing-Campaign-Incrementality/
├── README.md
├── requirements.txt
├── data/
│   └── Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
└── notebooks/
    └── Marketing_Campaign_Incrementality_and_Customer_Targeting.ipynb
