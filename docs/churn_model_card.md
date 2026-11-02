# Churn Model Card

## What this model does
This model predicts whether a customer is likely to leave ShopSmart.It helps the company identify customers who may need special offers to stay.

## Data
The model uses 2,000 synthetic customer records with features like tenure, monthly charges, support tickets, late payments, app engagement and contract type. The label used is churned.

## Model
Logistic Regression was selected because it achieved a higher ROC-AUC score (0.777) than Random Forest (0.758). It uses StandardScaler and max_iter=1000. The chosen decision threshold is 0.4.

## How well it works (test set)
| Metric | Score |
|---|---|
| Accuracy |0.806 |
| Precision |0.626 |
| Recall |0.508 |
| F1 |0.561 |
| ROC-AUC |0.783 |
| Baseline accuracy (always predict "stays") | 0.756 |

## Limits and risks
- The model is trained using synthetic data, not real customer data.
- It may miss some customers who are actually going to leave.
- Predictions may not be accurate for all types of real customers.
- Incorrect predictions may lead to unnecessary retention offers.

## Intended use
The model should be used to identify customers who may leave and help ShopSmart plan retention offers.It should not be used as the only basis for important customer-related decisions.
