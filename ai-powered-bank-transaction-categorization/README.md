# AI-Powered Bank Transaction Categorization with DistilBERT

An NLP portfolio project that fine-tunes DistilBERT to classify short bank-transaction descriptions across 18 spending categories.

> **Important:** This experiment uses synthetic transaction data. The reported results demonstrate the modeling and evaluation workflow and should not be interpreted as verified production performance on live banking data.

## Project overview

Transaction descriptions are short, inconsistent, and often contain merchant codes, payment-processing language, locations, and abbreviations. This project tests whether a contextual language model can convert those unstructured descriptions into useful financial categories.

The model predicts:

`Education` · `Entertainment` · `Fees` · `Gas & Fuel` · `Groceries` · `Healthcare` · `Income` · `Insurance` · `Mortgage` · `Personal Care` · `Rent` · `Restaurants` · `Shopping` · `Subscription` · `Transfer` · `Transportation` · `Travel` · `Utilities`

## Key results

| Metric | Final test result |
|---|---:|
| Records | 44,778 |
| Categories | 18 |
| Test accuracy | 99.93% |
| Macro precision | 0.9994 |
| Macro recall | 0.9994 |
| Macro F1 | 0.9994 |
| Weighted F1 | 0.9993 |
| Gas & Fuel F1 | 1.0000 |
| Correct test predictions | 6,716 / 6,721 |

The final model made five test errors. Most involved ambiguous category boundaries such as CVS or Rite Aid being labeled as either `Healthcare` or `Shopping`.

## The most important modeling decision

The first stratified row-level split produced almost perfect validation results. I audited normalized descriptions across the datasets and found that:

- 12.56% of validation descriptions appeared in training.
- 12.48% of test descriptions appeared in training.
- 827 unique descriptions overlapped between training and validation.
- 821 unique descriptions overlapped between training and test.

The split preserved class proportions, but repeated descriptions still crossed dataset boundaries. I corrected this by splitting **unique normalized descriptions first** and then assigning all matching transactions to the same dataset.

The final leakage-safe split contained:

| Dataset | Records | Share | Categories | Description overlap |
|---|---:|---:|---:|---:|
| Training | 31,380 | 70.08% | 18 | 0 |
| Validation | 6,677 | 14.91% | 18 | 0 |
| Test | 6,721 | 15.01% | 18 | 0 |

## Process

1. **Audited the data** for missing values, repeated descriptions, category balance, and text length.
2. **Created `model_description`** by removing leading `[debit]` and `[credit]` tags while retaining the original text for review.
3. **Expanded the taxonomy** from 17 to 18 categories by identifying 726 verified fuel transactions with a refined brand-and-keyword regex.
4. **Detected cross-split leakage** by comparing normalized descriptions across training, validation, and test datasets.
5. **Removed two contradictory rows** where the exact description `RITE AID - PHILADELPHIA` had both Healthcare and Shopping labels.
6. **Built leakage-safe grouped splits** with zero exact-description overlap.
7. **Fine-tuned DistilBERT** and selected the best checkpoint using validation macro F1.
8. **Evaluated once on the untouched test set** and reviewed all five errors individually.
9. **Tested new descriptions** to validate the inference pipeline before exporting the model and tokenizer.

## Model configuration

| Parameter | Value |
|---|---:|
| Base model | `distilbert-base-uncased` |
| Maximum sequence length | 64 |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Learning rate | `2e-5` |
| Epochs | 3 |
| Weight decay | 0.01 |
| Warmup steps | 589 |
| Best-model metric | Macro F1 |
| Hardware | NVIDIA T4 GPU with FP16 |

Validation macro F1 improved from `0.9953` to `0.9974` to `0.9979` across the three epochs, while validation loss continued to decrease.

## Adding the Gas & Fuel category

Fuel transactions were originally included under Transportation. A refined regex identified established fuel merchants and fuel-specific terms, including Shell, Exxon, Mobil, BP, Chevron, Sunoco, Citgo, Marathon, Valero, Phillips 66, Arco, and Costco Gas.

The category audit found:

- 726 candidate records
- 700 unique normalized descriptions
- 19 repeated description patterns
- No obvious false matches in the manual review sample
- All candidates originally labeled Transportation

After retraining, all 109 Gas & Fuel test transactions were classified correctly.

## Error analysis

The five incorrect test predictions were:

| Description | Actual | Predicted |
|---|---|---|
| Payment Thank You - 2ndA | Transfer | Shopping |
| Rite Aid #5658 Atlanta | Shopping | Healthcare |
| PayPal Inst Xfer Bath & Body Works | Shopping | Personal Care |
| CVS - San Diego | Shopping | Healthcare |
| CVS #55457 Las Vegas | Healthcare | Shopping |

These cases show why taxonomy design can become the performance ceiling. A more complex model cannot resolve distinctions that the available description does not contain.

## Potential industry contribution

A production version of this workflow could help enrich transactions when merchant metadata is incomplete or inconsistent. More reliable categorization could support:

- Customer budgeting and spending insights
- Merchant and portfolio analytics
- Personalized recommendations and offers
- Reduced manual categorization work
- Standardized inputs for dashboards and downstream models

The strongest implementation would be hybrid: accept high-confidence model predictions, enrich them with merchant category codes and structured metadata, and route ambiguous cases to business rules or human review.

## What I learned

- Stratification does not prevent group leakage.
- Taxonomy quality and label consistency can matter more than additional model complexity.
- Regex and contextual NLP can complement each other.
- Macro metrics and class-specific metrics are essential for imbalanced classification.
- Training and deployment preprocessing must remain identical.
- High confidence does not always mean the prediction is conceptually certain.
- A useful ML solution must define how predictions improve decisions and what happens when the model is wrong.

## Limitations and next steps

- Validate on manually reviewed real transaction data.
- Test generalization to completely unseen merchants and later time periods.
- Add merchant category codes and payment metadata.
- Calibrate prediction confidence and define a review threshold.
- Monitor merchant, category, and description-format drift.
- Formalize business rules for multi-purpose merchants.

## Files

- [`README.md`](README.md) — concise GitHub project overview
- [`Building_a_Leakage_Safe_DistilBERT_Transaction_Classifier.md`](Building_a_Leakage_Safe_DistilBERT_Transaction_Classifier.md) — full Medium article with the complete project narrative

