# Building a Leakage-Safe DistilBERT Classifier for Bank Transactions

*How I expanded a transaction taxonomy to 18 categories, caught a subtle data-leakage problem, and achieved 99.93% accuracy on a synthetic test set*

Bank transaction descriptions are short, inconsistent, and often filled with merchant codes, locations, abbreviations, and payment-processing text. A person may immediately recognize that “SHELL #12345 BROOKLYN” is a fuel purchase, but a reporting system only sees a small string of unstructured text.

I built an NLP classification project to test whether DistilBERT could automatically assign transaction descriptions to meaningful spending categories. The final model classified transactions across 18 categories, including a new Gas & Fuel category, and reached 99.93% accuracy with a macro F1 score of 0.9994 on a leakage-safe synthetic test set.

The score was impressive, but the most valuable part of the project was not the number. It was discovering that my first data split allowed repeated descriptions to appear across training, validation, and test sets—and rebuilding the evaluation correctly.

## The business problem

Transaction categorization supports budgeting tools, customer insights, merchant analytics, financial reporting, and personalized recommendations. The challenge is that transaction descriptions rarely arrive as clean merchant names.

Examples may look like:

- `MTG PMT PENFED CU`
- `PAYPAL INST XFER BATH & BODY WORKS`
- `COSTCO GAS #8241 PHOENIX`
- `US GOVERNMENT DIR DEP PPD ID: 547185326`

My objective was to build a model that accepted a transaction description and predicted one of the following 18 categories:

Education, Entertainment, Fees, Gas & Fuel, Groceries, Healthcare, Income, Insurance, Mortgage, Personal Care, Rent, Restaurants, Shopping, Subscription, Transfer, Transportation, Travel, and Utilities.

Because this was a portfolio experiment, I used synthetic transaction data. That distinction matters: synthetic data is useful for testing the modeling workflow, but strong performance on synthetic patterns does not prove equivalent performance on live banking data.

## Preparing the text

The original data contained fields such as `[debit]` and `[credit]`. These tokens can make classification artificially easy if certain categories are disproportionately associated with one transaction type. They also did not match the format I wanted the deployed model to receive.

I therefore created a separate `model_description` field that retained the meaningful merchant and transaction text while removing those leading tags. The original description remained available for auditing.

I also checked missing values, duplicates, category distributions, and description lengths before modeling. Since most transaction descriptions were short, I limited tokenization to 64 tokens and used dynamic padding so each training batch was padded only to the length of its longest sequence.

## Adding a new Gas & Fuel category

The initial taxonomy contained 17 categories. Fuel transactions were grouped into Transportation, but I wanted to test whether Gas & Fuel could stand as a useful separate category.

I used a refined regular expression to identify descriptions containing established fuel brands and fuel-specific language, including Shell, Exxon, Mobil, BP, Chevron, Sunoco, Citgo, Marathon, Valero, Phillips 66, Arco, and Costco Gas.

The audit produced:

- 726 candidate transactions
- 700 unique normalized descriptions
- 19 repeated description patterns
- 100% of candidates originally labeled Transportation
- No obvious false matches in the manual review sample

I retained repeated descriptions because two separate purchases can legitimately have the same merchant description. After relabeling the verified records, Gas & Fuel became the 18th category, with 726 examples.

## The most important discovery: data leakage

My first model produced almost perfect validation performance. That was encouraging, but it was also a reason to investigate further.

I compared normalized descriptions across the three datasets and found that:

- 12.56% of validation descriptions also appeared in training
- 12.48% of test descriptions also appeared in training
- 827 unique descriptions overlapped between training and validation
- 821 unique descriptions overlapped between training and test

The split had been stratified by category, but it had been performed at the row level. That meant repeated transaction descriptions could land in different datasets. A model could effectively see the same description during training and later receive it during evaluation.

This is a subtle but important lesson: a stratified split preserves class balance, but it does not automatically prevent group leakage.

I corrected the issue by splitting the unique normalized descriptions first and then assigning every matching transaction to the same dataset. This retained legitimate repeated transactions while ensuring that identical descriptions never crossed dataset boundaries.

I also found one exact description—`RITE AID - PHILADELPHIA`—assigned to both Healthcare and Shopping. Because the description alone could not distinguish the underlying purchase and only two rows were involved, I removed both contradictory records instead of forcing an arbitrary label.

The final leakage-safe dataset contained 44,778 records:

| Dataset | Records | Share | Categories | Gas & Fuel records |
|---|---:|---:|---:|---:|
| Training | 31,380 | 70.08% | 18 | 510 |
| Validation | 6,677 | 14.91% | 18 | 107 |
| Test | 6,721 | 15.01% | 18 | 109 |

Exact normalized-description overlap was zero across all three datasets.

## Fine-tuning DistilBERT

I chose `distilbert-base-uncased` because it provides contextual language understanding with a smaller and faster architecture than full BERT. That balance is useful for short-text classification, where descriptions contain abbreviations and word combinations that a simple keyword system may miss.

The final fine-tuning configuration was:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 64 tokens |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Warmup steps | 589 |
| Best-model metric | Validation macro F1 |
| Precision mode | FP16 on an NVIDIA T4 GPU |

Macro F1 was the model-selection metric because the classes were imbalanced. Unlike accuracy, macro F1 gives every category equal importance. I also tracked separate precision, recall, and F1 scores for Gas & Fuel to ensure that the newly introduced category was actually learned.

Validation macro F1 improved across all three epochs:

- Epoch 1: 0.9953
- Epoch 2: 0.9974
- Epoch 3: 0.9979

Validation loss also decreased each epoch, so there was no evidence of overfitting within the three-epoch run. Epoch 3 was automatically restored as the best checkpoint.

## Final test results

After model selection was complete, I evaluated the best checkpoint once on the untouched leakage-safe test set.

| Metric | Score |
|---|---:|
| Test loss | 0.0039 |
| Accuracy | 0.9993 |
| Macro precision | 0.9994 |
| Macro recall | 0.9994 |
| Macro F1 | 0.9994 |
| Weighted F1 | 0.9993 |
| Gas & Fuel precision | 1.0000 |
| Gas & Fuel recall | 1.0000 |
| Gas & Fuel F1 | 1.0000 |

The model correctly classified 6,716 of 6,721 test transactions. Fifteen of the 18 categories achieved perfect recall, and every Gas & Fuel record was classified correctly.

## What the five errors revealed

The five mistakes were more informative than the overall accuracy:

1. `Payment Thank You - 2ndA`: Transfer → Shopping
2. `Rite Aid #5658 Atlanta`: Shopping → Healthcare
3. `PayPal Inst Xfer Bath & Body Works`: Shopping → Personal Care
4. `CVS - San Diego`: Shopping → Healthcare
5. `CVS #55457 Las Vegas`: Healthcare → Shopping

Most of these are not random failures. They expose ambiguity in the category definitions.

CVS and Rite Aid can represent prescription purchases or general retail shopping. Bath & Body Works can reasonably fit either Shopping or Personal Care. The generic “Payment Thank You” description does not provide enough information to make Transfer obvious.

This suggests that the next improvement may not be a more complex model. It may be a clearer taxonomy, merchant-category-code features, or a business rule that defines how multi-purpose merchants should be treated.

## Testing new descriptions

Before exporting the model, I tested it on several invented descriptions that were not exact records from the dataset:

- `GREEN VALLEY ORGANIC MARKET` → Groceries (99.96%)
- `NORTHSTAR AIRLINES TICKET 4581` → Travel (99.98%)
- `ACME CORP PAYROLL DIRECT DEP` → Income (99.97%)
- `BRIGHTPATH ONLINE COURSE` → Education (99.69%)
- `SUNRISE ENERGY ELECTRIC BILLZIPCO` → Utilities (99.99%)

All five predictions were sensible, including the final description with an unusual suffix. These checks demonstrated that the inference pipeline worked and that the model could use contextual cues rather than relying only on exact memorization.

However, these examples also contained informative words such as “market,” “airlines,” “payroll,” “course,” and “electric bill.” They are encouraging demonstrations, not substitutes for evaluation on real production traffic.

## How this could contribute to financial services

In many financial products, transaction categories arrive incomplete, inconsistent, or too broad for meaningful analysis. A contextual classifier like this could help enrich those records when merchant metadata is missing or when a processor supplies only a short text description.

The immediate value is not simply replacing a manual label. More consistent categorization could support several downstream uses:

- **Customer-facing money management:** clearer spending summaries, budgets, and alerts.
- **Merchant and portfolio analytics:** more reliable reporting by spending category, customer segment, region, and cohort.
- **Personalization:** better inputs for relevant offers, recommendations, and financial insights.
- **Operational efficiency:** fewer uncategorized records and less manual review of obvious cases.
- **Data quality:** a standardized taxonomy that can feed dashboards and reusable analytical models.

The model would be most useful as part of a hybrid decision system rather than as an isolated prediction endpoint. High-confidence predictions could be accepted automatically, while ambiguous transactions—such as CVS, Rite Aid, or generic payment descriptions—could be combined with merchant category codes, transaction metadata, business rules, or human review.

This design would allow an organization to benefit from automation without hiding uncertainty. It would also create a feedback loop: reviewed exceptions could become new labeled examples for monitoring and future retraining.

## What I learned

The biggest lesson was that model performance depends as much on data design and evaluation discipline as on architecture.

**1. Stratification does not prevent every kind of leakage.**  
My first split preserved category proportions, but identical descriptions still appeared across datasets. For repeated text or customer-level data, the grouping unit must be considered explicitly before splitting.

**2. A taxonomy can become the real performance ceiling.**  
The remaining errors were concentrated around merchants that reasonably fit multiple categories. A more complex model cannot resolve a label definition that the available input does not support.

**3. Rules and machine learning can complement each other.**  
Regex was valuable for identifying and auditing candidate Gas & Fuel transactions. DistilBERT was then useful for learning the broader contextual classification task. The strongest production approach would likely combine both.

**4. Accuracy alone would have hidden important information.**  
Macro precision, recall, and F1 made every category visible despite class imbalance. Tracking Gas & Fuel separately also confirmed that the new category was not being ignored by a model dominated by larger classes.

**5. Deployment requires preprocessing consistency.**  
New transactions must receive the same cleaning, tokenizer, maximum sequence length, and label mapping used during training. Saving the tokenizer and model configuration together was therefore part of the modeling workflow, not an afterthought.

**6. High confidence is not the same as certainty.**  
The model was highly confident on several semantically ambiguous errors. In a real application, confidence should be calibrated and paired with monitoring, exception rules, and a review strategy.

Most importantly, I learned to treat this as a business system rather than only an NLP model. The useful question is not simply, “How accurate is DistilBERT?” It is, “Where can the prediction improve a decision, what happens when it is wrong, and how will the system learn from new transactions?”

## What I would do next

Before deploying this system in a real financial product, I would:

1. Evaluate it on manually reviewed, real transaction descriptions from a different time period.
2. Measure performance on completely unseen merchants, not only unseen full descriptions.
3. Add merchant category codes and structured payment metadata where available.
4. Introduce a confidence threshold and route uncertain predictions to human review or an “Other” category.
5. Monitor category drift, new merchants, and changes in processor-generated description formats.
6. Revisit ambiguous taxonomy rules for pharmacies, department stores, and multi-purpose merchants.

## Summary

This project showed that DistilBERT can be highly effective for short transaction-description classification, especially when merchant and payment language contain strong contextual signals.

But the larger lesson was about evaluation discipline. My first split was stratified and looked correct, yet repeated descriptions still leaked across datasets. Auditing overlap, rebuilding the split by unique description, resolving contradictory labels, and reserving the test set until the end made the final result far more credible.

The final leakage-safe model classified 18 categories with 99.93% test accuracy and a macro F1 score of 0.9994 on synthetic data, including perfect performance for the newly added Gas & Fuel category. More importantly, the project demonstrated how better categorization could contribute to customer insights, portfolio reporting, personalization, operational efficiency, and data quality.

The next challenge is not chasing another decimal point—it is validating how well the system performs on messy, ambiguous, real-world transactions and designing a responsible workflow for the cases it cannot confidently resolve.
