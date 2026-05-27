# Amazon Review Text as Marketing Intelligence

## Overview

This project investigates whether online review text can act as an early-warning marketing intelligence signal for future product reputation decline. The analysis uses public Amazon Reviews 2023 data from McAuley Lab and focuses on the All_Beauty category.

The project is designed as the empirical foundation for a peer-reviewed marketing analytics paper targeting an ABDC A-rated outlet such as Marketing Intelligence & Planning.

## Research Question

Do emerging patterns in online review text provide earlier warning signals of future product reputation decline than star ratings alone?

## Data

Source: McAuley Lab Amazon Reviews 2023 public dataset.

Category used: All_Beauty.

The raw and processed review-level files are not committed because they are large and reproducible from the download script.

Current audit from the full All_Beauty review file:

- Reviews: 700,808
- Parent products: 112,496
- Unique users: 631,352
- Date range: 2000-11-01 to 2023-09-09
- Average rating: 3.96
- 1- or 2-star review share: 20.7%
- Verified purchase share: 90.5%
- Average review length: 32.78 words

## Method

1. Download public Amazon review data.
2. Clean review text and create a review-level dataset.
3. Aggregate reviews into a product-month panel.
4. Create text intelligence features:
   - Lexicon sentiment
   - Negative text share
   - Complaint-language share
   - Text-rating mismatch
5. Create interpretable complaint-topic features:
   - Skin reaction
   - Scent/fragrance
   - Packaging/leakage
   - Authenticity/counterfeit
   - Effectiveness/quality
   - Value/price
   - Delivery condition
   - Texture/usability
6. Predict future three-month rating decline using rating-only, text, and topic-enhanced models.

## Initial Results

The first baseline models use a time-based test holdout beginning January 2021.

| Model | Features | Test R2 | RMSE | MAE |
|---|---:|---:|---:|---:|
| Rating controls only | 7 | 0.39570 | 1.27825 | 1.02357 |
| Rating controls + text | 14 | 0.39605 | 1.27788 | 1.02304 |
| Rating controls + text topics | 23 | 0.39783 | 1.27599 | 1.02070 |

The topic-enhanced model improves predictive performance, suggesting that interpretable review topics add signal beyond conventional rating metrics. This is promising for a marketing intelligence framing because the topics are actionable for product planning and customer experience monitoring.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the workflow from this folder:

```bash
python scripts/get_amazon_sample.py --sample-size 800000 --output data/processed/all_beauty_reviews_full.csv
python scripts/data_audit.py --input data/processed/all_beauty_reviews_full.csv
python scripts/build_product_month_panel.py --input data/processed/all_beauty_reviews_full.csv --output data/processed/product_month_panel_full.csv
python scripts/add_text_intelligence_features.py
python scripts/add_complaint_topics.py
python scripts/baseline_prediction_models.py
```

## Portfolio Value

This project demonstrates applied marketing analytics, natural language processing, panel-data construction, reproducible data pipelines, and research-oriented empirical design.

