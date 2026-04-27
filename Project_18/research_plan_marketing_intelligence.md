# Research Plan: Amazon Review Text as Marketing Intelligence

## Target Journal

Primary target: Marketing Intelligence & Planning.

ABDC 2025 rating: A.

Journal fit: The paper should be framed around marketing intelligence, planning, and managerial decision-making, not only machine learning performance.

## Working Title

Turning Online Reviews into Marketing Intelligence: Text-Based Early Warning Signals for Product Reputation Management

## Core Research Question

Do emerging patterns in online review text provide earlier warning signals of future product reputation decline than star ratings alone?

## Why This Is Publishable

Managers often monitor average ratings, but ratings are compressed, slow-moving, and backward-looking. Review text may reveal specific emerging problems before they become visible in aggregate ratings. This creates a marketing intelligence contribution: converting unstructured customer voice into actionable signals for product planning, customer experience management, and brand monitoring.

## Data Source

Use Amazon Reviews 2023 from McAuley Lab.

Recommended first category: All_Beauty.

Reason: It is large enough for serious analysis but small enough to manage locally. It includes review text, rating, helpful votes, timestamps, verified-purchase flags, product IDs, and product metadata.

## Unit Of Analysis

Main analysis level: product-month.

Raw unit: individual review.

Aggregate each product's reviews by month, then create lagged predictors from review text to predict future outcomes.

## Dependent Variables

1. Future average rating decline.
2. Future share of low-star reviews.
3. Future review volume change.
4. Future helpful votes for negative reviews.

## Main Predictors

1. Average text sentiment.
2. Negative emotion intensity.
3. Complaint topic prevalence.
4. Verified-purchase complaint share.
5. Review-title negativity.
6. Text-rating mismatch, such as negative text attached to 4-star or 5-star reviews.

## Baseline Models

1. Rating-only model.
2. Text-only model.
3. Rating plus text model.

The key test is whether text features improve prediction beyond ratings.

## Suggested Hypotheses

H1: Negative review-text signals predict future product reputation decline beyond current average ratings.

H2: Complaint-topic emergence predicts future low-star review share more strongly than general sentiment alone.

H3: Text-rating mismatch provides an early warning signal because customers may describe problems before reducing star ratings.

H4: Verified-purchase negative text has stronger predictive value than non-verified negative text.

## Analysis Roadmap

1. Download and document the Amazon Reviews 2023 All_Beauty data.
2. Clean reviews: remove missing text, convert timestamps, identify parent products.
3. Create product-month panel data.
4. Build text features: sentiment, emotion, complaint topics, text length, title negativity.
5. Model future outcomes using lagged text features.
6. Compare rating-only, text-only, and combined models.
7. Run robustness checks by product age, review volume, verified purchases, and minimum review thresholds.
8. Write the paper around managerial value: early-warning marketing intelligence.

## Paper Structure

1. Introduction
2. Literature Review
   - Online reviews and electronic word-of-mouth
   - Marketing intelligence and customer insight
   - Text analytics in marketing
   - Product reputation and customer experience monitoring
3. Conceptual Framework and Hypotheses
4. Data and Measures
5. Methodology
6. Results
7. Discussion
8. Managerial Implications
9. Limitations and Future Research
10. Conclusion

## First Milestone

Create a clean review-level sample and a data audit table:

1. Number of reviews.
2. Number of products.
3. Date range.
4. Rating distribution.
5. Verified purchase share.
6. Missing text share.
7. Reviews per product.

