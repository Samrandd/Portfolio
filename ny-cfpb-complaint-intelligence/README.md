# New York Financial Complaint Signals

An evidence-led Streamlit dashboard for exploring aggregated New York CFPB consumer-complaint patterns, company response outcomes, and exact 28-day changes.

## What it does

- Filters by product, issue, response outcome, submission channel, and timely response.
- Shows complaint volume and the response-outcome mix.
- Uses the daily aggregate to compare a selected 28-day period with the immediately preceding 28 days.
- Ranks issue-level change with both absolute counts and percentage change.
- Provides a deterministic, evidence-led “So what?” explanation with guardrails.

## What it does not claim

Complaint counts are investigation signals, not proof that a company caused harm or performed poorly. Monetary-relief records count outcomes only; they do not contain payment amounts. The dashboard intentionally excludes consumer narratives.

## Run locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app works with the existing monthly file. Add `data/daily_complaint_summary.csv` to activate the exact 28-day monitoring view.
