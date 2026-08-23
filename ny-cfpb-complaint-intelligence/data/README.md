# Dashboard data exports

The dashboard intentionally uses privacy-safe aggregates only. Do not add consumer narratives to this directory.

## Files

- `monthly_complaint_summary.csv` — already exported; powers the monthly trend and response-outcome view.
- `daily_complaint_summary.csv` — required for the exact selected-28-days versus previous-28-days monitor.

Run this final cell in the Colab notebook to create the daily file. It uses the original New York complaint data, not the deduplicated model-training data.

```python
# Daily dashboard file for exact 28-day comparisons
daily_summary = (
    dashboard_df.groupby(
        [
            "Date received",
            "Product",
            "Issue",
            "Submitted via",
            "Company response to consumer",
            "Timely response?"
        ]
    )
    .size()
    .reset_index(name="Complaints")
)

daily_summary.to_csv("daily_complaint_summary.csv", index=False)

print("Daily dashboard file created.")
print("Rows:", len(daily_summary))
daily_summary.head()
```

Then download the file and add it beside the monthly CSV.

## Evidence guardrails

1. Python calculates all complaint counts, shares, date windows, and changes.
2. The dashboard does not calculate a 28-day comparison from monthly data.
3. A ranked issue must have at least 20 complaints in the selected period; the whole comparison needs at least 10 complaints in the prior period.
4. “Monetary relief” is an outcome count, not a dollar value.
5. Any future LLM insight must receive only the verified aggregate evidence packet and must not add, change, or infer numbers.
6. The interface labels all results as signals for review, not proof of causation or company performance.
