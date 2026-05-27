"""
Add managerial complaint-topic features to the product-month panel.

These topics are designed for interpretability in a Marketing Intelligence &
Planning paper. They convert unstructured customer voice into actionable
planning signals such as packaging, skin reaction, authenticity, fragrance, and
value concerns.

Outputs:
- data/processed/review_complaint_topics_full.csv
- data/processed/product_month_topic_panel_full.csv
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd


TOPIC_PATTERNS = {
    "skin_reaction": [
        r"\brash(es)?\b",
        r"\birritat(?:ed|es|ing|ion)\b",
        r"\ballerg(?:ic|y|ies)\b",
        r"\bitch(?:y|ing)?\b",
        r"\bbreak\s?out(s)?\b",
        r"\bacne\b",
        r"\bburn(?:ed|ing)?\b",
        r"\bredness\b",
        r"\bswell(?:ing|ed)?\b",
    ],
    "scent_fragrance": [
        r"\bsmell(?:s|ed|ing)?\b",
        r"\bscent\b",
        r"\bfragrance\b",
        r"\bodou?r\b",
        r"\bstink(?:s|y)?\b",
        r"\bperfume\b",
        r"\btoo strong\b",
    ],
    "packaging_leakage": [
        r"\bleak(?:s|ed|ing)?\b",
        r"\bspill(?:ed|ing|s)?\b",
        r"\bbroken\b",
        r"\bdamaged\b",
        r"\bpackage\b",
        r"\bpackaging\b",
        r"\bbottle\b",
        r"\bpump\b",
        r"\bcap\b",
        r"\bseal(?:ed)?\b",
    ],
    "authenticity_counterfeit": [
        r"\bfake\b",
        r"\bcounterfeit\b",
        r"\bknock\s?off\b",
        r"\bnot authentic\b",
        r"\bnot genuine\b",
        r"\boriginal\b",
        r"\bauthentic\b",
    ],
    "effectiveness_quality": [
        r"\bdoes(?:n'?t| not) work\b",
        r"\bdid(?:n'?t| not) work\b",
        r"\bnot working\b",
        r"\bineffective\b",
        r"\buseless\b",
        r"\bpoor quality\b",
        r"\blow quality\b",
        r"\bcheap\b",
        r"\bdisappoint(?:ed|ing|ment)\b",
    ],
    "value_price": [
        r"\boverpriced\b",
        r"\btoo expensive\b",
        r"\bexpensive\b",
        r"\bprice\b",
        r"\bwaste of money\b",
        r"\bnot worth\b",
        r"\bworthless\b",
        r"\brefund\b",
    ],
    "delivery_condition": [
        r"\bshipping\b",
        r"\bdelivery\b",
        r"\barrived\b",
        r"\blate\b",
        r"\bbox\b",
        r"\bmissing\b",
        r"\bdamaged in transit\b",
    ],
    "texture_usability": [
        r"\bsticky\b",
        r"\bgreasy\b",
        r"\boily\b",
        r"\bdry(?:ing)?\b",
        r"\bthick\b",
        r"\bwatery\b",
        r"\brunny\b",
        r"\bclumpy\b",
        r"\bmessy\b",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        default="data/processed/all_beauty_reviews_full.csv",
    )
    parser.add_argument(
        "--panel",
        default="data/processed/product_month_text_panel_full.csv",
    )
    parser.add_argument(
        "--review-output",
        default="data/processed/review_complaint_topics_full.csv",
    )
    parser.add_argument(
        "--panel-output",
        default="data/processed/product_month_topic_panel_full.csv",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> pathlib.Path:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else project_dir / path


def compile_patterns() -> dict[str, re.Pattern]:
    return {
        topic: re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.I)
        for topic, patterns in TOPIC_PATTERNS.items()
    }


def main() -> None:
    args = parse_args()
    reviews_path = resolve_project_path(args.reviews)
    panel_path = resolve_project_path(args.panel)
    review_output_path = resolve_project_path(args.review_output)
    panel_output_path = resolve_project_path(args.panel_output)

    reviews = pd.read_csv(reviews_path)
    reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")
    reviews = reviews.dropna(subset=["review_date", "parent_asin"])
    reviews["month"] = reviews["review_date"].dt.to_period("M").dt.to_timestamp()

    combined_text = (
        reviews["review_title"].fillna("").astype(str)
        + " "
        + reviews["review_text"].fillna("").astype(str)
    )
    patterns = compile_patterns()

    topic_cols = []
    topic_frame = reviews[["parent_asin", "asin", "user_id", "review_date", "month"]].copy()
    for topic, pattern in patterns.items():
        col = f"topic_{topic}"
        topic_frame[col] = combined_text.map(lambda value, rx=pattern: bool(rx.search(str(value))))
        topic_cols.append(col)

    topic_frame["any_complaint_topic"] = topic_frame[topic_cols].any(axis=1)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    topic_frame.to_csv(review_output_path, index=False)

    topic_panel = (
        topic_frame.groupby(["parent_asin", "month"])
        .agg(**{f"{col}_share": (col, "mean") for col in topic_cols})
        .reset_index()
    )
    topic_panel["any_complaint_topic_share"] = (
        topic_frame.groupby(["parent_asin", "month"])["any_complaint_topic"]
        .mean()
        .to_numpy()
    )

    for col in topic_panel.columns:
        if col not in {"parent_asin", "month"}:
            topic_panel[col] = topic_panel[col].round(4)

    panel = pd.read_csv(panel_path, parse_dates=["month"])
    combined = panel.merge(topic_panel, on=["parent_asin", "month"], how="left")
    combined.to_csv(panel_output_path, index=False)

    topic_summary = (
        topic_frame[topic_cols + ["any_complaint_topic"]].mean().sort_values(ascending=False)
    )
    print(topic_summary.round(4).to_string())
    print(f"\nWrote review topic features to {review_output_path}")
    print(f"Wrote product-month topic panel to {panel_output_path}")
    print(f"Rows in combined panel: {len(combined):,}")


if __name__ == "__main__":
    main()
