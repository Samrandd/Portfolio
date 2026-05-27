"""
Build the product-month panel for the paper.

Input: cleaned review-level CSV from get_amazon_sample.py
Output: data/processed/product_month_panel.csv

The panel is the analytical backbone of the paper:
one product-month row, current-period review/rating/text measures, and future
three-month outcomes for later prediction models.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/all_beauty_reviews_sample.csv",
        help="Input CSV path relative to the project folder unless absolute.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/product_month_panel.csv",
        help="Output CSV path relative to the project folder unless absolute.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> pathlib.Path:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else project_dir / path


def share(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def build_current_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date", "parent_asin", "rating"])
    df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)
    df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce").fillna(0)
    df["is_low_star"] = df["rating"] <= 2
    df["is_five_star"] = df["rating"] == 5
    df["verified_bool"] = df["verified_purchase"].astype(str).str.lower().eq("true")

    panel = (
        df.groupby(["parent_asin", "month"])
        .agg(
            review_count=("rating", "size"),
            avg_rating=("rating", "mean"),
            low_star_share=("is_low_star", share),
            five_star_share=("is_five_star", share),
            verified_purchase_share=("verified_bool", share),
            avg_helpful_votes=("helpful_vote", "mean"),
            avg_text_words=("text_length", "mean"),
            median_text_words=("text_length", "median"),
            unique_reviewers=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values(["parent_asin", "month"])
    )

    for col in [
        "avg_rating",
        "low_star_share",
        "five_star_share",
        "verified_purchase_share",
        "avg_helpful_votes",
        "avg_text_words",
        "median_text_words",
    ]:
        panel[col] = panel[col].round(4)

    return panel


def add_future_three_month_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["parent_asin", "month"]).copy()
    future_parts = []

    for months_ahead in (1, 2, 3):
        future = panel[
            ["parent_asin", "month", "review_count", "avg_rating", "low_star_share"]
        ].copy()
        future["month"] = future["month"] - pd.DateOffset(months=months_ahead)
        future["weighted_rating_sum"] = future["avg_rating"] * future["review_count"]
        future["weighted_low_star_sum"] = (
            future["low_star_share"] * future["review_count"]
        )
        future_parts.append(
            future[
                [
                    "parent_asin",
                    "month",
                    "review_count",
                    "weighted_rating_sum",
                    "weighted_low_star_sum",
                ]
            ]
        )

    future_long = pd.concat(future_parts, ignore_index=True)
    future_panel = (
        future_long.groupby(["parent_asin", "month"], as_index=False)
        .agg(
            future_3m_review_count=("review_count", "sum"),
            weighted_rating_sum=("weighted_rating_sum", "sum"),
            weighted_low_star_sum=("weighted_low_star_sum", "sum"),
        )
    )
    future_panel["future_3m_avg_rating"] = (
        future_panel["weighted_rating_sum"] / future_panel["future_3m_review_count"]
    ).round(4)
    future_panel["future_3m_low_star_share"] = (
        future_panel["weighted_low_star_sum"] / future_panel["future_3m_review_count"]
    ).round(4)
    future_panel = future_panel.drop(
        columns=["weighted_rating_sum", "weighted_low_star_sum"]
    )

    merged = panel.merge(future_panel, on=["parent_asin", "month"], how="left")
    merged["future_3m_review_count"] = merged["future_3m_review_count"].fillna(0).astype(int)
    merged["future_3m_rating_decline"] = (
        merged["avg_rating"] - merged["future_3m_avg_rating"]
    ).round(4)
    merged["has_future_3m_reviews"] = merged["future_3m_review_count"] > 0
    return merged


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    panel = build_current_month_panel(df)
    panel = add_future_three_month_outcomes(panel)
    panel.to_csv(output_path, index=False)

    usable = int(panel["has_future_3m_reviews"].sum())
    print(f"Wrote {len(panel):,} product-month rows to {output_path}")
    print(f"Rows with future 3-month reviews: {usable:,}")
    print(f"Products in panel: {panel['parent_asin'].nunique():,}")
    print(f"Panel date range: {panel['month'].min().date()} to {panel['month'].max().date()}")


if __name__ == "__main__":
    main()
