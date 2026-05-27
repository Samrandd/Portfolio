"""
Create a first audit of the cleaned Amazon review sample.

Outputs:
- data/processed/data_audit_summary.csv
- data/processed/rating_distribution.csv
- data/processed/reviews_by_year.csv
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
    return parser.parse_args()


def resolve_project_path(path_value: str) -> pathlib.Path:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else project_dir / path


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = input_path.parents[0]

    df = pd.read_csv(input_path)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)
    df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce").fillna(0)

    summary = pd.DataFrame(
        [
            ("reviews", len(df)),
            ("unique_parent_products", df["parent_asin"].nunique()),
            ("unique_child_asins", df["asin"].nunique()),
            ("unique_users", df["user_id"].nunique()),
            ("date_min", df["review_date"].min().date().isoformat()),
            ("date_max", df["review_date"].max().date().isoformat()),
            ("average_rating", round(df["rating"].mean(), 3)),
            ("low_star_share_1_or_2", round((df["rating"] <= 2).mean(), 3)),
            ("verified_purchase_share", round(df["verified_purchase"].astype(str).str.lower().eq("true").mean(), 3)),
            ("average_text_words", round(df["text_length"].mean(), 2)),
            ("median_text_words", round(df["text_length"].median(), 2)),
            ("average_helpful_votes", round(df["helpful_vote"].mean(), 3)),
        ],
        columns=["metric", "value"],
    )

    rating_distribution = (
        df["rating"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="review_count")
    )
    rating_distribution["share"] = (rating_distribution["review_count"] / len(df)).round(4)

    reviews_by_year = (
        df.assign(year=df["review_date"].dt.year)
        .groupby("year", dropna=False)
        .size()
        .reset_index(name="review_count")
        .sort_values("year")
    )

    summary.to_csv(output_dir / "data_audit_summary.csv", index=False)
    rating_distribution.to_csv(output_dir / "rating_distribution.csv", index=False)
    reviews_by_year.to_csv(output_dir / "reviews_by_year.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote audit files to {output_dir}")


if __name__ == "__main__":
    main()

