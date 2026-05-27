"""
Create review-level and product-month text intelligence features.

This first pass uses transparent lexicon features that are easy to explain in
a marketing paper. Later, these can be extended with VADER, transformer
sentiment, or topic models.

Outputs:
- data/processed/review_text_features_full.csv
- data/processed/product_month_text_panel_full.csv
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd


POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "clean",
    "comfortable",
    "easy",
    "effective",
    "excellent",
    "favorite",
    "fresh",
    "gentle",
    "good",
    "great",
    "happy",
    "like",
    "love",
    "loved",
    "nice",
    "perfect",
    "recommend",
    "smooth",
    "soft",
    "works",
    "worth",
}

NEGATIVE_WORDS = {
    "allergic",
    "awful",
    "bad",
    "broke",
    "cheap",
    "complaint",
    "damaged",
    "disappointed",
    "disappointing",
    "dry",
    "expired",
    "fake",
    "hate",
    "horrible",
    "irritated",
    "irritation",
    "itchy",
    "leaked",
    "poor",
    "rash",
    "refund",
    "return",
    "smell",
    "sticky",
    "terrible",
    "waste",
    "worst",
}

COMPLAINT_PATTERNS = [
    r"\brefund\b",
    r"\breturn(ed|ing)?\b",
    r"\bdoes not work\b",
    r"\bdid not work\b",
    r"\bnot work(ing)?\b",
    r"\bwaste of money\b",
    r"\bbroke(n)?\b",
    r"\bdamaged\b",
    r"\bleak(ed|ing)?\b",
    r"\bexpired\b",
    r"\bfake\b",
    r"\brash\b",
    r"\birritat(ed|ion|ing)?\b",
    r"\ballergic\b",
    r"\bdry\b",
    r"\bbad smell\b",
    r"\bsmells bad\b",
    r"\bdisappointed\b",
]

TOKEN_RE = re.compile(r"[a-z']+")
COMPLAINT_RE = re.compile("|".join(COMPLAINT_PATTERNS), re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        default="data/processed/all_beauty_reviews_full.csv",
        help="Cleaned review-level CSV.",
    )
    parser.add_argument(
        "--panel",
        default="data/processed/product_month_panel_full.csv",
        help="Product-month panel CSV.",
    )
    parser.add_argument(
        "--review-output",
        default="data/processed/review_text_features_full.csv",
    )
    parser.add_argument(
        "--panel-output",
        default="data/processed/product_month_text_panel_full.csv",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> pathlib.Path:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else project_dir / path


def count_words(text: str, vocabulary: set[str]) -> int:
    return sum(1 for token in TOKEN_RE.findall(text.lower()) if token in vocabulary)


def add_review_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["review_text"].fillna("").astype(str)
    title = df["review_title"].fillna("").astype(str)
    combined = (title + " " + text).str.lower()

    df["positive_word_count"] = combined.map(lambda value: count_words(value, POSITIVE_WORDS))
    df["negative_word_count"] = combined.map(lambda value: count_words(value, NEGATIVE_WORDS))
    df["lexicon_sentiment"] = (
        (df["positive_word_count"] - df["negative_word_count"])
        / (df["positive_word_count"] + df["negative_word_count"] + 1)
    ).round(4)
    df["has_complaint_language"] = combined.str.contains(COMPLAINT_RE, regex=True)
    df["has_negative_title"] = title.str.lower().map(
        lambda value: count_words(value, NEGATIVE_WORDS) > 0
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text_rating_mismatch"] = (
        (df["rating"] >= 4)
        & ((df["lexicon_sentiment"] < 0) | df["has_complaint_language"])
    )
    df["negative_text_flag"] = (
        (df["lexicon_sentiment"] < 0) | df["has_complaint_language"] | df["has_negative_title"]
    )
    return df


def build_monthly_text_panel(review_features: pd.DataFrame) -> pd.DataFrame:
    review_features["review_date"] = pd.to_datetime(
        review_features["review_date"], errors="coerce"
    )
    review_features = review_features.dropna(subset=["review_date", "parent_asin"])
    review_features["month"] = review_features["review_date"].dt.to_period("M").dt.to_timestamp()

    text_panel = (
        review_features.groupby(["parent_asin", "month"])
        .agg(
            avg_lexicon_sentiment=("lexicon_sentiment", "mean"),
            negative_text_share=("negative_text_flag", "mean"),
            complaint_language_share=("has_complaint_language", "mean"),
            negative_title_share=("has_negative_title", "mean"),
            text_rating_mismatch_share=("text_rating_mismatch", "mean"),
            avg_positive_word_count=("positive_word_count", "mean"),
            avg_negative_word_count=("negative_word_count", "mean"),
        )
        .reset_index()
    )

    for col in text_panel.columns:
        if col not in {"parent_asin", "month"}:
            text_panel[col] = text_panel[col].round(4)

    return text_panel


def main() -> None:
    args = parse_args()
    reviews_path = resolve_project_path(args.reviews)
    panel_path = resolve_project_path(args.panel)
    review_output_path = resolve_project_path(args.review_output)
    panel_output_path = resolve_project_path(args.panel_output)

    reviews = pd.read_csv(reviews_path)
    review_features = add_review_text_features(reviews)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_features.to_csv(review_output_path, index=False)

    text_panel = build_monthly_text_panel(review_features)
    panel = pd.read_csv(panel_path, parse_dates=["month"])
    combined = panel.merge(text_panel, on=["parent_asin", "month"], how="left")
    combined.to_csv(panel_output_path, index=False)

    print(f"Wrote review text features to {review_output_path}")
    print(f"Wrote product-month text panel to {panel_output_path}")
    print(f"Rows in combined panel: {len(combined):,}")
    print(
        "Mean complaint-language share: "
        f"{combined['complaint_language_share'].mean():.4f}"
    )
    print(f"Mean negative-text share: {combined['negative_text_share'].mean():.4f}")


if __name__ == "__main__":
    main()

