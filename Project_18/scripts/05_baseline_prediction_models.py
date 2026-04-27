"""
Run first baseline prediction models.

Research test:
Do text intelligence features improve prediction of future product reputation
decline beyond ratings and review-volume controls?

This script uses NumPy OLS so it works without scikit-learn/statsmodels.

Outputs:
- data/processed/baseline_model_metrics.csv
- data/processed/baseline_model_coefficients.csv
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd


RATING_CONTROL_FEATURES = [
    "log_review_count",
    "avg_rating",
    "low_star_share",
    "five_star_share",
    "verified_purchase_share",
    "avg_helpful_votes",
    "avg_text_words",
]

TEXT_FEATURES = [
    "avg_lexicon_sentiment",
    "negative_text_share",
    "complaint_language_share",
    "negative_title_share",
    "text_rating_mismatch_share",
    "avg_positive_word_count",
    "avg_negative_word_count",
]

TOPIC_FEATURES = [
    "topic_skin_reaction_share",
    "topic_scent_fragrance_share",
    "topic_packaging_leakage_share",
    "topic_authenticity_counterfeit_share",
    "topic_effectiveness_quality_share",
    "topic_value_price_share",
    "topic_delivery_condition_share",
    "topic_texture_usability_share",
    "any_complaint_topic_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/product_month_topic_panel_full.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="data/processed/baseline_model_metrics.csv",
    )
    parser.add_argument(
        "--coef-output",
        default="data/processed/baseline_model_coefficients.csv",
    )
    parser.add_argument(
        "--test-start",
        default="2021-01-01",
        help="Months on/after this date are held out for testing.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str) -> pathlib.Path:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else project_dir / path


def prepare_model_frame(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month"])
    df = df[df["has_future_3m_reviews"].astype(str).str.lower().isin(["true", "1"])]
    df = df[df["future_3m_review_count"] >= 1].copy()
    df["log_review_count"] = np.log1p(df["review_count"])
    df["future_rating_decline_flag"] = df["future_3m_rating_decline"] >= 0.5

    needed = (
        RATING_CONTROL_FEATURES
        + TEXT_FEATURES
        + [feature for feature in TOPIC_FEATURES if feature in df.columns]
        + ["future_3m_rating_decline", "future_rating_decline_flag", "month"]
    )
    df = df.dropna(subset=needed)
    return df


def standardize_train_test(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[features].astype(float)
    test_x = test[features].astype(float)
    means = train_x.mean()
    stds = train_x.std(ddof=0).replace(0, 1)

    train_z = (train_x - means) / stds
    test_z = (test_x - means) / stds
    train_matrix = np.column_stack([np.ones(len(train_z)), train_z.to_numpy()])
    test_matrix = np.column_stack([np.ones(len(test_z)), test_z.to_numpy()])
    return train_matrix, test_matrix


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - np.sum(residual**2) / total) if total else np.nan
    return {"rmse": round(rmse, 5), "mae": round(mae, 5), "test_r2": round(r2, 5)}


def run_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    features: list[str],
) -> tuple[dict, pd.DataFrame]:
    x_train, x_test = standardize_train_test(train, test, features)
    y_train = train["future_3m_rating_decline"].to_numpy(dtype=float)
    y_test = test["future_3m_rating_decline"].to_numpy(dtype=float)

    beta = fit_ols(x_train, y_train)
    prediction = x_test @ beta
    metrics = evaluate(y_test, prediction)
    metrics.update(
        {
            "model": model_name,
            "features": len(features),
            "train_rows": len(train),
            "test_rows": len(test),
            "target": "future_3m_rating_decline",
        }
    )

    coefficients = pd.DataFrame(
        {
            "model": model_name,
            "feature": ["intercept"] + features,
            "coefficient": beta.round(6),
        }
    )
    return metrics, coefficients


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    metrics_output = resolve_project_path(args.metrics_output)
    coef_output = resolve_project_path(args.coef_output)

    df = prepare_model_frame(input_path)
    test_start = pd.Timestamp(args.test_start)
    train = df[df["month"] < test_start].copy()
    test = df[df["month"] >= test_start].copy()

    model_specs = [
        ("rating_controls_only", RATING_CONTROL_FEATURES),
        ("rating_controls_plus_text", RATING_CONTROL_FEATURES + TEXT_FEATURES),
        (
            "rating_controls_plus_text_topics",
            RATING_CONTROL_FEATURES + TEXT_FEATURES + TOPIC_FEATURES,
        ),
    ]

    metrics_rows = []
    coef_frames = []
    for model_name, features in model_specs:
        metrics, coefficients = run_model(train, test, model_name, features)
        metrics_rows.append(metrics)
        coef_frames.append(coefficients)

    metrics_df = pd.DataFrame(metrics_rows)
    coefficients_df = pd.concat(coef_frames, ignore_index=True)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output, index=False)
    coefficients_df.to_csv(coef_output, index=False)

    print(metrics_df.to_string(index=False))
    print(f"\nWrote metrics to {metrics_output}")
    print(f"Wrote coefficients to {coef_output}")


if __name__ == "__main__":
    main()
