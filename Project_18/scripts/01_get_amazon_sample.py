"""
Download and prepare a manageable Amazon Reviews 2023 sample.

This script uses public McAuley Lab files hosted on Hugging Face. It does not
scrape live Amazon pages.

Default category: All_Beauty.
Default output: data/processed/all_beauty_reviews_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import sys
import urllib.request


REVIEW_URL_TEMPLATE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/{category}.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/processed/all_beauty_reviews_sample.csv",
        help="Output path relative to the project folder unless absolute.",
    )
    return parser.parse_args()


def unix_ms_to_date(value: int | float | None) -> str:
    if value is None:
        return ""
    try:
        return dt.datetime.fromtimestamp(float(value) / 1000, tz=dt.UTC).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def reservoir_sample_jsonl(url: str, sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    seen = 0

    with urllib.request.urlopen(url) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            seen += 1
            record = json.loads(raw_line)
            if len(sample) < sample_size:
                sample.append(record)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    sample[j] = record

    print(f"Read {seen:,} reviews from {url}", file=sys.stderr)
    print(f"Sampled {len(sample):,} reviews", file=sys.stderr)
    return sample


def clean_record(record: dict) -> dict:
    text = str(record.get("text") or "").replace("\r", " ").replace("\n", " ").strip()
    title = str(record.get("title") or "").replace("\r", " ").replace("\n", " ").strip()
    timestamp = record.get("timestamp")
    helpful_vote = record.get("helpful_vote")

    return {
        "parent_asin": record.get("parent_asin") or "",
        "asin": record.get("asin") or "",
        "user_id": record.get("user_id") or "",
        "review_date": unix_ms_to_date(timestamp),
        "rating": record.get("rating"),
        "verified_purchase": record.get("verified_purchase"),
        "helpful_vote": helpful_vote if helpful_vote is not None else 0,
        "review_title": title,
        "review_text": text,
        "text_length": len(text.split()),
    }


def write_csv(rows: list[dict], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parent_asin",
        "asin",
        "user_id",
        "review_date",
        "rating",
        "verified_purchase",
        "helpful_vote",
        "review_title",
        "review_text",
        "text_length",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    url = REVIEW_URL_TEMPLATE.format(category=args.category)
    sample = reservoir_sample_jsonl(url, args.sample_size, args.seed)
    rows = [clean_record(record) for record in sample]
    rows = [row for row in rows if row["review_text"]]

    write_csv(rows, output_path)
    print(f"Wrote {len(rows):,} cleaned reviews to {output_path}")


if __name__ == "__main__":
    main()

