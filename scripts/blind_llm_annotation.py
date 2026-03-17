#!/usr/bin/env python3
"""Blind LLM annotation of PRIDE download locations using Ollama (Qwen3).

Classifies locations WITHOUT any classifier labels or confidence scores.
Only provides raw behavioral features + geographic enrichment.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import ollama
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
LLM_CORRECTIONS = DATA_DIR / "llm_corrections.csv"
ENRICHMENT_CACHE = DATA_DIR / "location_enrichment.json"
SYSTEM_PROMPT = DATA_DIR / "prompts" / "annotation_system_prompt.txt"

# Feature columns to include (NO classifier outputs)
FEATURE_COLS = [
    "geo_location", "city", "country",
    "unique_users", "downloads_per_user", "total_downloads",
    "unique_projects", "working_hours_ratio", "night_activity_ratio",
    "hourly_entropy", "years_span", "fraction_latest_year",
    "spike_ratio", "aspera_ratio", "globus_ratio",
    "user_entropy", "user_gini_coefficient",
]

# Columns that MUST NOT be included (classifier outputs)
EXCLUDED_COLS = [
    "behavior_type", "user_category", "automation_category",
    "classification_confidence", "needs_review",
    "is_bot", "is_hub", "is_organic", "is_protected_hub",
    "validation_zone", "manual_label", "reviewer_notes",
    "claude_evaluation", "claude_notes",
]


def format_location_prompt(row: pd.Series, enrichment: dict) -> str:
    """Format a single location as a prompt for the LLM."""
    city = row["city"]
    country = row["country"]
    key = f"{city}|{country}"

    enr = enrichment.get(key, {})
    enrichment_text = enr.get(
        "enrichment_text",
        f"Research context for {city}, {country}: No enrichment data available."
    )

    prompt = f"""Classify this download location from the PRIDE proteomics database (2021-2025):

## Location
- City: {city}
- Country: {country}
- Coordinates: {row['geo_location']}

## Behavioral Features
- Unique users (distinct IP hashes): {int(row['unique_users'])}
- Downloads per user (average): {row['downloads_per_user']:.1f}
- Total downloads (5 years): {int(row['total_downloads']):,}
- Unique PRIDE datasets accessed: {int(row['unique_projects'])}
- Working hours ratio (9am-6pm local): {row['working_hours_ratio']:.2f}
- Night activity ratio (midnight-6am local): {row['night_activity_ratio']:.2f}
- Hourly entropy (0=concentrated, 3.18=uniform): {row['hourly_entropy']:.2f}
- Years with activity: {int(row['years_span'])}
- Fraction of downloads in latest year: {row['fraction_latest_year']:.2f}
- Spike ratio (latest/previous year): {row['spike_ratio']:.1f}
- Aspera protocol share: {row['aspera_ratio']:.3f}
- Globus protocol share: {row['globus_ratio']:.3f}
- User entropy (download distribution): {row['user_entropy']:.2f}
- User Gini coefficient: {row['user_gini_coefficient']:.2f}

## Research Context
{enrichment_text}

Based on these features, classify this location as bot, hub, or organic."""

    return prompt


def parse_response(text: str) -> dict:
    """Parse LLM response to extract label, confidence, and reasoning."""
    result = {"label": None, "confidence": None, "reasoning": None, "raw_response": text}

    # Extract LABEL
    label_match = re.search(r"LABEL:\s*(bot|hub|organic)", text, re.IGNORECASE)
    if label_match:
        result["label"] = label_match.group(1).lower()

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    # Extract REASONING
    reason_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    return result


def load_progress(output_path: Path) -> set:
    """Load already-processed geo_locations from output file."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["geo_location"])
    return completed


def annotate_with_qwen3(
    df: pd.DataFrame,
    enrichment: dict,
    system_prompt: str,
    output_path: Path,
    model: str = "qwen3:latest",
):
    """Run blind annotation using Qwen3 via Ollama."""
    completed = load_progress(output_path)
    remaining = df[~df["geo_location"].isin(completed)]

    print(f"Total locations: {len(df)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All locations already annotated. Nothing to do.")
        return

    # Open output file in append mode
    file_exists = output_path.exists() and len(completed) > 0
    fieldnames = [
        "geo_location", "city", "country", "label", "confidence",
        "reasoning", "raw_response",
    ]

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for idx, (_, row) in enumerate(remaining.iterrows()):
            geo = row["geo_location"]
            city = row["city"]
            country = row["country"]

            prompt = format_location_prompt(row, enrichment)

            start_time = time.time()
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.1},  # Low temp for consistency
                )
                text = response["message"]["content"]
                parsed = parse_response(text)

                writer.writerow({
                    "geo_location": geo,
                    "city": city,
                    "country": country,
                    "label": parsed["label"],
                    "confidence": parsed["confidence"],
                    "reasoning": parsed["reasoning"],
                    "raw_response": text.replace("\n", " "),
                })
                f.flush()

                elapsed = time.time() - start_time
                label = parsed["label"] or "PARSE_ERROR"
                print(
                    f"  [{idx+1}/{len(remaining)}] {city}, {country}: "
                    f"{label} ({elapsed:.1f}s)"
                )

            except Exception as e:
                elapsed = time.time() - start_time
                print(
                    f"  [{idx+1}/{len(remaining)}] {city}, {country}: "
                    f"ERROR ({elapsed:.1f}s): {e}"
                )
                writer.writerow({
                    "geo_location": geo,
                    "city": city,
                    "country": country,
                    "label": None,
                    "confidence": None,
                    "reasoning": f"ERROR: {e}",
                    "raw_response": "",
                })
                f.flush()

    # Summary
    results = pd.read_csv(output_path)
    print(f"\n=== Annotation Complete ===")
    print(f"Total annotated: {len(results)}")
    print(f"Label distribution:")
    print(results["label"].value_counts().to_string())
    print(f"Parse errors: {results['label'].isna().sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Blind LLM annotation of PRIDE download locations"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATA_DIR / "blind_annotations_qwen3.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3:latest",
        help="Ollama model name",
    )
    args = parser.parse_args()

    # Load data
    print("Loading location data...")
    df = pd.read_csv(LLM_CORRECTIONS)

    # Verify no classifier columns leak
    for col in EXCLUDED_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Loading enrichment cache...")
    with open(ENRICHMENT_CACHE) as f:
        enrichment = json.load(f)

    print("Loading system prompt...")
    system_prompt = SYSTEM_PROMPT.read_text()

    output_path = Path(args.output)
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print()

    annotate_with_qwen3(df, enrichment, system_prompt, output_path, args.model)


if __name__ == "__main__":
    main()
