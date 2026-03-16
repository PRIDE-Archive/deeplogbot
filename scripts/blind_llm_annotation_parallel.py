#!/usr/bin/env python3
"""Parallel blind LLM annotation of PRIDE download locations using Ollama (Qwen3).

Uses ThreadPoolExecutor to send multiple concurrent requests to Ollama.
Disables Qwen3 thinking mode for faster inference.
Thread-safe CSV writing with a lock.
"""

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
LLM_CORRECTIONS = DATA_DIR / "llm_corrections.csv"
ENRICHMENT_CACHE = DATA_DIR / "location_enrichment.json"
SYSTEM_PROMPT = DATA_DIR / "prompts" / "annotation_system_prompt.txt"

EXCLUDED_COLS = [
    "behavior_type", "user_category", "automation_category",
    "classification_confidence", "needs_review",
    "is_bot", "is_hub", "is_organic", "is_protected_hub",
    "validation_zone", "manual_label", "reviewer_notes",
    "claude_evaluation", "claude_notes",
]


def format_location_prompt(row: pd.Series, enrichment: dict) -> str:
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

Based on these features, classify this location as bot, hub, or organic.
/no_think"""

    return prompt


def parse_response(text: str) -> dict:
    result = {"label": None, "confidence": None, "reasoning": None, "raw_response": text}

    label_match = re.search(r"LABEL:\s*(bot|hub|organic)", text, re.IGNORECASE)
    if label_match:
        result["label"] = label_match.group(1).lower()

    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    reason_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    return result


def load_progress(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["geo_location"])
    return completed


def annotate_single(row, enrichment, system_prompt, model, timeout=300):
    """Annotate a single location. Returns a dict with results."""
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
            options={"temperature": 0.1, "num_predict": 512},
        )
        text = response["message"]["content"]
        parsed = parse_response(text)
        elapsed = time.time() - start_time

        return {
            "geo_location": geo,
            "city": city,
            "country": country,
            "label": parsed["label"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "raw_response": text.replace("\n", " "),
            "elapsed": elapsed,
            "error": False,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "geo_location": geo,
            "city": city,
            "country": country,
            "label": None,
            "confidence": None,
            "reasoning": f"ERROR: {e}",
            "raw_response": "",
            "elapsed": elapsed,
            "error": True,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel blind LLM annotation via Ollama"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATA_DIR / "blind_annotations_qwen3.csv"),
    )
    parser.add_argument(
        "--model", type=str, default="qwen3:latest",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    print("Loading location data...")
    df = pd.read_csv(LLM_CORRECTIONS)
    for col in EXCLUDED_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Loading enrichment cache...")
    with open(ENRICHMENT_CACHE) as f:
        enrichment = json.load(f)

    print("Loading system prompt...")
    system_prompt = SYSTEM_PROMPT.read_text()

    output_path = Path(args.output)
    completed = load_progress(output_path)

    remaining = df[~df["geo_location"].isin(completed)]
    print(f"Total locations: {len(df)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Workers: {args.workers}")

    if len(remaining) == 0:
        print("All locations already annotated.")
        return

    fieldnames = [
        "geo_location", "city", "country", "label", "confidence",
        "reasoning", "raw_response",
    ]

    file_exists = output_path.exists() and len(completed) > 0
    write_lock = threading.Lock()
    counter = {"done": 0, "total": len(remaining)}

    f = open(output_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f.flush()

    rows = [row for _, row in remaining.iterrows()]

    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(annotate_single, row, enrichment, system_prompt, args.model): row
            for row in rows
        }

        for future in as_completed(futures):
            result = future.result()
            with write_lock:
                writer.writerow({
                    "geo_location": result["geo_location"],
                    "city": result["city"],
                    "country": result["country"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"],
                    "raw_response": result["raw_response"],
                })
                f.flush()
                counter["done"] += 1

            label = result["label"] or "PARSE_ERROR"
            status = "ERROR" if result["error"] else label
            elapsed_total = time.time() - start_all
            rate = counter["done"] / (elapsed_total / 3600)
            eta_hours = (counter["total"] - counter["done"]) / rate if rate > 0 else 0

            print(
                f"  [{counter['done']}/{counter['total']}] "
                f"{result['city']}, {result['country']}: {status} "
                f"({result['elapsed']:.1f}s) "
                f"[{rate:.0f}/hr, ETA {eta_hours:.1f}h]"
            )

    f.close()

    results = pd.read_csv(output_path)
    total_time = time.time() - start_all
    print(f"\n=== Annotation Complete ===")
    print(f"Total annotated: {len(results)}")
    print(f"Time: {total_time/3600:.1f}h")
    print(f"Label distribution:")
    print(results["label"].value_counts().to_string())
    print(f"Parse errors: {results['label'].isna().sum()}")


if __name__ == "__main__":
    main()
