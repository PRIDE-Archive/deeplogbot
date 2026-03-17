#!/usr/bin/env python3
"""Prepare location batches for Claude Code Agent-based annotation.

Splits 1,153 locations into batch files that can be fed to Claude Code
subagents for blind classification. Each batch file contains ~50 locations
with their behavioral features and geographic enrichment.
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
LLM_CORRECTIONS = DATA_DIR / "llm_corrections.csv"
ENRICHMENT_CACHE = DATA_DIR / "location_enrichment.json"
SYSTEM_PROMPT = DATA_DIR / "prompts" / "annotation_system_prompt.txt"
BATCH_DIR = DATA_DIR / "batches"

BATCH_SIZE = 50

# Columns that MUST NOT be included
EXCLUDED_COLS = [
    "behavior_type", "user_category", "automation_category",
    "classification_confidence", "needs_review",
    "is_bot", "is_hub", "is_organic", "is_protected_hub",
    "validation_zone", "manual_label", "reviewer_notes",
    "claude_evaluation", "claude_notes",
]


def format_location(row: pd.Series, enrichment: dict) -> str:
    """Format a single location for the batch file."""
    city = row["city"]
    country = row["country"]
    key = f"{city}|{country}"

    enr = enrichment.get(key, {})
    enrichment_text = enr.get(
        "enrichment_text",
        f"Research context for {city}, {country}: No enrichment data available."
    )

    def _safe_int(val, default=0):
        try:
            return int(val) if pd.notna(val) else default
        except (ValueError, TypeError):
            return default

    def _safe_float(val, fmt='.1f', default=0.0):
        try:
            return format(float(val), fmt) if pd.notna(val) else format(default, fmt)
        except (ValueError, TypeError):
            return format(default, fmt)

    return f"""### Location: {row['geo_location']}
- City: {city}
- Country: {country}
- Coordinates: {row['geo_location']}
- Unique users: {_safe_int(row.get('unique_users'))}
- Downloads per user: {_safe_float(row.get('downloads_per_user'), '.1f')}
- Total downloads: {_safe_int(row.get('total_downloads')):,}
- Unique PRIDE datasets accessed: {_safe_int(row.get('unique_projects'))}
- Working hours ratio (9am-6pm local): {_safe_float(row.get('working_hours_ratio'), '.2f')}
- Night activity ratio (midnight-6am local): {_safe_float(row.get('night_activity_ratio'), '.2f')}
- Hourly entropy (0=concentrated, 3.18=uniform): {_safe_float(row.get('hourly_entropy'), '.2f')}
- Years with activity: {_safe_int(row.get('years_span'))}
- Fraction in latest year: {_safe_float(row.get('fraction_latest_year'), '.2f')}
- Spike ratio: {_safe_float(row.get('spike_ratio'), '.1f')}
- Aspera share: {_safe_float(row.get('aspera_ratio'), '.3f')}
- Globus share: {_safe_float(row.get('globus_ratio'), '.3f')}
- User entropy: {_safe_float(row.get('user_entropy'), '.2f')}
- User Gini: {_safe_float(row.get('user_gini_coefficient'), '.2f')}
{enrichment_text}
"""


def main():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(LLM_CORRECTIONS)
    for col in EXCLUDED_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    with open(ENRICHMENT_CACHE) as f:
        enrichment = json.load(f)

    system_prompt = SYSTEM_PROMPT.read_text()

    # Split into batches
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Creating {n_batches} batches of ~{BATCH_SIZE} locations each...")

    batch_manifest = []

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end]

        batch_file = BATCH_DIR / f"batch_{batch_idx:03d}.txt"

        locations_text = ""
        geo_locations = []
        for _, row in batch_df.iterrows():
            locations_text += format_location(row, enrichment)
            locations_text += "---\n\n"
            geo_locations.append(row["geo_location"])

        batch_content = f"""# Blind Classification Batch {batch_idx:03d}
# Locations {start+1}-{end} of {len(df)}

## Instructions
{system_prompt}

## IMPORTANT
- Classify EACH location below independently
- Do NOT use any prior knowledge of DeepLogBot classifications
- For EACH location, output exactly:
  GEO: [coordinates]
  LABEL: [bot|hub|organic]
  CONFIDENCE: [high|medium|low]
  REASONING: [2-3 sentences]

## Locations to Classify

{locations_text}
"""

        with open(batch_file, "w") as f:
            f.write(batch_content)

        batch_manifest.append({
            "batch_idx": batch_idx,
            "file": str(batch_file),
            "n_locations": len(batch_df),
            "geo_locations": geo_locations,
        })

    # Save manifest
    manifest_path = BATCH_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(batch_manifest, f, indent=2)

    print(f"Created {n_batches} batch files in {BATCH_DIR}/")
    print(f"Manifest saved to {manifest_path}")
    print(f"Total locations: {len(df)}")


if __name__ == "__main__":
    main()
