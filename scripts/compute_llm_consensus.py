#!/usr/bin/env python3
"""Compute consensus labels from blind multi-LLM annotations.

Compares Claude and Qwen3 annotations, computes inter-annotator agreement
(Cohen's kappa), and produces consensus labels for the retraining pipeline.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"


def cohens_kappa(y1, y2):
    """Compute Cohen's kappa between two annotators."""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)

    # Confusion matrix
    matrix = {}
    for l1 in labels:
        matrix[l1] = {}
        for l2 in labels:
            matrix[l1][l2] = 0
    for a, b in zip(y1, y2):
        matrix[a][b] += 1

    # Observed agreement
    po = sum(matrix[l][l] for l in labels) / n

    # Expected agreement
    pe = 0
    for l in labels:
        p1 = sum(matrix[l][l2] for l2 in labels) / n
        p2 = sum(matrix[l1][l] for l1 in labels) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def per_class_agreement(y1, y2, label):
    """Compute agreement rate for a specific class."""
    pairs = [(a, b) for a, b in zip(y1, y2) if a == label or b == label]
    if not pairs:
        return 0.0
    agree = sum(1 for a, b in pairs if a == b)
    return agree / len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Compute multi-LLM consensus")
    parser.add_argument(
        "--claude", type=str,
        default=str(DATA_DIR / "blind_annotations_claude.csv"),
    )
    parser.add_argument(
        "--qwen3", type=str,
        default=str(DATA_DIR / "blind_annotations_qwen3.csv"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATA_DIR / "blind_llm_consensus.csv"),
    )
    parser.add_argument(
        "--report", type=str,
        default=str(DATA_DIR / "blind_llm_agreement_report.txt"),
    )
    args = parser.parse_args()

    print("Loading annotations...")
    claude_df = pd.read_csv(args.claude)
    qwen3_df = pd.read_csv(args.qwen3)

    print(f"Claude annotations: {len(claude_df)}")
    print(f"Qwen3 annotations: {len(qwen3_df)}")

    # Merge on geo_location
    merged = claude_df[["geo_location", "city", "country", "label", "confidence", "reasoning"]].merge(
        qwen3_df[["geo_location", "label", "confidence", "reasoning"]],
        on="geo_location",
        suffixes=("_claude", "_qwen3"),
    )

    print(f"Matched locations: {len(merged)}")

    # Filter out locations where either LLM failed to parse
    valid = merged.dropna(subset=["label_claude", "label_qwen3"])
    print(f"Both parsed successfully: {len(valid)}")

    # Compute consensus
    consensus = []
    for _, row in valid.iterrows():
        cl = row["label_claude"]
        ql = row["label_qwen3"]

        if cl == ql:
            consensus_label = cl
            agreement = "agree"
        else:
            consensus_label = None  # Needs human review
            agreement = "disagree"

        consensus.append({
            "geo_location": row["geo_location"],
            "city": row["city"],
            "country": row["country"],
            "claude_label": cl,
            "claude_confidence": row["confidence_claude"],
            "qwen3_label": ql,
            "qwen3_confidence": row["confidence_qwen3"],
            "consensus_label": consensus_label,
            "agreement": agreement,
            "claude_reasoning": row["reasoning_claude"],
            "qwen3_reasoning": row["reasoning_qwen3"],
        })

    consensus_df = pd.DataFrame(consensus)
    consensus_df.to_csv(args.output, index=False)
    print(f"\nConsensus saved to {args.output}")

    # Compute metrics
    y_claude = valid["label_claude"].tolist()
    y_qwen3 = valid["label_qwen3"].tolist()

    kappa = cohens_kappa(y_claude, y_qwen3)
    raw_agree = sum(1 for a, b in zip(y_claude, y_qwen3) if a == b) / len(y_claude)

    n_agree = consensus_df[consensus_df["agreement"] == "agree"].shape[0]
    n_disagree = consensus_df[consensus_df["agreement"] == "disagree"].shape[0]

    # Per-class stats
    labels = ["bot", "hub", "organic"]
    class_stats = {}
    for label in labels:
        agree_rate = per_class_agreement(y_claude, y_qwen3, label)
        claude_count = y_claude.count(label)
        qwen3_count = y_qwen3.count(label)
        class_stats[label] = {
            "agreement_rate": agree_rate,
            "claude_count": claude_count,
            "qwen3_count": qwen3_count,
        }

    # Confusion matrix between LLMs
    conf_matrix = pd.crosstab(
        pd.Series(y_claude, name="Claude"),
        pd.Series(y_qwen3, name="Qwen3"),
    )

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("BLIND MULTI-LLM ANNOTATION: AGREEMENT REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total locations: {len(valid)}")
    report.append(f"Agreed: {n_agree} ({n_agree/len(valid)*100:.1f}%)")
    report.append(f"Disagreed: {n_disagree} ({n_disagree/len(valid)*100:.1f}%)")
    report.append(f"")
    report.append(f"Cohen's Kappa: {kappa:.3f}")
    report.append(f"Raw Agreement: {raw_agree:.3f}")
    report.append("")
    report.append("Kappa interpretation:")
    if kappa > 0.8:
        report.append("  Almost perfect agreement (>0.8)")
    elif kappa > 0.6:
        report.append("  Substantial agreement (0.6-0.8)")
    elif kappa > 0.4:
        report.append("  Moderate agreement (0.4-0.6)")
    else:
        report.append("  Fair or poor agreement (<0.4)")
    report.append("")
    report.append("--- Per-class statistics ---")
    for label, stats in class_stats.items():
        report.append(
            f"  {label}: agreement={stats['agreement_rate']:.3f}, "
            f"claude={stats['claude_count']}, qwen3={stats['qwen3_count']}"
        )
    report.append("")
    report.append("--- Confusion Matrix (Claude vs Qwen3) ---")
    report.append(conf_matrix.to_string())
    report.append("")
    report.append("--- Consensus label distribution ---")
    consensus_labels = consensus_df["consensus_label"].value_counts(dropna=False)
    report.append(consensus_labels.to_string())
    report.append("")
    report.append("--- Disagreement cases (need human review) ---")
    disagreements = consensus_df[consensus_df["agreement"] == "disagree"]
    report.append(f"Total: {len(disagreements)}")
    if len(disagreements) > 0:
        report.append("")
        disagree_types = disagreements.apply(
            lambda r: f"{r['claude_label']}->{r['qwen3_label']}", axis=1
        ).value_counts()
        report.append("Disagreement patterns:")
        report.append(disagree_types.to_string())

    report_text = "\n".join(report)
    print(report_text)

    with open(args.report, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {args.report}")

    # Also save disagreements for human review
    if len(disagreements) > 0:
        disagree_path = Path(args.output).parent / "disagreements_for_review.csv"
        disagreements.to_csv(disagree_path, index=False)
        print(f"Disagreements saved to {disagree_path}")


if __name__ == "__main__":
    main()
