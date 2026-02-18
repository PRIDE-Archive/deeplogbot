#!/usr/bin/env python3
"""Generate the PRIDE overview supplementary figure (supp_pride_overview.png).

Computes all metrics directly from the Parquet data (2021-2025)
and PRIDE metadata files. No hardcoded numbers.

Usage:
    python scripts/generate_pride_overview.py
"""

import os
import subprocess
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "pride_data" / "data_downloads_parquet.parquet"
PROJECTS_JSON = PROJECT_ROOT / "pride_data" / "all_pride_projects.json"
FILES_JSON = PROJECT_ROOT / "pride_data" / "all_pride_files_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "paper" / "figures" / "supp_pride_overview.png"

MIN_YEAR = 2021


def _duckdb_conn():
    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='3GB'")
    tmp = os.path.abspath(str(PROJECT_ROOT / "duckdb-tmp"))
    os.makedirs(tmp, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{tmp}'")
    conn.execute("PRAGMA threads=2")
    return conn


def _grep_count(pattern: str, filepath: str) -> int:
    """Count occurrences of a pattern in a file using grep -c."""
    result = subprocess.run(
        ["grep", "-c", pattern, filepath],
        capture_output=True, text=True, timeout=300,
    )
    return int(result.stdout.strip()) if result.returncode == 0 else 0


def compute_metrics():
    """Compute all overview metrics from the Parquet and metadata files."""
    p = str(PARQUET_PATH).replace("'", "''")
    conn = _duckdb_conn()

    print("Computing metrics from Parquet (year >= 2021)...")

    row = conn.execute(f"""
        SELECT
            COUNT(*)                    AS total_downloads,
            COUNT(DISTINCT accession)   AS unique_projects,
            COUNT(DISTINCT filename)    AS unique_files,
            COUNT(DISTINCT "user")      AS unique_users,
            MIN(year)                   AS min_year,
            MAX(year)                   AS max_year
        FROM read_parquet('{p}')
        WHERE year >= {MIN_YEAR}
    """).fetchone()

    total_dl, n_projects, n_files, n_users, min_yr, max_yr = row

    countries_row = conn.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT country
            FROM read_parquet('{p}')
            WHERE year >= {MIN_YEAR}
                  AND country IS NOT NULL AND country != ''
                  AND country NOT LIKE '%{{%'
            GROUP BY country
            HAVING COUNT(*) > 100
        )
    """).fetchone()
    n_countries = countries_row[0]

    total_records = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{p}')
    """).fetchone()[0]

    conn.close()

    # Project coverage: parquet projects / total public PRIDE projects
    print("Counting total public PRIDE projects...")
    total_pride_projects = _grep_count('"accession" : "PXD', str(PROJECTS_JSON))

    # File coverage: parquet files / total PRIDE files
    print("Counting total PRIDE files...")
    total_pride_files = _grep_count('"fileName"', str(FILES_JSON))

    project_coverage = (n_projects / total_pride_projects * 100
                        if total_pride_projects > 0 else 0)
    file_coverage = (n_files / total_pride_files * 100
                     if total_pride_files > 0 else 0)

    metrics = {
        "total_downloads": total_dl,
        "unique_projects": n_projects,
        "unique_files": n_files,
        "unique_users": n_users,
        "dl_per_project": total_dl / max(n_projects, 1),
        "dl_per_file": total_dl / max(n_files, 1),
        "dl_per_user": total_dl / max(n_users, 1),
        "n_countries": n_countries,
        "project_coverage": round(project_coverage, 1),
        "file_coverage": round(file_coverage, 1),
        "total_records": total_records,
        "min_year": min_yr,
        "max_year": max_yr,
        "total_pride_projects": total_pride_projects,
        "total_pride_files": total_pride_files,
    }

    print("\nComputed metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.1f}" if isinstance(v, float) else f"  {k}: {v:,}")

    return metrics


def fmt_count(v):
    if v >= 1e6:
        return f"{v / 1e6:.2f} M"
    if v >= 1e3:
        return f"{v / 1e3:.2f} K"
    return f"{v:.0f}"


def make_figure(m):
    """Create the 2x3 panel overview figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(
        f"PRIDE Download Activity Overview ({m['min_year']}\u2013{m['max_year']})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # --- (A) Overall Scale Metrics ---
    ax = axes[0, 0]
    scale_labels = ["Total Downloads", "Unique Files", "Unique Projects", "Unique Users"]
    scale_values = [
        m["total_downloads"], m["unique_files"],
        m["unique_projects"], m["unique_users"],
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.barh(scale_labels, scale_values, color=colors,
                   edgecolor="white", height=0.6)
    for bar, v in zip(bars, scale_values):
        ax.text(bar.get_width() + max(scale_values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt_count(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(scale_values) * 1.25)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: fmt_count(x)))
    ax.set_title("Overall Scale", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- (B) Reuse Intensity ---
    ax = axes[0, 1]
    reuse_labels = ["Downloads\nper Project", "Downloads\nper File",
                    "Downloads\nper User"]
    reuse_values = [m["dl_per_project"], m["dl_per_file"], m["dl_per_user"]]
    bars = ax.bar(reuse_labels, reuse_values,
                  color=["#3498db", "#e67e22", "#2ecc71"],
                  edgecolor="white", width=0.6)
    for bar, v in zip(bars, reuse_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(reuse_values) * 0.02,
                f"{v:,.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(reuse_values) * 1.15)
    ax.set_title("Reuse Intensity", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Average Downloads")

    # --- (C) Geographic Reach ---
    ax = axes[0, 2]
    ax.text(0.5, 0.55, f"{m['n_countries']}", ha="center", va="center",
            fontsize=52, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    ax.text(0.5, 0.25, "Countries / Territories\n(>100 downloads each)",
            ha="center", va="center", fontsize=11, color="#7f8c8d",
            transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Geographic Reach", fontsize=12, fontweight="bold")

    # --- (D) Project Coverage ---
    ax = axes[1, 0]
    pc = m["project_coverage"]
    ax.pie([pc, 100 - pc], colors=["#3498db", "#ecf0f1"], startangle=90,
           wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{pc}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#2c3e50")
    ax.set_title("Project Coverage\n(of public datasets)",
                 fontsize=12, fontweight="bold")

    # --- (E) File Coverage ---
    ax = axes[1, 1]
    fc = m["file_coverage"]
    ax.pie([fc, 100 - fc], colors=["#2ecc71", "#ecf0f1"], startangle=90,
           wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{fc}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#2c3e50")
    ax.set_title("File Coverage\n(downloaded at least once)",
                 fontsize=12, fontweight="bold")

    # --- (F) Time period label ---
    ax = axes[1, 2]
    ax.text(0.5, 0.55,
            f"Jan {m['min_year']} \u2013 Dec {m['max_year']}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#2c3e50", transform=ax.transAxes)
    ax.text(0.5, 0.30,
            f"5 years of download logs\n{m['total_records'] / 1e6:.1f} M raw records",
            ha="center", va="center", fontsize=11, color="#7f8c8d",
            transform=ax.transAxes)
    ax.axis("off")
    ax.set_title("Study Period", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")


def main():
    metrics = compute_metrics()
    make_figure(metrics)


if __name__ == "__main__":
    main()
