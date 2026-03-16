#!/usr/bin/env python3
"""Generate publication-quality figures for the PRIDE manuscript.

Creates vector PDF figures from analysis data in output/full_deep_v11/.

Usage:
    python scripts/generate_figures.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import duckdb

# Style settings for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

project_root = Path(__file__).parent.parent
ANALYSIS_DIR = project_root / 'output' / 'full_deep_v11'
BENCHMARK_DIR = project_root / 'output' / 'full_deep_v11'
CLASSIFICATION_DIR = project_root / 'output' / 'full_deep_v11'
FIGURES_DIR = project_root / 'paper' / 'figures'
PARQUET_PATH = project_root / 'pride_data' / 'data_downloads_parquet.parquet'

EUROPEAN_COUNTRIES = [
    'Germany', 'United Kingdom', 'France', 'Spain', 'Italy', 'Netherlands',
    'Switzerland', 'Sweden', 'Denmark', 'Belgium', 'Finland', 'Norway',
    'Austria', 'Poland', 'Ireland',
]

# Wellcome Trust / OECD DAC low- and middle-income countries present in PRIDE data
# (https://wellcome.org/research-funding/guidance/prepare-to-apply/low-and-middle-income-countries)
# China excluded here (already dominant in other panels)
LMIC_COUNTRIES = [
    'India', 'Mexico', 'Brazil', 'Turkey', 'Argentina',
    'Philippines', 'South Africa', 'Ukraine', 'Panama', 'Malaysia',
    'Algeria', 'Thailand', 'Indonesia', 'Bulgaria', 'Colombia',
    'Pakistan', 'Romania', 'Serbia', 'Cuba', 'Sri Lanka', 'Morocco',
    'Egypt', 'Vietnam', 'Bangladesh', 'Peru', 'Tunisia', 'Kenya',
    'Iran', 'Nigeria',
]

# Color palette
COLORS = {
    'bot': '#E74C3C',       # red
    'hub': '#3498DB',       # blue
    'organic': '#2ECC71',   # green
    'rules': '#E67E22',     # orange
    'deep': '#9B59B6',      # purple
}


def figure_bot_detection_overview(output_dir):
    """Combined figure: (A) Pipeline workflow + (B) Full-dataset classification distribution."""
    print("  Bot detection overview (combined)...")
    csv_path = CLASSIFICATION_DIR / 'location_analysis.csv'
    summary_path = CLASSIFICATION_DIR / 'classification_summary.json'

    if summary_path.exists():
        with open(summary_path) as f:
            stats = json.load(f)
    elif csv_path.exists():
        # Generate stats from classification CSV
        _df = pd.read_csv(csv_path, usecols=['behavior_type', 'total_downloads'], low_memory=False)
        total_locs = len(_df)
        total_dl = int(_df['total_downloads'].sum())
        bot_locs = int((_df['behavior_type'] == 'bot').sum())
        hub_locs = int((_df['behavior_type'] == 'hub').sum())
        user_locs = int((_df['behavior_type'] == 'user').sum())
        bot_dl = int(_df.loc[_df['behavior_type'] == 'bot', 'total_downloads'].sum())
        hub_dl = int(_df.loc[_df['behavior_type'] == 'hub', 'total_downloads'].sum())
        user_dl = int(_df.loc[_df['behavior_type'] == 'user', 'total_downloads'].sum())
        stats = {
            'total_locations': total_locs, 'total_downloads': total_dl,
            'bot_locations': bot_locs, 'hub_locations': hub_locs,
            'organic_locations': user_locs,
            'bot_downloads': bot_dl, 'hub_downloads': hub_dl,
            'organic_downloads': user_dl,
            'organic_dl_pct': user_dl / total_dl * 100 if total_dl > 0 else 0,
            'hub_dl_pct': hub_dl / total_dl * 100 if total_dl > 0 else 0,
            'bot_dl_pct': bot_dl / total_dl * 100 if total_dl > 0 else 0,
        }
        del _df
    else:
        print("    SKIPPED - missing data")
        return

    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.05)

    # ---- Panel A: Pipeline workflow diagram ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(A) PRIDE Logs Workflow', fontsize=12, fontweight='bold', pad=10)

    # Style definitions
    def draw_box(ax, x, y, w, h, text, color='#EBF5FB', edge='#2980B9', fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # --- Component 1: nf-downloadstats (top section) ---
    # Dashed box around the nf-downloadstats component
    from matplotlib.patches import FancyBboxPatch as FBP
    import matplotlib.patches as mpatches
    nf_rect = mpatches.FancyBboxPatch((0.05, 8.1), 9.4, 1.7, boxstyle='round,pad=0.15',
                                       facecolor='none', edgecolor='#27AE60', linewidth=2.0,
                                       linestyle='--')
    ax.add_patch(nf_rect)
    ax.text(0.3, 9.65, 'nf-downloadstats', fontsize=10, fontweight='bold', color='#27AE60',
            fontstyle='italic')

    # Row 1: Data collection
    draw_box(ax, 0.2, 8.3, 2.2, 1.0, 'PRIDE\nLog Files\n(TSV)', color='#FDEBD0', edge='#E67E22', fontsize=9, bold=True)
    draw_arrow(ax, 2.4, 8.8, 3.0, 8.8)
    draw_box(ax, 3.0, 8.3, 3.0, 1.0, 'Parse, Filter\n& Merge', color='#D5F5E3', edge='#27AE60', fontsize=9)
    draw_arrow(ax, 6.0, 8.8, 6.6, 8.8)
    draw_box(ax, 6.6, 8.3, 2.6, 1.0, 'Parquet\n159M records\n(4.7 GB)', color='#FDEBD0', edge='#E67E22', fontsize=9, bold=True)

    # --- Component 2: DeepLogBot (bottom section) ---
    lg_rect = mpatches.FancyBboxPatch((0.05, -0.15), 9.4, 7.55, boxstyle='round,pad=0.15',
                                       facecolor='none', edgecolor='#2980B9', linewidth=2.0,
                                       linestyle='--')
    ax.add_patch(lg_rect)
    ax.text(0.3, 7.2, 'DeepLogBot', fontsize=10, fontweight='bold', color='#2980B9',
            fontstyle='italic')

    # Row 2: Location aggregation + Feature extraction
    draw_box(ax, 0.3, 5.9, 4.2, 1.0, 'Location Aggregation\ngeographic locations', color='#EBF5FB', edge='#2980B9', fontsize=9)
    draw_arrow(ax, 4.5, 6.4, 5.0, 6.4)
    draw_box(ax, 5.0, 5.9, 4.2, 1.0, 'Feature Extraction\n33 behavioral features\n(temporal, protocol, user dist.)', color='#EBF5FB', edge='#2980B9', fontsize=8.5)

    # Arrows inside DeepLogBot from top to both boxes
    midx = 4.75
    midy = 7.15
    draw_arrow(ax, midx, midy, 2.4, 6.9)
    draw_arrow(ax, midx, midy, 7.1, 6.9)

    # Arrows down to seed selection
    draw_arrow(ax, 2.4, 5.9, 4.75, 5.4)
    draw_arrow(ax, 7.1, 5.9, 4.75, 5.4)

    # Row 3: Seed Selection
    draw_box(ax, 2.0, 4.3, 5.5, 0.9, 'Seed Selection\nOrganic (3-tier) \u00b7 Bot (6-signal) \u00b7 Hub (structural)',
             color='#F5EEF8', edge='#8E44AD', fontsize=9)

    # Arrow down to fusion
    draw_arrow(ax, 4.75, 4.3, 4.75, 3.8)

    # Row 4: Fusion Meta-Learner
    draw_box(ax, 1.5, 2.7, 6.5, 0.9, 'Fusion Meta-Learner\nGradientBoosting (3-class) \u00b7 Platt calibration',
             color='#FADBD8', edge='#E74C3C', fontsize=9, bold=True)

    # Arrow down to hub protection
    draw_arrow(ax, 4.75, 2.7, 4.75, 2.2)

    # Row 5: Hub Protection + Finalize
    draw_box(ax, 2.0, 1.1, 5.5, 0.9, 'Hub Protection & Finalization\nStructural override \u00b7 Insufficient evidence filter',
             color='#D4EFDF', edge='#27AE60', fontsize=8.5)

    # Arrow to final output
    draw_arrow(ax, 4.75, 1.1, 4.75, 0.7)
    draw_box(ax, 1.5, 0.0, 6.5, 0.6, 'Bot-filtered dataset for downstream analysis',
             color='#D5F5E3', edge='#27AE60', fontsize=8.5, bold=True)

    # ---- Panel B: Classification distribution (download share bar chart) ----
    ax2 = fig.add_subplot(gs[0, 1])
    dl_cats = ['Organic', 'Hub', 'Bot']
    dl_vals = [stats['organic_dl_pct'], stats['hub_dl_pct'], stats['bot_dl_pct']]
    cat_colors = [COLORS['organic'], COLORS['hub'], COLORS['bot']]

    bars = ax2.bar(dl_cats, dl_vals, color=cat_colors, edgecolor='black', linewidth=0.5, width=0.65)
    ax2.set_ylabel('Percentage of Total Downloads', fontsize=10)
    ax2.set_title('(B) Full Dataset Classification', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, dl_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add location counts as secondary info inside bars
    loc_counts = [stats['organic_locations'], stats['hub_locations'], stats['bot_locations']]
    total_locs = sum(loc_counts)
    for bar, count in zip(bars, loc_counts):
        pct = count / total_locs * 100
        y_pos = max(bar.get_height() / 2, 5)
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'n={count:,}\n({pct:.1f}% locs)', ha='center', fontsize=9,
                color='white', fontweight='bold')

    plt.savefig(output_dir / 'figure_bot_detection_overview.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_1_pipeline_overview(output_dir):
    """Figure 1: Pipeline overview matching the 5-phase deep architecture.

    Pipeline: Seed Selection -> LLM Seed Refinement -> Fusion Meta-Learner -> Hub Protection -> Finalize.
    """
    print("  Figure 1: Pipeline overview...")
    csv_path = CLASSIFICATION_DIR / 'location_analysis.csv'
    if not csv_path.exists():
        print("    SKIPPED - no classification data")
        return

    # Load classification stats dynamically
    df = pd.read_csv(csv_path, usecols=['behavior_type', 'total_downloads'], low_memory=False)
    total_locs = len(df)
    bot_locs = int((df['behavior_type'] == 'bot').sum())
    hub_locs = int((df['behavior_type'] == 'hub').sum())
    user_locs = int((df['behavior_type'] == 'user').sum())
    insuff_locs = int((df['behavior_type'] == 'insufficient_evidence').sum())
    bot_dl = df.loc[df['behavior_type'] == 'bot', 'total_downloads'].sum()
    hub_dl = df.loc[df['behavior_type'] == 'hub', 'total_downloads'].sum()
    user_dl = df.loc[df['behavior_type'] == 'user', 'total_downloads'].sum()
    total_dl = df['total_downloads'].sum()
    bot_pct = bot_dl / total_dl * 100
    hub_pct = hub_dl / total_dl * 100
    user_pct = user_dl / total_dl * 100
    del df

    def fmt_dl(val):
        if val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.0f}K'
        return str(int(val))

    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    bh = 0.55   # box height
    pad = 0.08  # box corner padding

    def draw_box(x, y, w, h, text, color='#EBF5FB', edge='#2980B9', fontsize=8.5,
                 bold=False, text_color='black'):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
                             facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=text_color,
                linespacing=1.15)

    def draw_arrow(x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                    shrinkA=2, shrinkB=2))

    mid = 7.0  # horizontal center

    # ---- nf-downloadstats section (green dashed box) ----
    nf_rect = mpatches.FancyBboxPatch((0.2, 9.0), 13.1, 1.25, boxstyle='round,pad=0.1',
                                       facecolor='#F0FFF0', edgecolor='#27AE60',
                                       linewidth=1.8, linestyle='--')
    ax.add_patch(nf_rect)
    ax.text(0.5, 10.1, 'nf-downloadstats', fontsize=10, fontweight='bold',
            color='#1B7A3D', fontstyle='italic')

    draw_box(0.5, 9.2, 3.2, bh, 'PRIDE Log Files\n(TSV, 1\u2013237 GB)',
             color='#D5F5E3', edge='#27AE60', fontsize=8.5, bold=True)
    draw_arrow(3.7, 9.2 + bh/2, 4.4, 9.2 + bh/2)
    draw_box(4.4, 9.2, 4.0, bh, 'Parse, Filter & Merge\n(Nextflow + HPC)',
             color='#D5F5E3', edge='#27AE60', fontsize=8.5)
    draw_arrow(8.4, 9.2 + bh/2, 9.1, 9.2 + bh/2)
    draw_box(9.1, 9.2, 3.9, bh, 'Parquet Dataset\n159M records (4.7 GB)',
             color='#D5F5E3', edge='#27AE60', fontsize=8.5, bold=True)

    # Arrow down to DeepLogBot
    draw_arrow(mid, 9.0, mid, 8.6)

    # ---- DeepLogBot section (blue dashed box) ----
    dl_rect = mpatches.FancyBboxPatch((0.2, 0.3), 13.1, 8.1, boxstyle='round,pad=0.1',
                                       facecolor='#F0F8FF', edgecolor='#2980B9',
                                       linewidth=1.8, linestyle='--')
    ax.add_patch(dl_rect)
    ax.text(0.5, 8.2, 'DeepLogBot', fontsize=10, fontweight='bold',
            color='#1A5276', fontstyle='italic')

    # Row 1: Location Aggregation + Feature Extraction
    r1y = 7.5
    draw_box(0.6, r1y, 5.5, bh, f'Location Aggregation\n{total_locs:,} geographic locations',
             color='#D6EAF8', edge='#2980B9', fontsize=8.5)
    draw_arrow(6.1, r1y + bh/2, 7.4, r1y + bh/2)
    draw_box(7.4, r1y, 5.5, bh, 'Feature Extraction\n33 behavioral features per location',
             color='#D6EAF8', edge='#2980B9', fontsize=8.5)

    # Arrow down from center
    draw_arrow(mid, r1y, mid, r1y - 0.3)

    # Row 2: Phase 1 - Seed Selection (compact)
    r2y = 6.35
    sw = 9.0
    sx = mid - sw/2
    draw_box(sx, r2y, sw, bh,
             'Phase 1: Seed Selection\n'
             'Organic (3-tier) \u00b7 Bot (6-signal) \u00b7 Hub (structural + institutional)',
             color='#E8DAEF', edge='#8E44AD', fontsize=8.5)

    draw_arrow(mid, r2y, mid, r2y - 0.3)

    # Row 3: Phase 2 - LLM Seed Refinement
    r3y = 5.25
    lw = 9.0
    lx = mid - lw/2
    draw_box(lx, r3y, lw, bh,
             'Phase 2: LLM Seed Refinement\n'
             'Blind multi-LLM consensus (Claude + Qwen3) \u00b7 '
             '934 validated seed corrections',
             color='#FEF9E7', edge='#F39C12', fontsize=8.5, bold=True)

    draw_arrow(mid, r3y, mid, r3y - 0.3)

    # Row 4: Phase 3 - Fusion Meta-Learner (centered, emphasized)
    r4y = 4.15
    fw = 9.0
    fx = mid - fw/2
    draw_box(fx, r4y, fw, bh,
             'Phase 3: Fusion Meta-Learner\n'
             'GradientBoosting (3-class) \u00b7 Platt-calibrated probabilities \u00b7 '
             'confidence-weighted training',
             color='#FADBD8', edge='#C0392B', fontsize=9, bold=True)

    draw_arrow(mid, r4y, mid, r4y - 0.3)

    # Row 5: Phase 4 - Hub Protection & Finalization (merged)
    r5y = 3.05
    pw = 9.0
    px = mid - pw/2
    r6y = r5y  # alias for output arrows below
    draw_box(px, r5y, pw, bh,
             'Phase 4: Hub Protection & Finalization\n'
             'Institutional hub override \u00b7 Insufficient evidence filter (<3 DL) \u00b7 '
             'Boolean derivation',
             color='#D6EAF8', edge='#2980B9', fontsize=8.5)

    draw_arrow(mid, r5y, mid, r5y - 0.3)

    # Row 7: Output boxes (Bot, Hub, User + insufficient evidence)
    r7y = 0.5
    out_h = 0.65
    box_w = 3.5
    gap = 0.4
    total_w = 3 * box_w + 2 * gap
    x_start = mid - total_w / 2

    # Arrows from center to each output
    draw_arrow(mid - 2.5, r6y - 0.3, x_start + box_w / 2, r7y + out_h)
    draw_arrow(mid, r6y - 0.3, x_start + box_w + gap + box_w / 2, r7y + out_h)
    draw_arrow(mid + 2.5, r6y - 0.3, x_start + 2 * (box_w + gap) + box_w / 2, r7y + out_h)

    # Bot box (red)
    draw_box(x_start, r7y, box_w, out_h,
             f'Bot\n{bot_locs:,} loc. \u00b7 {fmt_dl(bot_dl)} DL ({bot_pct:.1f}%)',
             color='#FADBD8', edge='#E74C3C', fontsize=9, bold=True, text_color='#C0392B')

    # Hub box (blue)
    draw_box(x_start + box_w + gap, r7y, box_w, out_h,
             f'Hub\n{hub_locs:,} loc. \u00b7 {fmt_dl(hub_dl)} DL ({hub_pct:.1f}%)',
             color='#D6EAF8', edge='#2980B9', fontsize=9, bold=True, text_color='#1A5276')

    # User box (green)
    draw_box(x_start + 2 * (box_w + gap), r7y, box_w, out_h,
             f'User\n{user_locs:,} loc. \u00b7 {fmt_dl(user_dl)} DL ({user_pct:.1f}%)',
             color='#D5F5E3', edge='#27AE60', fontsize=9, bold=True, text_color='#1B7A3D')

    # Insufficient evidence annotation
    ax.text(x_start + 2 * (box_w + gap) + box_w + 0.2, r7y + out_h / 2,
            f'+{insuff_locs:,}\ninsufficient\nevidence',
            fontsize=8, color='#888888', fontstyle='italic', va='center')

    plt.savefig(output_dir / 'figure1_pipeline_overview.pdf', format='pdf',
                bbox_inches='tight', dpi=600)
    plt.close()
    print("    OK")


def figure_1_world_map(output_dir):
    """Figure 1: Geographic distribution of PRIDE downloads (bot-filtered)."""
    print("  Figure 1: Geographic distribution...")
    csv_path = ANALYSIS_DIR / 'geographic_by_country.csv'
    if not csv_path.exists():
        print("    SKIPPED - no geographic data")
        return

    df = pd.read_csv(csv_path)
    df = df[~df['country'].str.contains('%{', na=False)]
    df = df[df['country'] != 'Russia']
    top20 = df.head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(top20))
    bars = ax.barh(y_pos, top20['total_downloads'] / 1e6, color='#3498DB', edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['country'])
    ax.invert_yaxis()
    ax.set_xlabel('Total Downloads (millions)')
    ax.set_title('Top 20 Countries by PRIDE Downloads')

    for bar, val in zip(bars, top20['total_downloads']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.2f}M', va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_geographic_distribution.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_1b_regional(output_dir):
    """Figure 1b: Regional distribution pie chart."""
    print("  Figure 1b: Regional distribution...")
    csv_path = ANALYSIS_DIR / 'geographic_by_region.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    region_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#E67E22', '#1ABC9C']

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        df['total_downloads'],
        labels=df['region'],
        autopct='%1.1f%%',
        colors=region_colors[:len(df)],
        startangle=90,
        pctdistance=0.75,
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title('PRIDE Downloads by Region')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1b_regional_distribution.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_2_temporal(output_dir):
    """Figure 2: Downloads over time — stacked bar (user/hub/bot) + growth."""
    print("  Figure 2: Temporal trends...")

    # Try category breakdown first; fall back to simple yearly
    cat_path = ANALYSIS_DIR / 'temporal_yearly_by_category.csv'
    csv_path = ANALYSIS_DIR / 'temporal_yearly.csv'

    has_categories = cat_path.exists()
    if not has_categories and not csv_path.exists():
        print("    SKIPPED")
        return

    if has_categories:
        cat_df = pd.read_csv(cat_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Stacked bar — user / hub / bot downloads per year
    if has_categories:
        # Column may be 'behavior_type' or 'category' depending on analysis version
        cat_col = 'behavior_type' if 'behavior_type' in cat_df.columns else 'category'
        pivot = cat_df.pivot_table(values='total_downloads', index='year',
                                    columns=cat_col, fill_value=0)
        years = pivot.index.astype(int)
        # Ensure columns exist
        for col in ['user', 'hub', 'bot']:
            if col not in pivot.columns:
                pivot[col] = 0

        user_vals = pivot['user'].values / 1e6
        hub_vals = pivot['hub'].values / 1e6

        ax1.bar(years, user_vals, color=COLORS['organic'], edgecolor='white',
                width=0.7, label='Users')
        ax1.bar(years, hub_vals, bottom=user_vals, color=COLORS['hub'],
                edgecolor='white', width=0.7, label='Hubs')

        totals = user_vals + hub_vals
        max_dl = totals.max()
        ax1.set_ylim(0, max_dl * 1.35)
        for i, (yr, total) in enumerate(zip(years, totals)):
            ax1.text(yr, total + max_dl * 0.02, f"{total:.1f}M",
                     ha='center', fontsize=10)
            # Add YoY user growth % annotation
            if i > 0 and user_vals[i - 1] > 0:
                growth_pct = (user_vals[i] - user_vals[i - 1]) / user_vals[i - 1] * 100
                color = '#27AE60' if growth_pct > 0 else '#E74C3C'
                arrow = '\u2191' if growth_pct > 0 else '\u2193'
                ax1.text(yr, total + max_dl * 0.09,
                         f"user {arrow}{growth_pct:+.0f}%",
                         ha='center', fontsize=8, color=color, fontweight='bold')

        ax1.legend(frameon=False, loc='upper left')
    else:
        # Fallback: simple bar
        years = df['year'].astype(int)
        ax1.bar(years, df['total_downloads'] / 1e6, color='#3498DB',
                edgecolor='white', width=0.7)
        max_dl = df['total_downloads'].max() / 1e6
        ax1.set_ylim(0, max_dl * 1.25)
        for _, row in df.iterrows():
            ax1.text(int(row['year']), row['total_downloads'] / 1e6 + max_dl * 0.03,
                    f"{row['total_downloads']/1e6:.1f}M", ha='center', fontsize=11)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Downloads (millions)')
    ax1.set_title('A) Annual Download Volume')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if has_categories:
        ax1.set_xticks(years)
    else:
        ax1.set_xticks(df['year'].astype(int))
    ax1.set_xticklabels(ax1.get_xticks())

    # Panel B: Unique datasets and locations
    if df is not None:
        years_b = df['year'].astype(int)
        ax2b = ax2.twinx()
        l1 = ax2.plot(years_b, df['unique_datasets'] / 1e3, 'o-', color='#E67E22', label='Unique datasets (k)')
        l2 = ax2b.plot(years_b, df['unique_locations'] / 1e3, 's--', color='#9B59B6', label='Unique locations (k)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Unique Datasets (thousands)', color='#E67E22')
        ax2b.set_ylabel('Unique Locations (thousands)', color='#9B59B6')
        ax2.set_title('B) Dataset and Location Growth')
        ax2.spines['top'].set_visible(False)
        ax2.set_xticks(years_b)
        ax2.set_xticklabels(years_b)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_temporal_trends.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_3_algorithm_comparison(output_dir):
    """Figure 3: Bot detection algorithm comparison."""
    print("  Figure 3: Algorithm comparison...")
    csv_path = BENCHMARK_DIR / 'results' / 'method_comparison.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    methods = df['method'].tolist()
    method_colors = [COLORS.get(m, '#999') for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Classification distribution (stacked bars)
    ax = axes[0]
    bot_pcts = df['bot_locations_pct'].values
    hub_pcts = df['hub_locations_pct'].values
    organic_pcts = 100 - bot_pcts - hub_pcts
    x = range(len(methods))
    ax.bar(x, organic_pcts, label='Organic', color=COLORS['organic'])
    ax.bar(x, hub_pcts, bottom=organic_pcts, label='Hub', color=COLORS['hub'])
    ax.bar(x, bot_pcts, bottom=organic_pcts + hub_pcts, label='Bot', color=COLORS['bot'])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Percentage of Locations')
    ax.set_title('A) Classification Distribution')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: F1 scores per category
    ax = axes[1]
    categories = ['bot', 'hub', 'organic']
    x_pos = np.arange(len(methods))
    width = 0.25
    for i, cat in enumerate(categories):
        vals = df[f'{cat}_f1'].values
        ax.bar(x_pos + i * width, vals, width, label=cat.capitalize(), color=COLORS[cat])
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('F1 Score')
    ax.set_title('B) Per-Category F1 Scores')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Macro F1 with confidence intervals
    ax = axes[2]
    macro_f1s = df['macro_f1'].values
    ci_lower = df['macro_f1_ci_lower'].values if 'macro_f1_ci_lower' in df.columns else macro_f1s - 0.03
    ci_upper = df['macro_f1_ci_upper'].values if 'macro_f1_ci_upper' in df.columns else macro_f1s + 0.03
    errors = np.array([macro_f1s - ci_lower, ci_upper - macro_f1s])

    bars = ax.bar(x_pos, macro_f1s, color=method_colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_pos, macro_f1s, yerr=errors, fmt='none', ecolor='black', capsize=5, linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('C) Overall Performance (95% CI)')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, macro_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_algorithm_comparison.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_4_protocols(output_dir):
    """Figure 4: Protocol usage over time (stacked bar chart)."""
    print("  Figure 4: Protocol usage...")
    csv_path = ANALYSIS_DIR / 'protocol_usage.csv'
    if not csv_path.exists() or os.path.getsize(csv_path) < 10:
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return

    # Map raw protocol names to display names
    protocol_names = {
        'http': 'HTTP',
        'ftp': 'FTP',
        'fasp-aspera': 'Aspera',
        'gridftp-globus': 'Globus',
    }
    df['protocol'] = df['protocol'].map(lambda x: protocol_names.get(x, x))

    # Exclude 2020 (only 279 downloads — too sparse)
    df = df[df['year'] >= 2021]

    # Pivot to get protocols as columns
    pivot = df.pivot_table(values='downloads', index='year', columns='protocol', fill_value=0)

    # Order protocols for consistent stacking (exclude Globus)
    protocol_order = [p for p in ['FTP', 'HTTP', 'Aspera'] if p in pivot.columns]
    pivot = pivot[protocol_order]

    protocol_colors = {'HTTP': '#3498DB', 'FTP': '#E67E22', 'Aspera': '#2ECC71'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Absolute download counts (stacked bar)
    years = pivot.index.astype(str)
    bottom = np.zeros(len(years))
    for proto in protocol_order:
        values = pivot[proto].values
        ax1.bar(years, values / 1e6, bottom=bottom / 1e6,
                color=protocol_colors[proto], label=proto, edgecolor='white', linewidth=0.5)
        bottom += values

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Downloads (millions)')
    ax1.set_title('(A) Download Volume by Protocol')
    ax1.legend(title='Protocol', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Monthly protocol breakdown for 2025
    monthly_path = ANALYSIS_DIR / 'protocol_monthly_2025.csv'
    if monthly_path.exists():
        df_m = pd.read_csv(monthly_path)
        df_m['protocol'] = df_m['protocol'].map(lambda x: protocol_names.get(x, x))
        pivot_m = df_m.pivot_table(values='downloads', index='month', columns='protocol', fill_value=0)
        proto_order_m = [p for p in ['FTP', 'Aspera'] if p in pivot_m.columns]
        pivot_m = pivot_m[proto_order_m]

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = pivot_m.index.values
        x_pos = np.arange(len(months))

        bottom_m = np.zeros(len(months))
        for proto in proto_order_m:
            values = pivot_m[proto].values / 1e6
            ax2.bar(x_pos, values, bottom=bottom_m,
                    color=protocol_colors[proto], label=proto, edgecolor='white', linewidth=0.5)
            bottom_m += values

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([month_labels[m - 1] for m in months], rotation=45, ha='right')
        ax2.set_xlabel('Month (2025)')
        ax2.set_ylabel('Downloads (millions)')
        ax2.set_title('(B) FTP & Aspera Monthly Usage in 2025')
        ax2.legend(title='Protocol', frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_protocol_usage.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_5_concentration(output_dir):
    """Figure: Rank-frequency (A), top 20 datasets bar (B), consistency heatmap (C)."""
    print("  Figure 5: Concentration + top datasets + consistency...")
    stats_path = ANALYSIS_DIR / 'concentration_stats.json'
    top_path = ANALYSIS_DIR / 'top_datasets.csv'

    if not stats_path.exists():
        print("    SKIPPED - no concentration data")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    if top_path.exists():
        top_all = pd.read_csv(top_path)
        downloads = top_all['total_downloads'].sort_values(ascending=False).values
    else:
        print("    SKIPPED - no top_datasets.csv")
        return

    # Query yearly data for consistency heatmap (user-only)
    conn = _get_filtered_connection(mode='user_only')
    heatmap_data = None
    if conn is not None and PARQUET_PATH.exists():
        try:
            p = str(PARQUET_PATH).replace("'", "''")
            filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"
            top_ds = conn.execute(f"""
                SELECT accession, COUNT(*) as total FROM read_parquet('{p}')
                WHERE accession IS NOT NULL {filt}
                GROUP BY accession ORDER BY total DESC LIMIT 25
            """).df()
            accessions_sql = ','.join(f"'{a}'" for a in top_ds['accession'])
            yearly = conn.execute(f"""
                SELECT accession, year, COUNT(*) as downloads
                FROM read_parquet('{p}')
                WHERE accession IN ({accessions_sql}) AND year >= 2021 {filt}
                GROUP BY accession, year ORDER BY accession, year
            """).df()
            conn.close()
            pivot = yearly.pivot_table(values='downloads', index='accession', columns='year', fill_value=0)
            pivot = pivot.reindex(top_ds['accession'])
            for y in range(2021, 2026):
                if y not in pivot.columns:
                    pivot[y] = 0
            pivot = pivot[sorted(pivot.columns)]
            heatmap_data = pivot
        except Exception as e:
            print(f"    Warning: heatmap query failed: {e}")

    # Layout: top row = A (rank-freq) + B (top 20 bar), bottom = C (heatmap)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.7, 1],
                           wspace=0.35, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- Panel A: Rank-frequency (log-log) ----
    ranks = np.arange(1, len(downloads) + 1)
    ax1.scatter(ranks, downloads, s=8, alpha=0.5, color='steelblue', edgecolors='none')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset Rank')
    ax1.set_ylabel('Total Downloads')
    ax1.set_title('(A) Rank-Frequency Distribution', fontsize=11, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    top1_idx = int(len(downloads) * 0.01)
    if top1_idx > 0:
        ax1.axvline(x=top1_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(top1_idx * 1.3, downloads[0] * 0.5, 'Top 1%',
                 color='red', fontsize=11, fontweight='bold')

    textstr = (f'Gini = {stats["gini_coefficient"]:.2f}\n'
               f'Top 1%: {stats["top_1pct_downloads_pct"]:.1f}% of DL\n'
               f'Top 10%: {stats["top_10pct_downloads_pct"]:.1f}% of DL\n'
               f'Median: {stats["median_downloads"]:,} DL')
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ---- Panel B: Top 20 datasets horizontal bar ----
    top20 = top_all.nlargest(20, 'total_downloads')
    accessions = top20['accession'].values
    dl_vals = top20['total_downloads'].values
    n_countries = top20['unique_countries'].values if 'unique_countries' in top20.columns else [0] * 20

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    bars = ax2.barh(range(20), dl_vals, color=colors, edgecolor='#333333', linewidth=0.3, alpha=0.85)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(accessions, fontsize=9, fontfamily='monospace')
    ax2.invert_yaxis()
    ax2.set_xlabel('Total Downloads')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(B) Top 20 Most Downloaded Datasets', fontsize=11, fontweight='bold', loc='left')

    for i, (dl, nc) in enumerate(zip(dl_vals, n_countries)):
        label = f'{dl/1e3:.0f}K'
        if nc > 0:
            label += f' ({nc} countries)'
        ax2.text(dl + dl_vals[0] * 0.02, i, label, va='center', fontsize=8, color='#555555')

    # ---- Panel C: Consistency heatmap ----
    if heatmap_data is not None:
        active_years = (heatmap_data > 0).sum(axis=1)
        data = np.log10(heatmap_data.values + 1)
        im = ax3.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        ax3.set_xticks(range(len(heatmap_data.columns)))
        ax3.set_xticklabels(heatmap_data.columns.astype(int), fontsize=11)
        ax3.set_yticks(range(len(heatmap_data.index)))
        ylabels = [f'{acc}  ({active_years[acc]}/{len(heatmap_data.columns)} yrs)'
                   for acc in heatmap_data.index]
        ax3.set_yticklabels(ylabels, fontsize=8.5)
        ax3.set_xlabel('Year')

        cbar = plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.02)
        cbar.set_label('Downloads (log$_{10}$ scale)', fontsize=11)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = int(heatmap_data.iloc[i, j])
                if val > 0:
                    if val >= 1e6:
                        txt = f'{val/1e6:.1f}M'
                    elif val >= 1e3:
                        txt = f'{val/1e3:.0f}K'
                    else:
                        txt = str(val)
                    color = 'white' if data[i, j] > 3.5 else 'black'
                    ax3.text(j, i, txt, ha='center', va='center',
                             fontsize=7, color=color, fontweight='bold')

        ax3.set_title('(C) Top 25 Datasets: Download Consistency — Users Only (2021-2025)',
                       fontsize=11, fontweight='bold', loc='left', pad=15)
    else:
        ax3.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(C) Download Consistency', fontsize=11, fontweight='bold', loc='left')

    plt.savefig(output_dir / 'figure7_dataset_reuse.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_6_top_datasets(output_dir):
    """Figure 6: Top 20 most downloaded datasets (hub-only downloads)."""
    print("  Figure 6: Top datasets (hub-only)...")
    # Prefer hub-only CSV; fall back to user-only
    hub_path = ANALYSIS_DIR / 'top_datasets_hub.csv'
    user_path = ANALYSIS_DIR / 'top_datasets.csv'
    if hub_path.exists():
        csv_path = hub_path
        source_label = 'Hub'
        bar_color = COLORS['hub']
    elif user_path.exists():
        csv_path = user_path
        source_label = 'User'
        bar_color = COLORS['organic']
    else:
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Adaptive scale: use thousands or millions depending on data range
    max_dl = df['total_downloads'].max()
    if max_dl >= 1e6:
        scale, unit, fmt = 1e6, 'millions', lambda v: f'{v/1e6:.2f}M'
    else:
        scale, unit, fmt = 1e3, 'thousands', lambda v: f'{v/1e3:.1f}K'

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['total_downloads'] / scale, color=bar_color, edgecolor='white')

    labels = []
    for _, row in df.iterrows():
        label = row['accession']
        if 'title' in df.columns and pd.notna(row.get('title')):
            title = str(row['title'])[:40]
            label = f"{row['accession']} ({title})"
        labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f'Total {source_label} Downloads ({unit})')
    ax.set_title(f'Top 20 Most Downloaded PRIDE Datasets ({source_label} Downloads)')

    x_margin = max_dl / scale * 0.02
    for bar, val in zip(bars, df['total_downloads']):
        ax.text(bar.get_width() + x_margin, bar.get_y() + bar.get_height() / 2,
                fmt(val), va='center', fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_top_datasets.pdf', format='pdf')
    plt.close()
    print("    OK")


def supplementary_figure_agreement(output_dir):
    """Supplementary: Algorithm agreement/disagreement analysis."""
    print("  Supp Figure: Agreement analysis...")
    json_path = BENCHMARK_DIR / 'results' / 'agreement_matrix.json'
    if not json_path.exists():
        print("    SKIPPED")
        return

    with open(json_path) as f:
        agreement = json.load(f)

    methods = ['rules', 'deep']
    categories = ['bot', 'hub', 'organic']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Overall agreement + kappa heatmap
    ax = axes[0]
    matrix = np.ones((2, 2))
    for pair, data in agreement.items():
        m1, m2 = pair.split('_vs_')
        i, j = methods.index(m1), methods.index(m2)
        matrix[i, j] = data['overall_agreement']
        matrix[j, i] = data['overall_agreement']

    im = ax.imshow(matrix, cmap='YlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticklabels([m.upper() for m in methods])
    ax.set_title('A) Pairwise Agreement')

    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f'{matrix[i, j]:.1%}', ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: Per-category agreement bars
    ax = axes[1]
    x_pos = np.arange(len(list(agreement.keys())))
    width = 0.25
    for i, cat in enumerate(categories):
        vals = [data['category_agreement'].get(cat, 0) for data in agreement.values()]
        ax.bar(x_pos + i * width, vals, width, label=cat.capitalize(), color=COLORS[cat])

    ax.set_xticks(x_pos + width)
    pair_labels = [p.replace('_vs_', ' vs ').upper() for p in agreement.keys()]
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Category Agreement')
    ax.set_title('B) Per-Category Agreement')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_agreement.pdf', format='pdf')
    plt.close()
    print("    OK")


def _get_filtered_connection(mode='user_only'):
    """Get DuckDB connection with filtered locations.

    Args:
        mode: 'user_only' (exclude bot+hub), 'hub_only', 'no_bot' (exclude bot only),
              or 'all' (no filter).
    """
    labels_path = CLASSIFICATION_DIR / 'location_analysis.csv'
    if not labels_path.exists():
        return None
    labels_df = pd.read_csv(labels_path, low_memory=False)
    if 'behavior_type' not in labels_df.columns:
        return None

    if mode == 'user_only':
        filtered = labels_df[labels_df['behavior_type'] == 'user'][['geo_location']].drop_duplicates()
    elif mode == 'hub_only':
        filtered = labels_df[labels_df['behavior_type'] == 'hub'][['geo_location']].drop_duplicates()
    elif mode == 'no_bot':
        filtered = labels_df[labels_df['behavior_type'] != 'bot'][['geo_location']].drop_duplicates()
    else:
        filtered = labels_df[['geo_location']].drop_duplicates()

    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='4GB'")
    tmp = os.path.abspath('./duckdb-tmp/')
    os.makedirs(tmp, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{tmp}'")
    conn.execute("PRAGMA threads=2")
    conn.register('_cl', filtered)
    conn.execute("CREATE TEMP TABLE clean_locations AS SELECT * FROM _cl")
    return conn


def _query_country_yearly_trends(conn, countries):
    """Query yearly download counts for a list of countries (bot-filtered)."""
    p = str(PARQUET_PATH).replace("'", "''")
    filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"
    countries_sql = ','.join(f"'{c}'" for c in countries)
    df = conn.execute(f"""
        SELECT country, year, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE country IN ({countries_sql}) AND year >= 2021 {filt}
        GROUP BY country, year ORDER BY country, year
    """).df()
    return df


def _draw_bubble_panel(ax, df, title, show_colorbar=True,
                       priority_labels=None, max_labels=15, exclude_labels=None,
                       bold_labels=True):
    """Draw a country bubble chart on the given axes.

    Parameters
    ----------
    priority_labels : list or None
        Countries that must always be labeled (shown first).
    max_labels : int
        Maximum number of labels to show in total.
    exclude_labels : set or None
        Countries to never label.
    """
    dl_per_user = df['dl_per_user'].values
    size_raw = np.clip(dl_per_user, 1, 2000)
    sizes = (size_raw / size_raw.max()) * 800 + 15

    scatter = ax.scatter(
        df['unique_users'], df['total_downloads'],
        s=sizes,
        c=dl_per_user, cmap='viridis',
        alpha=0.75, edgecolors='black', linewidth=0.5,
        norm=plt.matplotlib.colors.LogNorm(vmin=max(dl_per_user.min(), 1), vmax=dl_per_user.max()),
        zorder=3,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Unique Users (log scale)')
    ax.set_ylabel('Total Downloads (log scale)')

    if show_colorbar:
        plt.colorbar(scatter, ax=ax, label='Downloads per User', fraction=0.03, pad=0.04)

    if priority_labels is None:
        priority_labels = []
    if exclude_labels is None:
        exclude_labels = set()

    # Build label list: priority countries first, then top by downloads
    priority_set = set(priority_labels)
    sorted_df = df.sort_values('total_downloads', ascending=False)
    countries_to_label = list(priority_labels)  # keep priority order
    for _, row in sorted_df.iterrows():
        name = row['country']
        if name not in priority_set and name not in exclude_labels:
            countries_to_label.append(name)
    countries_to_label = countries_to_label[:max_labels]
    label_set = set(countries_to_label)

    labeled = []
    for _, row in sorted_df.iterrows():
        x, y = row['unique_users'], row['total_downloads']
        name = row['country']
        if name not in label_set:
            continue
        ha, x_off, y_off = 'left', 8, 4
        if x > 50000:
            ha, x_off = 'right', -8
        # Check overlap with already placed labels
        for lx, ly in labeled:
            if abs(np.log10(x) - np.log10(lx)) < 0.2 and abs(np.log10(y) - np.log10(ly)) < 0.12:
                y_off += 10
        fontsize = 9.5 if row['total_downloads'] > 500000 else 8.5
        fontweight = 'bold' if bold_labels and row['total_downloads'] > 1000000 else 'normal'
        ax.annotate(
            name, (x, y),
            fontsize=fontsize, fontweight=fontweight,
            ha=ha, va='bottom',
            xytext=(x_off, y_off), textcoords='offset points',
        )
        labeled.append((x, y))

    legend_sizes = [10, 100, 500]
    legend_bubbles = []
    for val in legend_sizes:
        s = (np.clip(val, 1, 2000) / np.clip(dl_per_user, 1, 2000).max()) * 800 + 15
        legend_bubbles.append(
            ax.scatter([], [], s=s, c='gray', alpha=0.5, edgecolors='black', linewidth=0.5)
        )
    ax.legend(
        legend_bubbles, [f'{v}' for v in legend_sizes],
        title='DL/User (size)', loc='upper left',
        frameon=True, framealpha=0.9, fontsize=10, title_fontsize=10,
        labelspacing=1.5, borderpad=1.2,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, which='both')
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')


def figure_7_bubble_chart(output_dir):
    """Figure 7: Country bubble chart (A user, C hub) + European & LMIC trends."""
    print("  Figure 7: Country bubble chart with regional trends...")
    csv_path = ANALYSIS_DIR / 'country_bubble_data.csv'
    if not csv_path.exists():
        print("    SKIPPED - no bubble data")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return
    df = df[df['country'] != 'Russia']

    # Load hub bubble data if available
    hub_path = ANALYSIS_DIR / 'country_bubble_hub_data.csv'
    hub_df = pd.read_csv(hub_path) if hub_path.exists() else None
    if hub_df is not None:
        hub_df = hub_df[~hub_df['country'].str.contains('%{', na=False)]
        hub_df = hub_df[hub_df['country'] != 'Russia']

    # Query yearly trends from parquet (user-only)
    conn = _get_filtered_connection(mode='user_only')
    has_trends = conn is not None and PARQUET_PATH.exists()
    eu_df = lmic_df = None
    if has_trends:
        try:
            eu_df = _query_country_yearly_trends(conn, EUROPEAN_COUNTRIES)
            lmic_df = _query_country_yearly_trends(conn, LMIC_COUNTRIES)
            conn.close()
        except Exception as e:
            print(f"    Warning: could not query trends: {e}")
            has_trends = False

    # Layout: 2x2 grid — A (user bubble), B (EU trends), C (hub bubble), D (LMIC trends)
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_bubble = fig.add_subplot(gs[0, 0])   # top-left
    ax_europe = fig.add_subplot(gs[0, 1])   # top-right
    ax_hub = fig.add_subplot(gs[1, 0])      # bottom-left
    ax_lmic = fig.add_subplot(gs[1, 1])     # bottom-right

    # ---- Panel A: User bubble chart (select ~15 labels to avoid overlap) ----
    _draw_bubble_panel(ax_bubble, df, '(A) Downloads vs. Users by Country',
                       priority_labels=['United States', 'United Kingdom', 'Germany',
                                        'China', 'Canada', 'Japan', 'France',
                                        'India', 'Australia', 'Brazil'],
                       max_labels=15)

    # ---- Panel B: European trends ----
    if has_trends and eu_df is not None and not eu_df.empty:
        eu_colors = plt.cm.tab20(np.linspace(0, 1, len(EUROPEAN_COUNTRIES)))
        for i, country in enumerate(EUROPEAN_COUNTRIES):
            cdf = eu_df[eu_df['country'] == country].sort_values('year')
            if len(cdf) > 0:
                ax_europe.plot(cdf['year'], cdf['downloads'] / 1e6, 'o-',
                               label=country, linewidth=1.3, markersize=4,
                               color=eu_colors[i])
        ax_europe.set_xlabel('Year')
        ax_europe.set_ylabel('Downloads (millions)')
        ax_europe.set_title('(B) European Countries', fontsize=11, fontweight='bold', loc='left')
        ax_europe.legend(loc='upper left', fontsize=8, ncol=3, frameon=False)
        ax_europe.spines['top'].set_visible(False)
        ax_europe.spines['right'].set_visible(False)
        ax_europe.grid(True, alpha=0.2)
        ax_europe.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax_europe.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax_europe.transAxes)
        ax_europe.set_title('(B) European Countries', fontsize=11, fontweight='bold', loc='left')

    # ---- Panel C: LMIC trends ----
    if has_trends and lmic_df is not None and not lmic_df.empty:
        top_lmic = lmic_df.groupby('country')['downloads'].sum().nlargest(15).index.tolist()
        lmic_colors = plt.cm.tab20(np.linspace(0, 1, len(top_lmic)))
        for i, country in enumerate(top_lmic):
            cdf = lmic_df[lmic_df['country'] == country].sort_values('year')
            if len(cdf) > 0:
                ax_lmic.plot(cdf['year'], cdf['downloads'] / 1e3, 'o-',
                             label=country, linewidth=1.3, markersize=4,
                             color=lmic_colors[i])
        ax_lmic.set_xlabel('Year')
        ax_lmic.set_ylabel('Downloads (thousands)')
        ax_lmic.set_title('(D) Low/Middle Income Countries', fontsize=11, fontweight='bold', loc='left')
        ax_lmic.legend(loc='upper left', fontsize=8, ncol=3, frameon=False)
        ax_lmic.spines['top'].set_visible(False)
        ax_lmic.spines['right'].set_visible(False)
        ax_lmic.grid(True, alpha=0.2)
        ax_lmic.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax_lmic.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax_lmic.transAxes)
        ax_lmic.set_title('(D) Low/Middle Income Countries', fontsize=11, fontweight='bold', loc='left')

    # ---- Panel C: Hub-only bubble chart ----
    if hub_df is not None and not hub_df.empty:
        _draw_bubble_panel(ax_hub, hub_df, '(C) Hub Downloads by Country',
                           priority_labels=['Belgium', 'United States', 'China',
                                            'Denmark', 'United Kingdom', 'Germany',
                                            'Japan', 'South Korea', 'France'],
                           max_labels=12, exclude_labels={'Panama'},
                           bold_labels=False)
    else:
        ax_hub.text(0.5, 0.5, 'Hub data not available', ha='center', va='center', transform=ax_hub.transAxes)
        ax_hub.set_title('(C) Hub Downloads by Country', fontsize=11, fontweight='bold', loc='left')

    plt.savefig(output_dir / 'figure8_country_bubble.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_dataset_consistency(output_dir):
    """Figure: Top dataset download consistency heatmap over years."""
    print("  Dataset consistency heatmap...")
    conn = _get_filtered_connection(mode='user_only')
    if conn is None or not PARQUET_PATH.exists():
        print("    SKIPPED - no data")
        return

    p = str(PARQUET_PATH).replace("'", "''")
    filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"

    try:
        # Get top 25 datasets by total downloads (bot-filtered)
        top_ds = conn.execute(f"""
            SELECT accession, COUNT(*) as total FROM read_parquet('{p}')
            WHERE accession IS NOT NULL {filt}
            GROUP BY accession ORDER BY total DESC LIMIT 25
        """).df()

        accessions_sql = ','.join(f"'{a}'" for a in top_ds['accession'])
        yearly = conn.execute(f"""
            SELECT accession, year, COUNT(*) as downloads
            FROM read_parquet('{p}')
            WHERE accession IN ({accessions_sql}) AND year >= 2021 {filt}
            GROUP BY accession, year ORDER BY accession, year
        """).df()
        conn.close()
    except Exception as e:
        print(f"    Error querying: {e}")
        return

    pivot = yearly.pivot_table(values='downloads', index='accession', columns='year', fill_value=0)
    # Reorder by total downloads (top first)
    pivot = pivot.reindex(top_ds['accession'])
    # Ensure all years present
    for y in range(2021, 2026):
        if y not in pivot.columns:
            pivot[y] = 0
    pivot = pivot[sorted(pivot.columns)]

    # Count years with >0 downloads for each dataset
    active_years = (pivot > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.log10(pivot.values + 1)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=10)
    ax.set_yticks(range(len(pivot.index)))

    # Label with accession + active years
    ylabels = [f'{acc}  ({active_years[acc]}/{len(pivot.columns)} yrs)'
               for acc in pivot.index]
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Dataset Accession')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Downloads (log$_{10}$ scale)', fontsize=10)

    # Annotate cells with actual download counts
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(pivot.iloc[i, j])
            if val > 0:
                # Format: K for thousands, M for millions
                if val >= 1e6:
                    txt = f'{val/1e6:.1f}M'
                elif val >= 1e3:
                    txt = f'{val/1e3:.0f}K'
                else:
                    txt = str(val)
                color = 'white' if data[i, j] > 3.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=6, color=color, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_dataset_consistency.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_filetype_by_region(output_dir):
    """Figure: File type download patterns by region."""
    print("  File type by region figure...")
    csv_path = ANALYSIS_DIR / 'filetype_by_region.csv'
    if not csv_path.exists():
        print("    SKIPPED - no filetype data (run analysis first)")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return

    # Order categories by importance for the narrative
    cat_order = ['raw', 'processed_spectra', 'result', 'tabular', 'database', 'metadata', 'compressed', 'other']
    cat_labels = {
        'raw': 'Raw instrument\nfiles',
        'processed_spectra': 'Processed\nspectra',
        'result': 'Search\nresults',
        'tabular': 'Tabular\n(csv/tsv/xlsx)',
        'database': 'Sequence\ndatabases',
        'metadata': 'Metadata\n(xml/sdrf)',
        'compressed': 'Compressed\narchives',
        'other': 'Other',
    }
    cat_colors = {
        'raw': '#E74C3C',
        'processed_spectra': '#E67E22',
        'result': '#F1C40F',
        'tabular': '#2ECC71',
        'database': '#1ABC9C',
        'metadata': '#3498DB',
        'compressed': '#9B59B6',
        'other': '#BDC3C7',
    }

    region_order = ['East Asia', 'North America', 'Europe', 'LMIC']
    region_labels = {'East Asia': 'East Asia', 'North America': 'N. America',
                     'Europe': 'Europe', 'LMIC': 'LMIC'}

    # Compute percentages
    region_totals = df.groupby('region')['downloads'].sum()
    df['pct'] = df.apply(lambda r: r['downloads'] / region_totals[r['region']] * 100, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.4, 1]})

    # ---- Panel A: Grouped bar chart ----
    ax = axes[0]
    n_regions = len(region_order)
    n_cats = len(cat_order)
    bar_width = 0.18
    x = np.arange(n_cats)

    for i, region in enumerate(region_order):
        rdf = df[df['region'] == region]
        vals = []
        for cat in cat_order:
            v = rdf.loc[rdf['file_category'] == cat, 'pct']
            vals.append(v.values[0] if len(v) > 0 else 0)
        offset = (i - n_regions / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=region_labels[region],
                      color=plt.cm.Set2(i / n_regions), edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([cat_labels[c] for c in cat_order], fontsize=10)
    ax.set_ylabel('Percentage of Downloads (%)')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('(A) File Type Downloads by Region (Users Only)', fontsize=11, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.2)

    # ---- Panel B: Stacked horizontal bars for key categories only ----
    ax2 = axes[1]
    # Aggregate into: Raw/Processed (reanalysis) vs Result/Tabular (lightweight) vs Other
    def agg_type(cat):
        if cat in ('raw', 'processed_spectra'):
            return 'Raw + Processed\n(full reanalysis)'
        elif cat in ('result', 'tabular', 'database', 'metadata'):
            return 'Results + Metadata\n(lightweight reuse)'
        else:
            return 'Archives + Other'

    df['agg_type'] = df['file_category'].apply(agg_type)
    agg = df.groupby(['region', 'agg_type'])['downloads'].sum().reset_index()
    agg_totals = agg.groupby('region')['downloads'].sum()
    agg['pct'] = agg.apply(lambda r: r['downloads'] / agg_totals[r['region']] * 100, axis=1)

    agg_order = ['Raw + Processed\n(full reanalysis)', 'Results + Metadata\n(lightweight reuse)', 'Archives + Other']
    agg_colors = ['#E74C3C', '#3498DB', '#BDC3C7']

    y_pos = np.arange(len(region_order))
    left = np.zeros(len(region_order))

    for j, atype in enumerate(agg_order):
        vals = []
        for region in region_order:
            v = agg.loc[(agg['region'] == region) & (agg['agg_type'] == atype), 'pct']
            vals.append(v.values[0] if len(v) > 0 else 0)
        vals = np.array(vals)
        bars = ax2.barh(y_pos, vals, left=left, color=agg_colors[j],
                        edgecolor='white', linewidth=0.5, label=atype.replace('\n', ' '))
        # Label percentages
        for k, (v, l) in enumerate(zip(vals, left)):
            if v > 5:
                ax2.text(l + v / 2, k, f'{v:.0f}%', ha='center', va='center',
                         fontsize=11, fontweight='bold', color='white')
        left += vals

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([region_labels[r] for r in region_order], fontsize=11)
    ax2.set_xlabel('Percentage of Downloads (%)')
    ax2.legend(fontsize=8, loc='lower right', frameon=True, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(B) Reanalysis vs. Lightweight Reuse', fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_filetype_by_region.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_hub_distribution(output_dir):
    """Figure: Hub geographic distribution and characteristics."""
    print("  Hub distribution figure...")
    csv_path = CLASSIFICATION_DIR / 'location_analysis.csv'
    if not csv_path.exists():
        print("    SKIPPED - no classification data")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    if 'behavior_type' in df.columns:
        hubs = df[df['behavior_type'] == 'hub'].copy()
    elif 'final_label' in df.columns:
        hubs = df[df['final_label'] == 'hub'].copy()
    elif 'is_hub' in df.columns:
        hubs = df[df['is_hub'] == True].copy()
    else:
        print("    SKIPPED - no label column")
        return
    if hubs.empty:
        print("    SKIPPED - no hubs")
        return
    hubs = hubs[hubs['country'] != 'Russia']

    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # ---- Panel A: World scatter of hub locations ----
    lats, lons = [], []
    for _, row in hubs.iterrows():
        try:
            parts = str(row['geo_location']).split(',')
            lat, lon = float(parts[0]), float(parts[1])
            lats.append(lat)
            lons.append(lon)
        except (ValueError, IndexError):
            lats.append(np.nan)
            lons.append(np.nan)
    hubs['lat'] = lats
    hubs['lon'] = lons
    valid = hubs.dropna(subset=['lat', 'lon'])

    dl_vals = valid['total_downloads'].values
    sizes = np.clip(dl_vals / dl_vals.max() * 300, 10, 300)

    ax_map.scatter(valid['lon'], valid['lat'], s=sizes,
                   c='#3498DB', alpha=0.6, edgecolors='navy', linewidth=0.4, zorder=3)

    # Simple world outline
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-60, 85)
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.axhline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax_map.axvline(0, color='gray', linewidth=0.3, alpha=0.5)
    # Continental outlines via grid
    ax_map.grid(True, alpha=0.15)
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)
    ax_map.set_title('(A) Hub Locations Worldwide', fontsize=11, fontweight='bold', loc='left')

    # Size legend
    for dl, label in [(10000, '10K'), (100000, '100K'), (500000, '500K')]:
        s = np.clip(dl / dl_vals.max() * 300, 10, 300)
        ax_map.scatter([], [], s=s, c='#3498DB', alpha=0.6, edgecolors='navy',
                       linewidth=0.4, label=label)
    ax_map.legend(title='Downloads', loc='lower left', fontsize=9, title_fontsize=9,
                  frameon=True, framealpha=0.9, labelspacing=1.2)

    # ---- Panel B: Top 15 countries by hub count ----
    country_counts = hubs['country'].value_counts().head(15)
    country_downloads = hubs.groupby('country')['total_downloads'].sum()

    colors_bar = ['#3498DB'] * len(country_counts)
    bars = ax_bar.barh(range(len(country_counts)), country_counts.values, color=colors_bar,
                       edgecolor='navy', linewidth=0.3, alpha=0.8)
    ax_bar.set_yticks(range(len(country_counts)))
    ax_bar.set_yticklabels(country_counts.index, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Number of Hubs')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_title('(B) Hubs per Country', fontsize=11, fontweight='bold', loc='left')

    # Annotate bars with hub count and download volume
    for i, (country, count) in enumerate(country_counts.items()):
        dl = country_downloads.get(country, 0)
        dl_label = f'{dl/1e6:.1f}M DL' if dl >= 100000 else f'{dl/1e3:.0f}K DL'
        ax_bar.text(count + 0.5, i, f'{count} ({dl_label})', va='center', fontsize=8, color='gray')

    plt.savefig(output_dir / 'figure2_hub_geography.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 70)

    figure_1_pipeline_overview(FIGURES_DIR)
    figure_bot_detection_overview(FIGURES_DIR)
    figure_1_world_map(FIGURES_DIR)
    figure_1b_regional(FIGURES_DIR)
    figure_2_temporal(FIGURES_DIR)
    figure_3_algorithm_comparison(FIGURES_DIR)
    figure_4_protocols(FIGURES_DIR)
    figure_5_concentration(FIGURES_DIR)
    figure_6_top_datasets(FIGURES_DIR)
    figure_7_bubble_chart(FIGURES_DIR)
    figure_hub_distribution(FIGURES_DIR)
    figure_filetype_by_region(FIGURES_DIR)
    supplementary_figure_agreement(FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    # List output
    for f in sorted(FIGURES_DIR.glob('*.pdf')):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
