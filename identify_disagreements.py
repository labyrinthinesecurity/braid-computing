#!/usr/bin/python3
"""
Identify (SCC, WAR) pairs where Burau LE and the abelian rate disagree
at regime level.

Loads pre-computed data from a CSV produced by burau.py (default:
1000sccs.csv).  Thresholds are derived from the corpus itself:

  LE_THRESH   = median of per-SCC mean(lyapunov_exponent)
  RATE_THRESH = median of per-SCC mean(le_counting)

Both medians are computed fresh each run so the analysis is self-contained
and does not depend on the old hard-coded 82-SCC value (0.4619).

Usage:
    python identify_disagreements.py [csv_path]

csv_path defaults to 1000sccs.csv.  The CSV must have at least the
columns: SCC_ID, lyapunov_exponent, le_counting, dws.
"""
import sys
import math
import numpy as np
import pandas as pd
from collections import Counter

CSV_DEFAULT = 'scc_war_ratios.csv'
REGIME_NAMES = {0: 'focused', 1: 'dispersed'}


def main(csv_path=CSV_DEFAULT):
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        sys.exit(f"CSV not found: {csv_path}\n"
                 "Run burau.py on the full topology set first.")

    required = {'SCC_ID', 'lyapunov_exponent', 'le_counting', 'dws'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")

    n_rows = len(df)
    n_sccs = df['SCC_ID'].nunique()
    print(f"Loaded {n_rows} (SCC, WAR) rows across {n_sccs} SCCs from {csv_path}\n")

    # ── Derive thresholds from corpus ─────────────────────────────────────────
    # Per-SCC mean LE — one value per topology, then take corpus median.
    # Matches the paper's θ = median(mean LE per SCC), now over the full corpus.
    per_scc_le   = df.groupby('SCC_ID')['lyapunov_exponent'].mean()
    per_scc_rate = df.groupby('SCC_ID')['le_counting'].mean()

    LE_THRESH   = float(np.median(per_scc_le))
    RATE_THRESH = float(np.median(per_scc_rate))

    print(f"Derived thresholds (corpus medians over {n_sccs} SCCs):")
    print(f"  LE_THRESH   = {LE_THRESH:.4f}  (was 0.4619 on 82-SCC corpus)")
    print(f"  RATE_THRESH = {RATE_THRESH:.4f}")
    print()

    # ── Classify each row ─────────────────────────────────────────────────────
    df['le_regime']   = (df['lyapunov_exponent'] >= LE_THRESH).astype(int)
    df['rate_regime'] = (df['le_counting']        >= RATE_THRESH).astype(int)
    df['disagree']    = df['le_regime'] != df['rate_regime']

    # ── Summary counts ────────────────────────────────────────────────────────
    n_disagree = int(df['disagree'].sum())
    pct        = 100 * n_disagree / n_rows
    print(f"{'SCC_ID':>7}  {'WAR':>25}  {'LE':>8}  {'LE_regime':>10}  "
          f"{'rate':>8}  {'rate_regime':>12}  direction")
    print("-" * 110)

    disagree_rows = df[df['disagree']].copy()

    # Build WAR tuple string if W-columns present
    w_cols = [c for c in df.columns if c.startswith('W') and c[1:].isdigit()]
    w_cols_sorted = sorted(w_cols, key=lambda c: int(c[1:]))

    directions = []
    for _, row in disagree_rows.iterrows():
        lr = int(row['le_regime'])
        rr = int(row['rate_regime'])
        direction = (f"LE={REGIME_NAMES[lr]}, rate={REGIME_NAMES[rr]}")

        war_str = (str(tuple(int(row[c]) for c in w_cols_sorted))
                   if w_cols_sorted else 'n/a')

        print(f"{int(row['SCC_ID']):>7}  {war_str:>25}  "
              f"{row['lyapunov_exponent']:>8.4f}  {REGIME_NAMES[lr]:>10}  "
              f"{row['le_counting']:>8.4f}  {REGIME_NAMES[rr]:>12}  {direction}")
        directions.append(direction)

    print(f"\nTotal disagreements: {n_disagree} / {n_rows} ({pct:.2f}%)")

    dir_counts = Counter(directions)
    for direction, count in sorted(dir_counts.items()):
        pct_dir = 100 * count / n_disagree if n_disagree else 0
        print(f"  {direction}: {count}  ({pct_dir:.0f}% of disagreements)")

    # Direction check: per paper the bias is one-directional
    le_focused_rate_dispersed = dir_counts.get('LE=focused, rate=dispersed', 0)
    le_dispersed_rate_focused = dir_counts.get('LE=dispersed, rate=focused', 0)
    if le_dispersed_rate_focused == 0 and le_focused_rate_dispersed > 0:
        print("\n✓ Bias is one-directional: rate always over-calls dispersed")
    elif le_focused_rate_dispersed == 0 and le_dispersed_rate_focused > 0:
        print("\n✗ Reversed bias: rate under-calls dispersed (unexpected)")
    elif le_focused_rate_dispersed > 0 and le_dispersed_rate_focused > 0:
        print(f"\n✗ Bias is two-directional ({le_focused_rate_dispersed} vs "
              f"{le_dispersed_rate_focused}) — abelian over-counting not systematic")

    # Per-SCC disagreement summary
    disagree_sccs = [int(x) for x in sorted(disagree_rows['SCC_ID'].unique())]
    print(f"\nDistinct SCCs with at least one disagreement: "
          f"{len(disagree_sccs)} / {n_sccs}")
    print(f"  SCC IDs: {disagree_sccs}")

    # Per-SCC disagreement rate
    per_scc_disagree = df.groupby('SCC_ID')['disagree'].mean()
    print(f"\nPer-SCC disagreement rate:")
    print(f"  Mean:   {per_scc_disagree.mean():.3f}")
    print(f"  Max:    {per_scc_disagree.max():.3f}  "
          f"(SCC {per_scc_disagree.idxmax()})")
    print(f"  SCCs with >10% disagreement: "
          f"{int((per_scc_disagree > 0.10).sum())}")

    # Regime distribution
    print(f"\nRegime distribution (Burau LE):")
    focused   = int((df['le_regime'] == 0).sum())
    dispersed = int((df['le_regime'] == 1).sum())
    print(f"  focused:   {focused} rows  ({100*focused/n_rows:.1f}%)")
    print(f"  dispersed: {dispersed} rows  ({100*dispersed/n_rows:.1f}%)")

    focused_sccs   = int((per_scc_le < LE_THRESH).sum())
    dispersed_sccs = int((per_scc_le >= LE_THRESH).sum())
    print(f"  focused SCCs:   {focused_sccs}")
    print(f"  dispersed SCCs: {dispersed_sccs}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    main(csv_path)
