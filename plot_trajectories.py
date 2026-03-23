"""
Plot car trajectories from CSV files.

CSV format (space or comma separated):
  step  x  y
  0     0  0
  1     0.132  -0.0005
  ...

Usage:
  python plot_trajectories.py                        # auto-detect *.csv in current dir
  python plot_trajectories.py a.csv b.csv c.csv      # specify files explicitly
  python plot_trajectories.py --dir /path/to/csvs    # specify directory
"""

import sys
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_csv(filepath):
    """Load CSV/TSV with flexible separator; skip corrupted header names."""
    # Try common separators
    for sep in [r'\s+', ',', '\t', ';']:
        try:
            df = pd.read_csv(filepath, sep=sep, engine='python')
            if df.shape[1] >= 3:
                # Rename columns to step/x/y regardless of original names
                cols = list(df.columns)
                df = df.rename(columns={cols[0]: 'step', cols[1]: 'x', cols[2]: 'y'})
                df = df[['step', 'x', 'y']].apply(pd.to_numeric, errors='coerce').dropna()
                return df
        except Exception:
            continue
    raise ValueError(f"Cannot parse file: {filepath}")


def plot_trajectories(files, output=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = cm.tab10(np.linspace(0, 1, len(files)))

    for i, filepath in enumerate(files):
        label = os.path.splitext(os.path.basename(filepath))[0]
        df = load_csv(filepath)
        ax.plot(df['x'], df['y'], color=colors[i], linewidth=1.5, label=label)
        # Mark start point
        ax.scatter(df['x'].iloc[0], df['y'].iloc[0], color=colors[i],
                   marker='o', s=60, zorder=5)
        # Mark end point
        ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color=colors[i],
                   marker='s', s=60, zorder=5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Car Trajectories', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot car trajectories from CSV files')
    parser.add_argument('files', nargs='*', help='CSV files to plot')
    parser.add_argument('--dir', default='.', help='Directory to search for CSV files')
    parser.add_argument('--output', '-o', default=None,
                        help='Save figure to file (e.g. trajectories.png)')
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.dir, '*.csv')))

    if not files:
        print(f"No CSV files found in '{args.dir}'. "
              "Pass file paths directly or use --dir.")
        sys.exit(1)

    print(f"Plotting {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    plot_trajectories(files, output=args.output)


if __name__ == '__main__':
    main()
