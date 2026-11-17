"""Report baseline MAE and Spearman IC from saved 1h/4h predictions. No fitting."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Completed run folder; defaults to latest completed run.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'experiments/short_horizon_median_v1'
    folder = args.run_dir
    if folder is None:
        candidates = sorted(p.parent for p in root.glob('*/COMPLETE.json'))
        if not candidates:
            raise SystemExit(f'No completed runs found in {root}. Supply --run-dir.')
        folder = candidates[-1]
    print(f'Reading: {folder.resolve()}\n')
    rows = []
    for hours in (1, 4):
        path = folder / f'predictions_{hours}h.csv'
        frame = pd.read_csv(path)
        for name in ('mean_baseline', 'median_baseline'):
            truth = frame['truth_log_bps'].to_numpy(dtype=float)
            pred = frame[name].to_numpy(dtype=float)
            if len(truth) < 2 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
                raise ValueError(f'{path}: need at least two rows, all finite; no rows silently dropped.')
            unique = int(np.unique(pred).size)
            if unique == 1:
                ic, reason = np.nan, 'Undefined: constant predictions'
            elif np.unique(truth).size == 1:
                ic, reason = np.nan, 'Undefined: constant actual values'
            else:
                ic = float(spearmanr(truth, pred).statistic)
                reason = 'Calculated from saved row-level predictions'
            rows.append(dict(horizon_hours=hours, baseline=name, rows=len(truth),
                             mae_log_bps=float(np.mean(np.abs(truth - pred))),
                             spearman_ic=ic, unique_predictions=unique, explanation=reason))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print('\nNaN means IC is undefined, not zero. No models were fitted and no files were changed.')


if __name__ == '__main__':
    main()
