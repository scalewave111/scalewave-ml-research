"""User-run frozen 1h/4h magnitude median regressions; no import-time execution.

Run: python Code/fit_short_horizon_median.py --features pair
Optional: --features eight (includes derived seven-day realized volatility).
Requires numpy, pandas, scipy, scikit-learn, joblib, duckdb.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import QuantileRegressor

WORKSPACE = Path(__file__).resolve().parents[1]
WEEK1 = WORKSPACE.parent / 'Week 1'
sys.path.insert(0, str(WEEK1 / 'Code'))
import canonical_return_features as cr

PAIR = [
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
]
EIGHT = PAIR + [
    'PRICE__vol_ratio_4h_24h',
    'F4Z__liquidation__venue_breadth_percentile__1h__value',
    'PRICE__realized_vol_168h',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
    'F4Z__funding__funding_persistence_percentile__5m__value',
    'F4Z__basis__basis_level_rz__1h__value',
]
CUTOFF = pd.Timestamp('2024-01-01', tz='UTC')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def preprocessing(frame):
    median = frame.median()
    require(median.notna().all(), 'A selected feature has no valid training observations.')
    filled = frame.fillna(median)
    scale = filled.std(ddof=0).replace(0, 1)
    return dict(median=median, center=filled.mean(), scale=scale)


def design(frame, prep):
    standardized = (frame.fillna(prep['median']) - prep['center']) / prep['scale']
    # One explicit missingness flag per source feature, including flags all zero in training.
    result = np.column_stack([standardized.to_numpy(float), frame.isna().to_numpy(float)])
    require(np.isfinite(result).all(), 'Nonfinite model inputs after preprocessing.')
    return result


def metrics(truth, prediction):
    require(np.isfinite(prediction).all(), 'Nonfinite predictions.')
    ic = None
    if np.ptp(prediction) > 0 and np.ptp(truth) > 0:
        value = float(spearmanr(truth, prediction).statistic)
        ic = value if np.isfinite(value) else None
    return dict(rows=len(truth), mae_log_bps=float(np.mean(abs(prediction - truth))),
                rmse_log_bps=float(np.sqrt(np.mean((prediction - truth) ** 2))),
                spearman_ic=ic, mean_prediction_log_bps=float(np.mean(prediction)),
                mean_actual_log_bps=float(np.mean(truth)))


def run(args):
    source = WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    returns_path = WEEK1 / 'Dataset/canonical_returns_5m.parquet'
    for path in [source, returns_path]:
        require(path.is_file(), f'Missing input: {path}')
    columns = PAIR if args.features == 'pair' else EIGHT
    print('Loading trusted local prepared artifact...', flush=True)
    data = joblib.load(source)
    require(isinstance(data, (tuple, list)) and len(data) == 4, 'Unexpected prepared artifact format.')
    x, old_y, evaluation_index, _ = data
    require(isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None, 'Expected timezone-aware timestamps.')
    require(x.index.is_unique and x.index.is_monotonic_increasing, 'Feature timestamps must be unique and sorted.')
    require(x.index.equals(old_y.index), 'Prepared feature/target index mismatch.')
    evaluation_index = pd.DatetimeIndex(evaluation_index)
    require(evaluation_index.tz is not None and evaluation_index.is_unique
            and evaluation_index.is_monotonic_increasing, 'Invalid evaluation index.')
    require(evaluation_index.isin(x.index).all() and len(evaluation_index) > 0, 'Evaluation timestamps missing.')
    require((evaluation_index >= CUTOFF).all(), 'Evaluation contains timestamps before 2024.')
    require(set(columns).issubset(x.columns), 'Selected features missing from prepared artifact.')
    x = x[columns].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    print('Checking canonical returns and rebuilding horizon-specific targets...', flush=True)
    r = cr.load_returns(returns_path)
    require(np.isfinite(r.dropna()).all(), 'Nonfinite valid returns.')
    canonical_audit = cr.validate(r, old_y)
    if args.features == 'eight':
        derived = cr.features(r, x.index)
        for name in columns:
            if name.startswith('PRICE__'):
                x[name] = derived[name]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = WORKSPACE / 'Code/experiments/short_horizon_median_v1' / f'{stamp}_{args.features}'
    out.mkdir(parents=True, exist_ok=False)
    framework = WORKSPACE.parents[1] / 'ml_framework'
    instruction_paths = [WORKSPACE / 'AGENTS.md', WORKSPACE / 'PROJECT_CONTEXT.md',
                         WORKSPACE / 'Code/AGENTS.md'] + [framework / name for name in
                         ['AGENTS.md', 'CONTEXT.md', 'docs/INDEX.md', 'docs/methods/WORKFLOW.md',
                          'docs/methods/DATA_AND_METRICS.md', 'docs/findings/RECORDED_RESULTS.md']]
    inputs = [source, returns_path, Path(cr.__file__), Path(__file__)]
    save_json(out / 'MANIFEST.json', dict(input_sha256={str(p): sha(p) for p in inputs},
              instructions_sha256={str(p): sha(p) for p in instruction_paths if p.is_file()},
              python=platform.python_version(), versions={n: importlib.metadata.version(n) for n in
              ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'duckdb']}))
    save_json(out / 'CONTRACT.json', dict(features=columns, horizons_hours=[1, 4],
              target='Absolute endpoint log return, in log basis points; not conditional UP/DOWN.',
              model='QuantileRegressor(quantile=0.5, alpha=0, solver=highs), nonnegative predictions',
              training='Fit once per horizon. Label endpoint strictly before 2024-01-01 UTC.',
              preprocessing='Training-only median imputation, mean/std scaling, missingness indicators.',
              evaluation='Existing prepared development timestamps; require valid new labels for every row.',
              limitation='Previously examined historical data. New experiment, not exact reproduction of old RF or MP0004 scores.',
              canonical_audit=canonical_audit))
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    scores = []
    for hours in [1, 4]:
        y = cr.make_labels(r, x.index, hours)
        require(evaluation_index.isin(y.index).all(), f'{hours}h labels missing on evaluation rows; refusing silent row drops.')
        require(np.isfinite(y['magnitude_log_bps']).all(), 'Nonfinite targets.')
        train = y.index[y.label_end < CUTOFF]
        require(len(train) >= 100, 'Insufficient historical training data.')
        require(train.intersection(evaluation_index).empty, 'Training/evaluation overlap.')
        # Independently check exact forward intervals at sampled train/evaluation origins.
        for keys in [train, evaluation_index]:
            for origin in keys[::max(1, len(keys) // 20)]:
                window = r[(r.index > origin) & (r.index <= origin + pd.Timedelta(hours=hours))]
                require(len(window) == hours * 12 and window.notna().all(), 'Invalid forward target window.')
                np.testing.assert_allclose(window.sum(), y.loc[origin, 'signed_return_log_bps'], rtol=0, atol=1e-8)
        prep = preprocessing(x.loc[train])
        train_y = y.loc[train, 'magnitude_log_bps'].to_numpy(float)
        truth = y.loc[evaluation_index, 'magnitude_log_bps'].to_numpy(float)
        model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')
        print(f'Fitting {hours}h model: {len(train):,} training rows, {len(truth):,} evaluation rows...', flush=True)
        model.fit(design(x.loc[train], prep), train_y)
        pred = np.maximum(0, model.predict(design(x.loc[evaluation_index], prep)))
        predictions = pd.DataFrame(dict(truth_log_bps=truth, median_regression=pred,
                    mean_baseline=np.full(len(truth), train_y.mean()),
                    median_baseline=np.full(len(truth), np.median(train_y))), index=evaluation_index)
        predictions.to_csv(out / f'predictions_{hours}h.csv', index_label='decision_time')
        artifact = dict(model=model, prep=prep, columns=columns, hours=hours, cutoff=CUTOFF,
                        train_index=train, evaluation_index=evaluation_index)
        joblib.dump(artifact, out / f'model_{hours}h.joblib', compress=3)
        replay = joblib.load(out / f'model_{hours}h.joblib')
        np.testing.assert_allclose(np.maximum(0, replay['model'].predict(design(x.loc[evaluation_index], replay['prep']))), pred, rtol=0, atol=1e-10)
        for name in ['median_regression', 'mean_baseline', 'median_baseline']:
            scores.append(dict(horizon_hours=hours, model=name, **metrics(truth, predictions[name].to_numpy())))
        save_json(out / f'audit_{hours}h.json', dict(training_rows=len(train), evaluation_rows=len(truth),
                  last_training_label_end=y.loc[train, 'label_end'].max(),
                  training_missing_fraction=x.loc[train].isna().mean().to_dict(),
                  evaluation_missing_fraction=x.loc[evaluation_index].isna().mean().to_dict(),
                  saved_model_replay=True, sampled_forward_window_checks=True))
    table = pd.DataFrame(scores)
    table.to_csv(out / 'scores.csv', index=False)
    summary = table.to_string(index=False) + '\n\nUnits: log bps. Historical exploratory evaluation.\n'
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    save_json(out / 'COMPLETE.json', dict(status='EXECUTED', scores=scores))
    print(summary, flush=True)
    print(f'Output: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', choices=['pair', 'eight'], default='pair')
    run(parser.parse_args())
