r"""Compare saved forecasts against the user-specified canonical Y directory.
python .\Code\compare_signed_returns_canonical_y_48h.py
Read-only sources; no training or prediction changes.
"""
from pathlib import Path
import argparse
import json
import duckdb
import numpy as np
import pandas as pd
import tune_conditional_mean_magnitudes_48h as base

EXP = Path(__file__).resolve().parent / 'experiments'
YDIR = Path(r'C:\Users\Siddh\Desktop\ML\October\Week 3\Dataset\HANDOFF_2_CANONICAL_XY_V1_1\HANDOFF_2_CANONICAL_XY_V1_1\y\horizon_minutes=2880\symbol=BTCUSDT')


def convert_actual(direction, magnitude):
    return 10000. * np.expm1(direction * magnitude / 10000.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--y-dir', type=Path, default=YDIR)
    p.add_argument('--prediction-run', type=Path, default=EXP / 'best_ridge_signed_48h_v1/20261112T090717301790Z')
    p.add_argument('--output-dir', type=Path, default=EXP / 'canonical_y_signed_comparison_48h_v1')
    args = p.parse_args()
    predpath = args.prediction_run / 'signed_return_predictions.csv'
    done = json.loads((args.prediction_run / 'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and done['prediction_sha256'] == base.sha(predpath), 'Prediction integrity failed')
    f = pd.read_csv(predpath)
    f['decision_time'] = pd.to_datetime(f.decision_time, utc=True, errors='raise')
    base.check(len(f) > 0 and f.decision_time.notna().all() and f.decision_time.is_unique, 'Invalid forecast timestamps')
    base.check(f.symbol.eq('BTCUSDT').all() and f.horizon_hours.eq(48).all(), 'Wrong forecast asset/horizon')
    base.check(np.isfinite(f.signed_return_forecast_bps).all(), 'Nonfinite forecast')
    files = sorted(args.y_dir.rglob('*.parquet'))
    base.check(len(files) > 0, 'No Parquet files in Y directory')
    print(f'Reading {len(files)} Y partitions; matching exact UTC forecast timestamps...', flush=True)
    with duckdb.connect() as con:
        con.execute("SET TimeZone='UTC'")
        con.register('forecast_keys', f[['decision_time']])
        raw = con.execute('''SELECT y.decision_time, y.canonical_symbol, y.horizon_minutes,
                y.direction, y.net_magnitude_log_bps, y.label_status
            FROM read_parquet(?, hive_partitioning=false) y
            INNER JOIN forecast_keys k ON y.decision_time=k.decision_time''',
            [[str(v) for v in files]]).fetchdf()
    raw['decision_time'] = pd.to_datetime(raw.decision_time, utc=True)
    base.check(raw.decision_time.is_unique, 'Duplicate Y origins: refusing ambiguous comparison')
    base.check(raw.canonical_symbol.eq('BTCUSDT').all()
               and pd.to_numeric(raw.horizon_minutes).eq(2880).all(), 'Y asset/horizon mismatch')
    raw = raw.rename(columns={'direction': 'source_direction', 'net_magnitude_log_bps': 'source_net_magnitude_log_bps',
                              'label_status': 'source_label_status'}).drop(columns=['canonical_symbol', 'horizon_minutes'])
    f = f.rename(columns={'actual_signed_return_bps': 'previous_actual_signed_return_bps',
                          'actual_signed_return_pct': 'previous_actual_signed_return_pct'})
    f = f.drop(columns=['error_bps', 'absolute_error_bps'], errors='ignore')
    f = f.merge(raw, on='decision_time', how='left', validate='one_to_one', indicator=True).sort_values('decision_time')
    finite = np.isfinite(f.source_direction.astype(float)) & np.isfinite(f.source_net_magnitude_log_bps.astype(float))
    valid = f.source_label_status.eq('VALID') & finite & f.source_direction.isin([-1, 0, 1]) & f.source_net_magnitude_log_bps.ge(0)
    valid &= f.source_direction.eq(0).eq(f.source_net_magnitude_log_bps.eq(0))
    f['comparison_status'] = 'invalid_source_label_or_value'
    f.loc[f['_merge'].eq('left_only'), 'comparison_status'] = 'missing_source_timestamp'
    f.loc[valid, 'comparison_status'] = 'compared'
    f['actual_signed_return_log_bps'] = (f.source_direction * f.source_net_magnitude_log_bps).where(valid)
    f['actual_signed_return_bps'] = convert_actual(f.source_direction, f.source_net_magnitude_log_bps).where(valid)
    base.check(np.isfinite(f.loc[valid, 'actual_signed_return_bps']).all(), 'Conversion overflow')
    f['actual_signed_return_pct'] = f.actual_signed_return_bps / 100.
    f['signed_return_forecast_pct'] = f.signed_return_forecast_bps / 100.
    f['error_bps'] = f.signed_return_forecast_bps-f.actual_signed_return_bps
    f['absolute_error_bps'] = f.error_bps.abs()
    f['actual_difference_vs_previous_bps'] = f.actual_signed_return_bps-f.previous_actual_signed_return_bps
    f = f.drop(columns='_merge')
    g = f.loc[valid].copy()
    names = {'ridge_signed_return': 'signed_return_forecast_bps', 'side_mean_baseline': 'side_mean_baseline_bps',
             'training_signed_mean': 'training_signed_mean_bps', 'previous_gru_hybrid': 'previous_gru_hybrid_bps', 'zero': 'zero_bps'}
    rows = []
    if len(g):
        for period, h in [('ALL', g)] + [(str(k), v) for k, v in g.groupby('issue')]:
            for name, col in names.items():
                rows.append(dict(period=period, model=name, **base.metric(h.actual_signed_return_bps, h[col])))
    metrics = pd.DataFrame(rows)
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    f.to_csv(out / 'prediction_vs_canonical_actual.csv', index=False)
    f.loc[~valid].to_csv(out / 'missing_or_invalid_actuals.csv', index=False)
    differences = f.loc[valid & f.actual_difference_vs_previous_bps.abs().gt(1e-7)]
    differences.to_csv(out / 'actual_mismatches_vs_previous.csv', index=False)
    metrics.to_csv(out / 'signed_metrics.csv', index=False)
    summary = dict(forecast_rows=len(f), matched_source_rows=len(raw), compared_rows=len(g),
        status_counts=f.comparison_status.value_counts().to_dict(),
        actual_mismatch_rows_over_1e_7_bps=len(differences),
        max_actual_difference_vs_previous_bps=float(g.actual_difference_vs_previous_bps.abs().max()) if len(g) else None)
    base.dump(out / 'COMPARISON_AUDIT.json', summary)
    base.dump(out / 'COMPLETE.json', dict(status='COMPLETE', comparison_available=bool(len(g)),
        training_performed=False, forecast_changes=False, **summary,
        formula='signed_log_bps=direction*net_magnitude_log_bps; ordinary_bps=10000*expm1(signed_log_bps/10000)',
        source_files={str(v): base.sha(v) for v in files}, prediction_source_sha256=base.sha(predpath),
        source_sha256=base.sha(Path(__file__)), comparison_sha256=base.sha(out/'prediction_vs_canonical_actual.csv')))
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    summary_text = ('SIGNED FORECAST VS SPECIFIED CANONICAL Y - ordinary bps\n'
        + json.dumps(summary, indent=2) + '\n\n'
        + (metrics[metrics.period.eq('ALL')].to_string(index=False) if len(metrics) else 'No valid matched actuals; no metrics computed.')
        + '\nMissing/invalid actuals remain blank. Exact UTC matching, no nearest-time joins.\n'
        'Y up/down_magnitude_log_bps excursions are not used. This is historical comparison, not fresh testing.\n')
    (out / 'SUMMARY.txt').write_text(summary_text, encoding='utf-8')
    print(summary_text)
    print(f[['decision_time', 'signed_return_forecast_pct', 'actual_signed_return_pct', 'error_bps']].head(10).to_string(index=False))
    print('Outputs saved:', out)


if __name__ == '__main__':
    main()
