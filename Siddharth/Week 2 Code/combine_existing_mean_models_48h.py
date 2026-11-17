"""Reuse saved mean-trained UP RF and DOWN hybrid predictions. No training."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent / 'experiments'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    done = json.loads((folder / 'COMPLETE.json').read_text())
    path = folder / 'signed_return_predictions.csv'
    check(done.get('status') == 'COMPLETE', f'Incomplete run: {folder}')
    check(done['prediction_sha256'] == sha(path), f'Prediction hash mismatch: {path}')
    check(done['contract_sha256'] == sha(folder / 'CONTRACT.json'), f'Contract hash mismatch: {folder}')
    f = pd.read_csv(path)
    f['decision_time'] = pd.to_datetime(f.decision_time, utc=True, errors='raise')
    check(len(f) > 0 and not f.decision_time.isna().any() and not f.decision_time.duplicated().any(),
          'Invalid timestamps')
    f['issue'] = pd.to_datetime(f.issue, utc=True, errors='raise')
    return f.set_index('decision_time').sort_index(), path


def metrics(actual, prediction):
    error = prediction - actual
    ic = prediction.corr(actual, method='spearman') if prediction.nunique() > 1 and actual.nunique() > 1 else None
    return dict(rows=len(actual), mae_bps=float(error.abs().mean()),
        rmse_bps=float(np.sqrt((error**2).mean())), bias_bps=float(error.mean()),
        rank_ic=float(ic) if ic is not None and np.isfinite(ic) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rf-run', type=Path, default=EXP / 'mlp_rf_conditional_means_48h_v1/20261112T064753697944Z')
    p.add_argument('--hybrid-run', type=Path, default=EXP / 'conditional_means_signed_48h_v1/20261112T063535834150Z')
    p.add_argument('--output-dir', type=Path, default=EXP / 'existing_mean_models_signed_48h_v1')
    args = p.parse_args()
    rf, rf_path = load(args.rf_run)
    hybrid, hybrid_path = load(args.hybrid_run)
    check(rf.index.equals(hybrid.index), 'Timestamp sets differ; no silent row dropping')
    check(rf.issue.equals(hybrid.issue) and rf.issue.notna().all()
          and (rf.issue <= rf.index).all(), 'Missing, inconsistent or future model issuance')
    for frame in [rf, hybrid]:
        check(frame.symbol.eq('BTCUSDT').all() and frame.horizon_hours.eq(48).all(), 'Wrong asset/horizon')
        check((pd.to_datetime(frame.forecast_end, utc=True) == frame.index + pd.Timedelta(hours=48)).all(),
              'Forecast endpoints do not match 48-hour horizon')
    np.testing.assert_allclose(rf[['p_up', 'p_down']], hybrid[['p_up', 'p_down']], rtol=0, atol=1e-7)
    np.testing.assert_allclose(rf.actual_signed_return_bps, hybrid.actual_signed_return_bps, rtol=0, atol=1e-7)
    np.testing.assert_allclose(rf.DOWN_reference_mean_bps, hybrid.down_mean_magnitude_bps, rtol=0, atol=1e-7)
    f = rf[['symbol', 'horizon_hours', 'forecast_end', 'issue', 'p_up', 'p_down']].copy()
    f['up_mean_bps'] = rf.UP_rf_mean_bps
    f['down_mean_magnitude_bps'] = hybrid.down_mean_magnitude_bps
    f['actual_signed_return_bps'] = rf.actual_signed_return_bps
    numeric = ['p_up', 'p_down', 'up_mean_bps', 'down_mean_magnitude_bps', 'actual_signed_return_bps']
    check(np.isfinite(f[numeric].to_numpy(float)).all(), 'Nonfinite inputs')
    check(f.p_up.between(0, 1).all() and f.p_down.between(0, 1).all(), 'Invalid probabilities')
    check((f[['up_mean_bps', 'down_mean_magnitude_bps']] >= 0).all().all(), 'Negative magnitude estimate')
    check((f.actual_signed_return_bps != 0).all(), 'Flat outcomes need an explicit flat policy')
    complement_error = float((f.p_up + f.p_down - 1.).abs().max())
    check(complement_error <= 2 * np.finfo(np.float32).eps, 'Probability complement mismatch')
    f['p_down'] = 1. - f.p_up.astype(float)
    f['up_mean_pct'] = f.up_mean_bps / 100.
    f['down_mean_magnitude_pct'] = f.down_mean_magnitude_bps / 100.
    f['up_contribution_bps'] = f.p_up * f.up_mean_bps
    f['down_contribution_bps'] = -f.p_down * f.down_mean_magnitude_bps
    f['signed_return_forecast_bps'] = f.up_contribution_bps + f.down_contribution_bps
    f['signed_return_forecast_pct'] = f.signed_return_forecast_bps / 100.
    f['actual_signed_return_pct'] = f.actual_signed_return_bps / 100.
    f['side_mean_baseline_bps'] = rf.side_mean_baseline_bps
    f['previous_mlp_signed_return_bps'] = rf.mlp_signed_return_bps
    f['previous_gru_hybrid_signed_return_bps'] = hybrid.signed_return_forecast_bps
    f['training_signed_mean_bps'] = rf.training_signed_mean_bps
    f['zero_baseline_bps'] = 0.
    checks = ['side_mean_baseline_bps', 'previous_mlp_signed_return_bps',
              'previous_gru_hybrid_signed_return_bps', 'training_signed_mean_bps']
    check(np.isfinite(f[checks].to_numpy(float)).all(), 'Invalid comparison forecasts')
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    f.to_csv(out / 'signed_return_predictions.csv', index_label='decision_time')
    rows = []
    models = dict(up_rf_down_hybrid='signed_return_forecast_bps',
        previous_mlp='previous_mlp_signed_return_bps', previous_gru_hybrid='previous_gru_hybrid_signed_return_bps',
        side_means='side_mean_baseline_bps', training_signed_mean='training_signed_mean_bps', zero='zero_baseline_bps')
    for period, g in [('ALL', f)] + [(str(k), v) for k, v in f.groupby('issue')]:
        for name, column in models.items():
            rows.append(dict(period=period, model=name, **metrics(g.actual_signed_return_bps, g[column])))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'signed_metrics.csv', index=False)
    manifest = dict(status='COMPLETE', training_performed=False, inference_reexecuted=False,
        method='Reuses saved mean-trained model predictions; no conversion of MAE predictions',
        up='UP_rf_mean_bps: earlier-validation-selected 4/5/8 feature RF, squared-error training',
        down='down_mean_magnitude_bps: Ridge + squared-error RF, three-window MSE weight selection',
        units='ordinary bps; pct=bps/100; down magnitude positive, contribution negative',
        formula='p_up*up_mean_bps - p_down*down_mean_magnitude_bps',
        complement_error_before_correction=complement_error, rows=len(f),
        limitation='Family combination proposed after inspecting historical scores. Exploratory, not independent confirmation. Mean-targeting models do not guarantee calibrated means.',
        inputs={str(rf_path): sha(rf_path), str(hybrid_path): sha(hybrid_path)},
        model_lineage_files={str(args.rf_run / 'SELECTIONS.json'): sha(args.rf_run / 'SELECTIONS.json'),
                             str(args.hybrid_run / 'FIT_AUDIT.json'): sha(args.hybrid_run / 'FIT_AUDIT.json')},
        source_sha256=sha(Path(__file__)), prediction_sha256=sha(out / 'signed_return_predictions.csv'))
    (out / 'COMPLETE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(table[table.period.eq('ALL')].to_string(index=False))
    print('\nMean magnitudes and signed returns (ordinary percent):')
    print(f[['up_mean_pct', 'down_mean_magnitude_pct', 'signed_return_forecast_pct']].head(10).to_string())
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    main()
