"""Combine saved 48h direction and report-winning magnitude predictions. No training."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE / 'experiments'


def read(path):
    f = pd.read_csv(path)
    f['decision_time'] = pd.to_datetime(f.decision_time, utc=True, errors='raise')
    if f.decision_time.isna().any() or f.decision_time.duplicated().any() or f.empty:
        raise ValueError(f'Empty file or invalid/duplicate timestamps: {path}')
    return f.set_index('decision_time').sort_index()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction-run', type=Path, default=EXP / 'direction_v45_style_v1/20261112T045009318177Z')
    p.add_argument('--up-run', type=Path, default=EXP / 'gru_conditional_mae_v1')
    p.add_argument('--down-run', type=Path, default=EXP / 'original_multiwindow_mae_v1')
    p.add_argument('--output-dir', type=Path, default=EXP / 'signed_return_48h_v1')
    args = p.parse_args()
    for folder in [args.direction_run, args.up_run, args.down_run]:
        if not (folder / 'COMPLETE.json').is_file():
            raise ValueError(f'Missing completion record: {folder}')
    paths = [args.direction_run / 'development_predictions.csv',
             args.up_run / 'predictions.csv', args.down_run / 'predictions.csv']
    direction, up, down = [read(path) for path in paths]
    if not direction.index.equals(up.index) or not direction.index.equals(down.index):
        raise ValueError('All three timestamp sets must match exactly; no silent row dropping')
    values = pd.DataFrame(dict(p_up=direction.p_up, p_down=direction.p_down,
                               magnitude_up_bps=up.UP_gru,
                               magnitude_down_bps=down.DOWN_three_windows))
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError('Nonfinite probabilities or magnitudes')
    if not values.p_up.between(0, 1).all() or not values.p_down.between(0, 1).all():
        raise ValueError('Probabilities outside [0,1]')
    # The direction runner exports float32 probabilities to CSV. Independently
    # rounded complements can differ from one by about 6e-8 after CSV parsing.
    probability_sum_error = float((values.p_up + values.p_down - 1).abs().max())
    probability_tolerance = float(2 * np.finfo(np.float32).eps)
    if probability_sum_error > probability_tolerance:
        raise ValueError(f'UP/DOWN probabilities disagree beyond float32 rounding: {probability_sum_error:.9g}')
    # p_up is the model output; DOWN was defined as its complement, not fitted
    # independently. Recompute in float64 only after checking the saved column.
    values['p_up'] = values.p_up.astype(np.float64)
    values['p_down'] = 1.0 - values.p_up
    if (values[['magnitude_up_bps', 'magnitude_down_bps']] < 0).any().any():
        raise ValueError('Conditional magnitude must be nonnegative')
    # Outcomes below are used only for checking and scoring, never forecasting.
    if not up.actual_direction.equals(down.actual_direction):
        raise ValueError('Magnitude files disagree on actual direction')
    if not np.allclose(up.actual_magnitude_bps, down.actual_magnitude_bps, rtol=0, atol=1e-7):
        raise ValueError('Magnitude files disagree on actual magnitude or units')
    if not up.actual_direction.isin(['UP', 'DOWN', 'FLAT']).all():
        raise ValueError('Unknown actual direction')
    if (up.actual_direction == 'FLAT').any():
        raise ValueError('Zero returns need a separate flat-probability convention; current magnitudes exclude flats')
    if not np.array_equal(direction.y.to_numpy(), up.actual_direction.eq('UP').astype(int).to_numpy()):
        raise ValueError('Direction labels disagree with magnitude labels')
    if not np.isfinite(up.actual_magnitude_bps).all() or (up.actual_magnitude_bps < 0).any():
        raise ValueError('Invalid actual magnitudes')
    issue_up = pd.to_datetime(up.issue, utc=True)
    issue_down = pd.to_datetime(down.issue, utc=True)
    if not issue_up.equals(issue_down) or (issue_up > up.index).any():
        raise ValueError('Invalid or inconsistent magnitude issuance dates')
    values['horizon_hours'] = 48
    values['forecast_end'] = values.index + pd.Timedelta(hours=48)
    values['magnitude_model_issue'] = issue_up
    values['up_contribution_bps'] = values.p_up * values.magnitude_up_bps
    values['down_contribution_bps'] = values.p_down * values.magnitude_down_bps
    values['signed_return_forecast_bps'] = values.up_contribution_bps - values.down_contribution_bps
    values['signed_return_forecast_pct'] = values.signed_return_forecast_bps / 100
    values['forecast_sign'] = np.where(values.signed_return_forecast_bps > 0, 'UP',
                                      np.where(values.signed_return_forecast_bps < 0, 'DOWN', 'FLAT'))
    values['actual_signed_return_bps'] = up.actual_magnitude_bps * np.where(up.actual_direction.eq('UP'), 1, -1)
    values['actual_signed_return_pct'] = values.actual_signed_return_bps / 100
    metrics = []
    groups = [('ALL', values)] + [(str(k), g) for k, g in values.groupby('magnitude_model_issue')]
    for period, g in groups:
        for name, forecast in [('combined', g.signed_return_forecast_bps), ('zero_baseline', g.signed_return_forecast_bps * 0)]:
            err = forecast - g.actual_signed_return_bps
            ic = forecast.corr(g.actual_signed_return_bps, method='spearman') if forecast.nunique() > 1 else None
            metrics.append(dict(period=period, model=name, rows=len(g), mae_bps=float(err.abs().mean()),
                                rmse_bps=float(np.sqrt((err**2).mean())), bias_bps=float(err.mean()),
                                rank_ic=float(ic) if ic is not None and np.isfinite(ic) else None))
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    values.to_csv(out / 'signed_return_predictions.csv', index_label='decision_time')
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    manifest = dict(status='COMPLETE', training_performed=False, horizon_hours=48,
                    formula='p_up * UP_gru - p_down * DOWN_three_windows',
                    units='ordinary basis points; percent = bps/100',
                    interpretation='Probability-weighted conditional MAE-model forecasts; not a calibrated conditional mean or mixture median',
                    evaluation='Historical exposed data; magnitude models chosen by reported test MAE; no fresh confirmation',
                    schedule='Frozen direction model plus quarterly expanding magnitude models',
                    probability_sum_error_before_correction=probability_sum_error,
                    probability_sum_tolerance=probability_tolerance,
                    probability_correction='p_down recomputed as 1-p_up in float64 after validation',
                    rows=len(values), inputs={str(path): digest(path) for path in paths},
                    source_sha256=digest(Path(__file__)))
    (out / 'COMPLETE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f'48-hour signed forecasts saved to: {out / "signed_return_predictions.csv"}')


if __name__ == '__main__':
    main()
