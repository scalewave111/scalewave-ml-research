"""Combine existing Ridge means and frozen direction probabilities; no training."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tune_conditional_mean_magnitudes_48h as b

EXP = Path(__file__).resolve().parent / 'experiments'
MAG = EXP / 'controlled_mean_ablation_48h_v1/20261112T082233470776Z'
DIR = EXP / 'direction_v45_style_v1/20261112T045009318177Z'
OLD = EXP / 'conditional_means_signed_48h_v1/20261112T063535834150Z'
MODELS = {'UP': 'core4_drop_liq_breadth_1h', 'DOWN': 'full8_drop_liq_1h'}


def load(folder, filename, hash_key):
    done = json.loads((folder / 'COMPLETE.json').read_text())
    b.check(done['status'] == 'COMPLETE', 'Incomplete source')
    expected = done['artifacts'][filename] if hash_key == 'artifacts' else done[hash_key]
    b.check(b.sha(folder / filename) == expected, 'Source prediction hash mismatch')
    contract_hash = done['artifacts']['CONTRACT.json'] if hash_key == 'artifacts' else done['contract_sha256']
    b.check(b.sha(folder / 'CONTRACT.json') == contract_hash, 'Contract hash mismatch')
    f = pd.read_csv(folder / filename, index_col='decision_time')
    f.index = pd.to_datetime(f.index, utc=True)
    b.check(f.index.is_unique and not f.index.hasnans, 'Invalid origins')
    return f.sort_index(), done


def main():
    m, _ = load(MAG, 'magnitude_predictions.csv', 'predictions_sha256')
    d, ddone = load(DIR, 'development_predictions.csv', 'artifacts')
    old, _ = load(OLD, 'signed_return_predictions.csv', 'prediction_sha256')
    b.check(len(m) == 18481 and m.index.equals(d.index) and m.index.equals(old.index), 'Exact alignment failed')
    m['issue'] = pd.to_datetime(m.issue, utc=True)
    b.check(m.issue.equals(pd.to_datetime(old.issue, utc=True)) and (m.issue <= m.index).all(), 'Issue mismatch')
    b.check(old.symbol.eq('BTCUSDT').all() and old.horizon_hours.eq(48).all(), 'Wrong symbol/horizon')
    b.check((pd.to_datetime(old.forecast_end, utc=True) == m.index + pd.Timedelta(hours=48)).all(), 'Endpoint mismatch')
    pipe = DIR / 'final/pipeline.joblib'
    b.check(b.sha(pipe) == ddone['artifacts']['final\\pipeline.joblib'], 'Direction pipeline hash mismatch')
    cutoff = pd.Timestamp(joblib.load(pipe)['max_label_end'])
    b.check(cutoff < m.index.min(), 'Direction labels overlap evaluation')
    np.testing.assert_allclose(d.p_up, old.p_up, rtol=0, atol=1e-7)
    b.check(np.isfinite(d[['p_up', 'p_down']]).all().all() and d.p_up.between(0, 1).all()
            and d.p_down.between(0, 1).all(), 'Invalid probabilities')
    b.check(float((d.p_up+d.p_down-1).abs().max()) <= 2*np.finfo(np.float32).eps, 'Complement mismatch')
    prepared = Path(__file__).resolve().parents[2] / 'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    contract = json.loads((MAG / 'CONTRACT.json').read_text())
    b.check(b.sha(prepared) == contract['prepared_sha256'], 'Prepared input changed')
    x, y, dev, _ = joblib.load(prepared)
    actual = pd.Series(10000*np.expm1(y.loc[m.index, 'signed_return_log_bps']/10000), index=m.index)
    np.testing.assert_allclose(actual, old.actual_signed_return_bps, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual.abs(), m.actual_magnitude_bps, rtol=0, atol=1e-7)
    b.check((actual != 0).all() and np.array_equal(d.y, (actual > 0).astype(int)), 'Outcome definition mismatch')
    ledger = json.loads((MAG / 'FIT_AUDIT.json').read_text())
    f = pd.DataFrame({'symbol': 'BTCUSDT', 'horizon_hours': 48, 'issue': m.issue,
                      'forecast_end': m.index+pd.Timedelta(hours=48), 'p_up': d.p_up,
                      'p_down': 1.-d.p_up.astype(float)}, index=m.index)
    verified = []
    for side, feature_set in MODELS.items():
        col = f'{side}_ridge_{feature_set}'
        f[f'{side.lower()}_mean_magnitude_bps'] = m[col]
        for issue, g in m.groupby('issue'):
            entries = [a for a in ledger if a['side'] == side and a['feature_set'] == feature_set
                       and a['family'] == 'ridge' and pd.Timestamp(a['issue']) == issue]
            b.check(len(entries) == 1, 'Missing unique model audit')
            a = entries[0]
            b.check(pd.Timestamp(a['max_training_label_end']) < issue, 'Magnitude label maturity failed')
            path = MAG / a['artifact']
            b.check(b.sha(path) == a['sha256'], 'Magnitude model hash mismatch')
            replay = b.predict(joblib.load(path), x.loc[g.index].replace([np.inf, -np.inf], np.nan))
            np.testing.assert_allclose(replay, g[col], rtol=0, atol=1e-6)
            verified.append(a)
    b.check(np.isfinite(f[['up_mean_magnitude_bps', 'down_mean_magnitude_bps']]).all().all()
            and (f[['up_mean_magnitude_bps', 'down_mean_magnitude_bps']] >= 0).all().all(), 'Invalid means')
    f['signed_return_forecast_bps'] = f.p_up*f.up_mean_magnitude_bps-f.p_down*f.down_mean_magnitude_bps
    f['actual_signed_return_bps'] = actual
    f['signed_return_forecast_pct'] = f.signed_return_forecast_bps/100
    f['actual_signed_return_pct'] = actual/100
    f['error_bps'] = f.signed_return_forecast_bps-actual
    f['absolute_error_bps'] = f.error_bps.abs()
    f['side_mean_baseline_bps'] = f.p_up*m.UP_training_mean_bps-f.p_down*m.DOWN_training_mean_bps
    np.testing.assert_allclose(f.side_mean_baseline_bps, old.side_mean_baseline_bps, rtol=0, atol=1e-7)
    f['training_signed_mean_bps'] = old.signed_mean_baseline_bps
    f['previous_gru_hybrid_bps'] = old.signed_return_forecast_bps
    f['zero_bps'] = 0.
    names = dict(best_ridge='signed_return_forecast_bps', side_means='side_mean_baseline_bps',
                 training_signed_mean='training_signed_mean_bps', previous_gru_hybrid='previous_gru_hybrid_bps', zero='zero_bps')
    rows = []
    for period, g in [('ALL', f)] + [(str(k), v) for k, v in f.groupby('issue')]:
        for name, col in names.items():
            rows.append(dict(period=period, model=name, **b.metric(g.actual_signed_return_bps, g[col])))
    metrics = pd.DataFrame(rows)
    out = EXP / 'best_ridge_signed_48h_v1' / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    f.to_csv(out / 'signed_return_predictions.csv', index_label='decision_time')
    metrics.to_csv(out / 'signed_metrics.csv', index=False)
    f[['signed_return_forecast_pct', 'actual_signed_return_pct', 'error_bps']].head(10).to_csv(out / 'first_10_comparison.csv', index_label='decision_time')
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    b.dump(out / 'MODEL_AUDIT.json', verified)
    (out / 'README.txt').write_text('BTCUSDT 48-hour expected signed endpoint returns. Ordinary bps; percent=bps/100.\n'
        'Forecast=p_up*UP_mean_magnitude-(1-p_up)*DOWN_mean_magnitude. Both means used at every origin.\n'
        'UP=three-feature Ridge; DOWN=seven-feature Ridge from controlled ablation. Eight model artifacts replayed.\n'
        'Direction=reused v45-style probabilities, NOT exact reported v45. Direction training labels end '+str(cutoff)+'\n'
        'Actuals converted individually from canonical log-bps endpoint labels using expm1 and matched to previous actuals.\n'
        'Reference price inherited from canonical label construction; exchange/price type not re-established.\n'
        'All 18481 available origins retained; no missing forecast fills. Direction and magnitude timestamp/label checks passed.\n'
        'These configurations were chosen after historical evaluation. Results are exploratory, not untouched-test evidence.\n'
        'No training, calibration or model selection performed in this evaluation. Overlapping 48h targets are not independent.\n', encoding='utf-8')
    b.dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(f), training_performed=False,
        magnitude_model_replay_passed=True, direction_retrained=False, exact_v45_reproduction=False,
        direction_max_training_label_end=cutoff, models=MODELS,
        inputs={str(p): b.sha(p) for p in [MAG/'magnitude_predictions.csv', DIR/'development_predictions.csv',
                                         OLD/'signed_return_predictions.csv', prepared, pipe]},
        source_sha256=b.sha(Path(__file__)), prediction_sha256=b.sha(out/'signed_return_predictions.csv')))
    print(metrics[metrics.period.eq('ALL')].to_string(index=False))
    print(f[['signed_return_forecast_pct', 'actual_signed_return_pct']].head(5).to_string())
    print('Outputs saved:', out)


if __name__ == '__main__':
    main()
