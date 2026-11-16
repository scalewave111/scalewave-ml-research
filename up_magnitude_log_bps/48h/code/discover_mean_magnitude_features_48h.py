r"""User-run causal price-feature investigation for conditional mean magnitudes.
python .\Code\discover_mean_magnitude_features_48h.py
Fixed Ridge; earlier-only feature-block selection; no signed forecasts.
"""
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as base

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent.parent / 'Week 1'
sys.path.insert(0, str(WEEK1 / 'Code'))
import canonical_return_features as canonical

ANCHORS = {'UP': [base.COLS[i] for i in [0, 1, 3, 5]],
           'DOWN': [base.COLS[i] for i in [0, 1, 3, 5, 7]]}
GROUPS = {
    'volatility_levels': [f'NEW__rv_{h}h' for h in [4, 24, 48, 168]],
    'side_volatility': [f'NEW__{s}_rv_{h}h' for h in [24, 48] for s in ['up', 'down']],
    'recent_moves': [f'NEW__{s}_{h}h' for h in [4, 24, 48] for s in ['net', 'abs_net']],
    'jumps': [f'NEW__{s}_{h}h' for h in [24, 48] for s in ['max_abs', 'concentration']],
    'volatility_change': ['NEW__rv_ratio_24_168', 'NEW__rv_ratio_48_168'],
}


def price_features(r, index):
    """r is indexed at five-minute return END time; excludes the current endpoint."""
    values = {}
    for h in [4, 24, 48, 168]:
        opts = dict(window=f'{h}h', closed='left', min_periods=h*12)
        square = r.pow(2).rolling(**opts).sum()
        values[f'NEW__rv_{h}h'] = np.sqrt(square)
        if h in [4, 24, 48]:
            net = r.rolling(**opts).sum()
            values[f'NEW__net_{h}h'], values[f'NEW__abs_net_{h}h'] = net, net.abs()
        if h in [24, 48]:
            for side, rr in [('up', r.clip(lower=0)), ('down', r.clip(upper=0))]:
                values[f'NEW__{side}_rv_{h}h'] = np.sqrt(rr.pow(2).rolling(**opts).sum())
            peak = r.abs().rolling(**opts).max()
            values[f'NEW__max_abs_{h}h'] = peak
            values[f'NEW__concentration_{h}h'] = peak.pow(2) / square.clip(lower=1e-12)
    for h in [24, 48]:
        values[f'NEW__rv_ratio_{h}_168'] = (values[f'NEW__rv_{h}h']/np.sqrt(h)
            / (values['NEW__rv_168h']/np.sqrt(168)).clip(lower=1e-6))
    return pd.DataFrame(values).reindex(index)


def causal_check(r, index):
    cut = index[len(index)//2]
    origins = index[index <= cut][-20:]
    original = price_features(r, origins)
    pd.testing.assert_frame_equal(original, price_features(r.loc[r.index <= cut], origins))
    modified = r.copy()
    modified.loc[modified.index >= cut] = 100000.
    pd.testing.assert_frame_equal(price_features(r, pd.DatetimeIndex([cut])),
                                  price_features(modified, pd.DatetimeIndex([cut])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepared', type=Path, default=WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    parser.add_argument('--returns', type=Path, default=WEEK1 / 'Dataset/canonical_returns_5m.parquet')
    parser.add_argument('--reference-run', type=Path, default=base.EXP / 'controlled_mean_ablation_48h_v1/20261112T082233470776Z')
    parser.add_argument('--output-dir', type=Path, default=base.EXP / 'mean_magnitude_price_features_48h_v1')
    args = parser.parse_args()
    base.check(args.threads > 0, 'Positive thread count required')
    ref_path = args.reference_run / 'magnitude_predictions.csv'
    done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and done['predictions_sha256'] == base.sha(ref_path), 'Reference integrity failed')
    ref = pd.read_csv(ref_path, index_col='decision_time')
    ref.index = pd.to_datetime(ref.index, utc=True)
    ref['issue'] = pd.to_datetime(ref.issue, utc=True)
    ref = ref.sort_index()
    x, y, dev, _ = joblib.load(args.prepared)
    dev = pd.DatetimeIndex(dev).sort_values()
    base.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing
               and x.index.tz is not None and x.columns.is_unique, 'Invalid prepared index')
    base.check(len(dev) == 18481 and ref.index.is_unique and dev.equals(ref.index), 'Evaluation population differs')
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    base.check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Invalid horizon')
    print('Checking canonical source, targets and causal feature construction...', flush=True)
    r = canonical.load_returns(args.returns)
    audit = canonical.validate(r, y)
    causal_check(r, x.index)
    derived = price_features(r, x.index)
    x = x.join(derived).replace([np.inf, -np.inf], np.nan)
    log_target = y.signed_return_log_bps.astype(float)
    target = pd.Series(np.abs(10000*np.expm1(log_target.to_numpy()/10000)), index=y.index)
    base.check(np.isfinite(target).all() and (log_target.loc[dev] != 0).all(), 'Invalid outcomes')
    np.testing.assert_allclose(target.loc[dev], ref.actual_magnitude_bps, rtol=0, atol=1e-7)
    base.check(np.array_equal(np.where(log_target.loc[dev] > 0, 'UP', 'DOWN'), ref.actual_direction), 'Side mismatch')
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    groups = {'anchor': [], **GROUPS, 'all_new': sum(GROUPS.values(), [])}
    for side in ANCHORS:
        for name, columns in groups.items():
            base.PACKS[f'{side}_{name}'] = ANCHORS[side] + columns
    base.dump(out / 'CONTRACT.json', dict(symbol='BTCUSDT', horizon_hours=48, anchors=ANCHORS, groups=groups,
        target='Positive magnitude of ordinary endpoint return, conditional on UP or DOWN; no target clipping',
        model='Ridge alpha=0.01*training rows, fixed expanding history; original-scale squared error',
        feature_units='Derived price inputs use past log-bps returns; prediction target uses ordinary bps',
        selection='Among blocks improving pooled earlier validation MSE versus anchor AND training mean and beating anchor in >=2/3 windows, choose minimum pooled MSE; otherwise keep anchor',
        baseline='Benchmark only; fallback is fixed anchor, never silently replaced with baseline',
        validation='Offsets 9,6,3 months, two-month origin windows; strict label maturity purge',
        availability='Price return endpoints strictly before origin. Existing F4Z availability inherited, not newly certified.',
        evaluation='Previously exposed history; no independent confirmation or significance claim',
        tail='Above eligible training-side P90; diagnostic only, never reweights selection',
        source_sha256=base.sha(Path(__file__)), helper_sha256=base.sha(Path(base.__file__)),
        canonical_helper_sha256=base.sha(Path(canonical.__file__)), prepared_sha256=base.sha(args.prepared),
        reference_sha256=base.sha(ref_path), canonical_audit=audit, new_features_causal_check=True,
        signed_return_forecasts=False))
    for path, name in [(Path(__file__), 'source_snapshot.py'), (Path(base.__file__), 'tuning_helper_snapshot.py'),
                        (Path(canonical.__file__), 'canonical_helper_snapshot.py')]:
        (out / name).write_bytes(path.read_bytes())
    frames, validation, selections, audits = [], [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            base.check(len(te) > 0 and (te >= issue).all(), 'Invalid outer fold')
            f = ref.loc[te, ['issue', 'actual_direction', 'actual_magnitude_bps']].copy()
            f['symbol'], f['horizon_hours'] = 'BTCUSDT', 48
            for side in ['UP', 'DOWN']:
                mask = log_target > 0 if side == 'UP' else log_target < 0
                local = {}
                for name in groups:
                    print(f'{issue.date()} {side}: {name}', flush=True)
                    local[name] = []
                    for offset in [9, 6, 3]:
                        cut = issue-pd.DateOffset(months=offset)
                        tr = base.train_keys(y, mask, cut, None)
                        va = y.index[(y.index >= cut) & (y.index < cut+pd.DateOffset(months=2))
                                     & (y.label_end < cut+pd.DateOffset(months=3)) & mask]
                        base.check(len(va) >= 100 and y.loc[va, 'label_end'].max() < issue, 'Invalid validation')
                        config = dict(family='ridge', pack=f'{side}_{name}', alpha_per_row=.01, history_days=None)
                        artifact = base.fit(x, target, tr, config, 48)
                        pred = base.predict(artifact, x.loc[va])
                        m = base.metric(target.loc[va], pred)
                        baseline_mse = float(np.mean((target.loc[va]-target.loc[tr].mean())**2))
                        row = dict(issue=issue, side=side, block=name, offset=offset,
                            training_rows=len(tr), max_training_label_end=y.loc[tr, 'label_end'].max(),
                            max_validation_label_end=y.loc[va, 'label_end'].max(), baseline_mse=baseline_mse, **m)
                        local[name].append(row)
                        validation.append(row)
                pooled = {n: sum(z['mse_bps2']*z['rows'] for z in rows)/sum(z['rows'] for z in rows)
                          for n, rows in local.items()}
                baseline = sum(z['baseline_mse']*z['rows'] for z in local['anchor'])/sum(z['rows'] for z in local['anchor'])
                eligible = [n for n in groups if n != 'anchor' and pooled[n] < min(pooled['anchor'], baseline)
                    and sum(a['mse_bps2'] < b['mse_bps2'] for a, b in zip(local[n], local['anchor'])) >= 2]
                selected = min(eligible, key=lambda n: (pooled[n], len(groups[n]), n)) if eligible else 'anchor'
                selections.append(dict(issue=issue, side=side, selected=selected, eligible=eligible,
                                       validation_mse=pooled, baseline_mse=baseline))
                base.dump(out / 'SELECTIONS.json', selections)
                tr = base.train_keys(y, mask, issue, None)
                f[f'{side}_training_mean'] = float(target.loc[tr].mean())
                f[f'{side}_tail_threshold'] = float(target.loc[tr].quantile(.9))
                for name in groups:
                    config = dict(family='ridge', pack=f'{side}_{name}', alpha_per_row=.01, history_days=None)
                    artifact = base.fit(x, target, tr, config, 48)
                    artifact.update(issue=issue, side=side, max_training_label_end=y.loc[tr, 'label_end'].max())
                    f[f'{side}_{name}'] = base.predict(artifact, x.loc[te])
                    path = out / f'{issue:%Y-%m}_{side}_{name}.joblib'
                    joblib.dump(artifact, path, compress=3)
                    np.testing.assert_allclose(f[f'{side}_{name}'], base.predict(joblib.load(path), x.loc[te]), rtol=0, atol=1e-6)
                    audits.append(dict(issue=issue, side=side, block=name, artifact=path.name,
                        sha256=base.sha(path), training_rows=len(tr), max_training_label_end=artifact['max_training_label_end']))
                f[f'{side}_selected'] = f[f'{side}_{selected}']
                previous = 'ridge_core4_drop_liq_breadth_1h' if side == 'UP' else 'ridge_full8_drop_liq_1h'
                f[f'{side}_previous_fixed'] = ref.loc[te, f'{side}_{previous}']
                print(f'DONE {issue.date()} {side}: {selected}', flush=True)
            frames.append(f)
            pd.DataFrame(validation).to_csv(out / 'validation_metrics.csv', index=False)
            base.dump(out / 'FIT_AUDIT.json', audits)
    result = pd.concat(frames).sort_index()
    base.check(result.index.equals(dev), 'Final population differs')
    result.to_csv(out / 'magnitude_predictions.csv', index_label='decision_time')
    metrics, effects, tails = [], [], []
    for period, f in [('ALL', result)] + [(str(k), g) for k, g in result.groupby('issue')]:
        for side in ['UP', 'DOWN']:
            g = f.loc[f.actual_direction.eq(side)]
            anchor_mse = base.metric(g.actual_magnitude_bps, g[f'{side}_anchor'])['mse_bps2']
            for name in list(groups) + ['selected', 'training_mean', 'previous_fixed']:
                m = base.metric(g.actual_magnitude_bps, g[f'{side}_{name}'])
                metrics.append(dict(period=period, side=side, model=name, **m))
                effects.append(dict(period=period, side=side, block=name, mse_reduction_vs_anchor=anchor_mse-m['mse_bps2']))
                large = g.actual_magnitude_bps > g[f'{side}_tail_threshold']
                for label, condition in [('ordinary', ~large), ('large', large)]:
                    h = g.loc[condition]
                    if len(h):
                        tm = base.metric(h.actual_magnitude_bps, h[f'{side}_{name}'])
                        tails.append(dict(period=period, side=side, model=name, group=label,
                            squared_error_share=tm['mse_bps2']*len(h)/max(m['mse_bps2']*len(g), 1e-30), **tm))
    table = pd.DataFrame(metrics)
    table.to_csv(out / 'magnitude_metrics.csv', index=False)
    pd.DataFrame(effects).to_csv(out / 'feature_block_effects.csv', index=False)
    pd.DataFrame(tails).to_csv(out / 'tail_errors.csv', index=False)
    summary = ('CAUSAL PRICE-FEATURE CONDITIONAL MEAN MAGNITUDES - ordinary bps\n'
        'No signed forecasts. Historical exposed evaluation. Positive block MSE reduction means improvement.\n'
        + table[table.period.eq('ALL')].to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    base.dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(result), model_replay_passed=True,
        signed_return_forecasts=False, contract_sha256=base.sha(out / 'CONTRACT.json'),
        predictions_sha256=base.sha(out / 'magnitude_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    main()
