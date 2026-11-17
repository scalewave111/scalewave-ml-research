"""Fast historical-forecast correction. No new median/forest fits."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import repair_package_median_hybrid_e2_48h as full

b = full.b
repair = full.repair


def residual_history(y, signed, baseline, side):
    result = pd.Series(np.nan, index=y.index, dtype=float)
    keys = baseline.index[full.side_mask(signed.loc[baseline.index], side)]
    result.loc[keys] = signed.loc[keys].abs()-baseline.loc[keys, f'{side}_hybrid']
    return result


def selection_window(y, residual, issue):
    # Config selection is separate from fit_candidate's internal early-stop slice.
    cut = issue-pd.Timedelta(days=30)
    tr = y.index[(y.label_end < cut) & residual.notna()]
    va = y.index[(y.index >= cut) & (y.label_end < issue) & residual.notna()]
    return cut, tr, va


def self_test():
    idx = pd.date_range('2024-01-01', periods=160*48, freq='30min', tz='UTC')
    y = pd.DataFrame({'label_end': idx+pd.Timedelta(hours=48)}, index=idx)
    signed = pd.Series(np.where(np.arange(len(idx)) % 2, 200., -100.), index=idx)
    baseline = pd.DataFrame({'UP_hybrid': 150., 'DOWN_hybrid': 120.}, index=idx)
    issue = pd.Timestamp('2024-04-01', tz='UTC')
    for side, expected in [('UP', 50.), ('DOWN', -20.)]:
        residual = residual_history(y, signed, baseline, side)
        assert (residual.dropna() == expected).all()
        assert residual.notna().equals(full.side_mask(signed, side))
        cut, tr, va = selection_window(y, residual, issue)
        assert len(tr) >= 500 and len(va) >= 100
        assert y.loc[tr, 'label_end'].max() < cut <= va.min()
        assert y.loc[va, 'label_end'].max() < issue
        changed = baseline.copy()
        changed.loc[changed.index >= issue] = 1e9
        other = residual_history(y, signed, changed, side)
        pd.testing.assert_series_equal(residual.loc[y.label_end < issue], other.loc[y.label_end < issue])
        assert not len(selection_window(y, residual, pd.Timestamp('2024-01-01', tz='UTC'))[1])
    print('PASS: saved forecast residuals, side masks, config holdout purge, future invariance, January warmup fallback. No fitting.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-data-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    b.check(args.threads > 0, 'Positive threads required')
    if args.self_test:
        self_test()
        return
    x, features, y, signed, ref, baseline, package, columns, cards, artifacts, audit = full.load()
    residuals = {side: residual_history(y, signed, baseline, side) for side in ['UP', 'DOWN']}
    if args.check_data_only:
        for issue in b.ISSUES:
            for side in residuals:
                cut, tr, va = selection_window(y, residuals[side], issue)
                print(issue.date(), side, 'correction training/selection rows:', len(tr), len(va))
        print(f'PASS: {len(ref)} origins, {len(features.columns)} features; eight saved hybrid bundles replayed; no fitting.')
        return
    lgb = repair.old.lightgbm()
    specs = [(3, 4, .01), (3, 4, .03), (4, 8, .02)]
    configs = [dict(id=c['id'], pack='long', days=None, params=c) for c in repair.old.schedule()
               if (c['max_depth'], c['num_leaves'], c['learning_rate']) in specs]
    files = [Path(__file__), Path(full.__file__), Path(repair.__file__), Path(repair.old.__file__),
             Path(full.median.__file__), Path(full.median.data.__file__), Path(b.__file__),
             Path(repair.price.__file__), Path(repair.price.canonical.__file__),
             Path(repair.direction_source.__file__), Path(repair.direction_source.old.__file__),
             full.median.data.PREPARED, full.median.data.DATA/'X_train.parquet',
             full.median.data.DATA/'X_development.parquet', full.median.data.REF/'signed_return_predictions.csv',
             repair.price.WEEK1/'Dataset/canonical_returns_5m.parquet',
             full.median.E2/'COMPLETE.json', full.median.E2/'predictions.csv',
             full.BASE/'COMPLETE.json', full.BASE/'CONTRACT.json', full.BASE/'predictions.csv',
             full.median.PACKAGE/'config.json', full.median.PACKAGE/'features.csv']
    files += list(full.BASE.glob('*.joblib')) + list(full.BASE.glob('2024-*.json'))
    files += [full.median.PACKAGE/f'models/fold_{i}_selected.json' for i in range(1, 5)]
    identity = dict(hashes={str(p): b.sha(p) for p in files}, threads=args.threads, configs=configs,
        lightgbm=lgb.__version__, versions={name: importlib.metadata.version(name)
        for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']})
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = args.resume or b.EXP/'package_median_hybrid_fast_repair_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not (out/'COMPLETE.json').exists(), 'Run already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature'] == signature, 'Resume signature mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        b.dump(out/'CONTRACT.json', dict(signature=signature, identity=identity, input_audit=audit,
            method='Saved quarterly out-of-time hybrid errors, not monthly refitted OOF errors. Zero new base fits. Same235E2 features and L2 correction fitter,three configs.',
            selection='Last30days before each issue, with48h maturity purge. Correction trained strictly before that window; its own earlier20percent purged slice chooses iterations. Lowest holdout side MSE chooses config. Refit chosen correction on all mature prior forecast errors.',
            warmup='No saved pre2024 forecasts: Cycle1 unchanged. Need>=500 preselection residuals and>=100 selection residuals, otherwise unchanged.',
            regimes='Historical residuals come from each previous quarters actual card; current outer base retains its own original card. Card changes can weaken transfer.',
            policies='corrected best validation config; selected keeps unchanged side if correction fails to beat holdout baseline MSE. No claim of future improvement.',
            limitations='Different, faster experiment than full E2 transfer. Historical exposed evaluation and inherited selected package cards. Corrected output mean-oriented, not pure median. All comparisons same origins; eligible-period metrics additionally exclude warmup.'))
        for p in files:
            if p.suffix == '.py':
                (out/p.name).write_bytes(p.read_bytes())
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    def cached(name, compute):
        path = cache/(name+'.joblib')
        receipt = path.with_suffix('.json')
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256'] == b.sha(path):
            return joblib.load(path)
        value = compute()
        joblib.dump(value, path, compress=3)
        b.dump(receipt, dict(sha256=b.sha(path)))
        return value
    frames, ledger, selections = [], [], []
    print('Output:', out, flush=True)
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            frame = baseline.loc[te].copy()
            frame['issue'] = issue
            for side in ['UP', 'DOWN']:
                residual = residuals[side]
                cut, tr, va = selection_window(y, residual, issue)
                eligible = len(tr) >= 500 and len(va) >= 100
                frame[f'{side}_correction_eligible'] = eligible
                chosen, model, use, base_mse, chosen_mse = None, None, False, None, None
                if eligible:
                    raw = baseline.loc[va, f'{side}_hybrid'].to_numpy()
                    truth = signed.loc[va].abs().to_numpy()
                    base_mse = float(np.mean((raw-truth)**2))
                    scores = []
                    for cfg in configs:
                        a = cached(f'Validation_{issue:%Y%m}_{side}_{cfg["id"]}',
                            lambda cfg=cfg: repair.fit_candidate(features, residual, y, cut, cfg, args.threads, lgb))
                        pred = np.maximum(0, raw+repair.correction(a, features.loc[va]))
                        m = b.metric(truth, pred)
                        scores.append(m['mse_bps2'])
                        ledger.append(dict(issue=issue, side=side, config=cfg['id'], baseline_mse=base_mse, **m))
                        pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv', index=False)
                        print(issue.date(), side, cfg['id'], 'selection MSE', scores[-1], flush=True)
                    i = min(range(len(configs)), key=lambda i: (scores[i], configs[i]['id']))
                    chosen, chosen_mse = configs[i], scores[i]
                    use = chosen_mse < base_mse-1e-9
                    model = cached(f'Final_{issue:%Y%m}_{side}_{chosen["id"]}',
                        lambda: repair.fit_candidate(features, residual, y, issue, chosen, args.threads, lgb))
                for name, correction in [('corrected', model), ('selected', model if use else None)]:
                    bundle = dict(base=artifacts[(issue, side)], correction=correction, issue=issue, side=side)
                    path = out/f'{issue:%Y-%m}_{side}_{name}.joblib'
                    joblib.dump(bundle, path, compress=3)
                    pred = full.predict(bundle, x.loc[te], features.loc[te])
                    np.testing.assert_allclose(pred, full.predict(joblib.load(path), x.loc[te], features.loc[te]), atol=1e-7, rtol=0)
                    if correction is None:
                        np.testing.assert_allclose(pred, baseline.loc[te, f'{side}_hybrid'], atol=1e-7, rtol=0)
                    frame[f'{side}_{name}'] = pred
                    selections.append(dict(issue=issue, side=side, model=name, eligible=eligible,
                        preselection_rows=len(tr), selection_rows=len(va), selection_start=cut,
                        validation_mse=chosen_mse, baseline_validation_mse=base_mse,
                        config=chosen if correction is not None else None,
                        max_residual_label_end=correction['max_training_label_end'] if correction else None,
                        best_iteration=correction['best_iteration'] if correction else 0,
                        artifact=path.name, sha256=b.sha(path), replay_passed=True))
                b.dump(out/'SELECTIONS_AND_FIT_AUDIT.json', selections)
            for name in ['corrected', 'selected']:
                frame[f'{name}_signed_bps'] = frame.p_up*frame[f'UP_{name}']-frame.p_down*frame[f'DOWN_{name}']
            frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv', index_label='decision_time')
            frames.append(frame)
    result = pd.concat(frames).sort_index()
    b.check(result.index.equals(ref.index), 'Final origin mismatch')
    result.to_csv(out/'predictions.csv', index_label='decision_time')
    summary = full.reports(result, y, signed.abs(), out)
    eligible = result.UP_correction_eligible & result.DOWN_correction_eligible
    rows = [dict(model=name, **b.metric(result.loc[eligible, 'actual_signed_return_bps'],
            result.loc[eligible, f'{name}_signed_bps'])) for name in ['hybrid', 'corrected', 'selected', 'E2', 'ridge']]
    pd.DataFrame(rows).to_csv(out/'eligible_period_signed_metrics.csv', index=False)
    summary = summary.replace('E2 TECHNIQUE ON PACKAGE MEDIAN HYBRID', 'FAST SAVED-FORECAST CORRECTION OF PACKAGE MEDIAN HYBRID')
    summary += '\nCycle1 is warmup/unchanged; correction uses prior saved quarterly forecasts, not monthly refits.\n'
    summary += '\nEligible-period metrics:\n'+pd.DataFrame(rows)[['model', 'rows', 'rmse_bps', 'mae_bps', 'rank_ic']].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    b.dump(out/'COMPLETE.json', dict(status='COMPLETE', rows=len(result), replay_passed=True,
        prediction_sha256=b.sha(out/'predictions.csv'), signature=signature))
    print(summary+'\nSaved to '+str(out), flush=True)


if __name__ == '__main__':
    main()
