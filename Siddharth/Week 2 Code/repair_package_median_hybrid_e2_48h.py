"""User-run E2 residual correction of the frozen package median hybrid."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_package_median_hybrid_signed_48h as median
import lightgbm_repair_experiments_48h as repair

b = median.b
BASE = b.EXP / 'package_median_hybrid_signed_48h_v1/20261113T071027569020Z'


def settings(card):
    return {k: card[k] for k in ['depth', 'leaf', 'max_samples', 'weight']}


def card_id(card):
    return hashlib.sha256(json.dumps(settings(card), sort_keys=True).encode()).hexdigest()[:12]


def side_mask(signed, side):
    return signed > 0 if side == 'UP' else signed < 0


def oof_plan(y, signed, side, stop, minimum=1024):
    months = y.index.tz_convert('UTC').tz_localize(None).to_period('M')
    mask = side_mask(signed, side)
    for month in months.unique():
        cut = month.start_time.tz_localize('UTC')
        if cut >= stop:
            break
        tr = y.index[(y.label_end < cut) & mask]
        va = y.index[(months == month) & mask & (y.label_end < stop)]
        if len(tr) >= minimum and len(va):
            b.check(y.loc[tr, 'label_end'].max() < va.min(), 'OOF label leakage')
            yield cut, tr, va


def predict(bundle, x, features):
    base = median.components(bundle['base'], x)['hybrid']
    if bundle['correction'] is None:
        return base
    return np.maximum(0., base + repair.correction(bundle['correction'], features))


def load():
    mx, my, ms, mr, e2, package, columns, cards = median.load()
    _, features, y, signed, ref, _, audit = repair.load_inputs()
    pd.testing.assert_frame_equal(my, y)
    pd.testing.assert_series_equal(ms, signed)
    b.check(mx.index.equals(features.index) and mr.index.equals(ref.index), 'Input alignment differs')
    done = json.loads((BASE / 'COMPLETE.json').read_text())
    b.check(done['status'] == 'COMPLETE' and done['replay_passed'], 'Baseline incomplete')
    b.check(b.sha(BASE / 'predictions.csv') == done['prediction_sha256'], 'Baseline prediction hash differs')
    baseline = pd.read_csv(BASE / 'predictions.csv', index_col='decision_time')
    baseline.index = pd.to_datetime(baseline.index, utc=True)
    b.check(baseline.index.is_unique and baseline.index.equals(ref.index), 'Baseline population differs')
    for col in ['p_up', 'p_down', 'actual_signed_return_bps']:
        np.testing.assert_allclose(baseline[col], ref[col], atol=1e-7, rtol=0)
    artifacts = {}
    for issue, card in zip(b.ISSUES, cards):
        te = ref.index[ref.issue.eq(issue)]
        for side in ['UP', 'DOWN']:
            path = BASE / f'{issue:%Y-%m}_{side}.joblib'
            receipt = json.loads(path.with_suffix('.json').read_text())
            b.check(b.sha(path) == receipt['sha256'], 'Baseline model hash differs')
            a = joblib.load(path)
            b.check(settings(a['settings']) == settings(card), 'Baseline settings differ')
            b.check(a['max_training_label_end'] < issue, 'Baseline label leakage')
            np.testing.assert_allclose(median.components(a, mx.loc[te])['hybrid'],
                                       baseline.loc[te, f'{side}_hybrid'], atol=1e-7, rtol=0)
            artifacts[(issue, side)] = a
    return mx, features, y, signed, ref, baseline, package, columns, cards, artifacts, audit


def reports(result, y, target, out):
    names = ['hybrid', 'corrected', 'selected', 'E2', 'ridge', 'zero']
    groups = [('ALL', result)] + [(f'Cycle {q}', result[(result.index.year == 2024) &
               (result.index.quarter == q)]) for q in range(1, 5)] + [('Jan2025', result[result.index.year == 2025])]
    rows, sides, tails, comparisons = [], [], [], []
    for period, g in groups:
        if g.empty:
            continue
        actual = g.actual_signed_return_bps
        refs = {name: b.metric(actual, g[f'{name}_signed_bps']) for name in ['hybrid', 'E2', 'ridge']}
        threshold = pd.Series(index=g.index, dtype=float)
        for issue, part in g.groupby('issue'):
            threshold.loc[part.index] = target.loc[y.label_end < issue].quantile(.9)
        tail = actual.abs() >= threshold
        for name in names:
            pred = g[f'{name}_signed_bps']
            m = b.metric(actual, pred)
            gains = {f'{metric}_gain_vs_{base}': values[f'{metric}_bps'] - m[f'{metric}_bps']
                     for base, values in refs.items() for metric in ['rmse', 'mae']}
            rows.append(dict(period=period, model=name, **m, **gains,
                direction_accuracy_pct=float(100 * (median.data.direction(pred) == median.data.direction(actual)).mean())))
            excess = (pred-actual)**2 - (g.hybrid_signed_bps-actual)**2
            tails.append(dict(period=period, model=name, tail_rows=int(tail.sum()),
                excess_mse=float(excess.mean()), tail_excess_contribution=float(excess[tail].sum()/len(g)),
                nontail_excess_contribution=float(excess[~tail].sum()/len(g))))
            if name != 'zero':
                for side in ['UP', 'DOWN']:
                    mask = side_mask(actual, side)
                    sides.append(dict(period=period, model=name, side=side,
                        **b.metric(actual[mask].abs(), g.loc[mask, f'{side}_{name}'])))
    for name in names:
        f = result[['actual_signed_return_bps']].copy()
        f['model'] = name
        f['predicted_return_bps'] = result[f'{name}_signed_bps']
        f['actual_return_pct'] = f.actual_signed_return_bps/100
        f['predicted_return_pct'] = f.predicted_return_bps/100
        f['actual_direction'] = median.data.direction(f.actual_signed_return_bps)
        f['predicted_direction'] = median.data.direction(f.predicted_return_bps)
        f['error_bps'] = f.predicted_return_bps-f.actual_signed_return_bps
        comparisons.append(f)
    table = pd.DataFrame(rows)
    table.to_csv(out/'cycle_signed_metrics.csv', index=False)
    pd.DataFrame(sides).to_csv(out/'cycle_magnitude_metrics.csv', index=False)
    pd.DataFrame(tails).to_csv(out/'tail_error_contributions.csv', index=False)
    pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv', index_label='decision_time')
    mean = table[table.period.isin(['Cycle 2', 'Cycle 3', 'Cycle 4'])].groupby('model')[
        ['rmse_bps', 'mae_bps', 'rmse_gain_vs_hybrid', 'mae_gain_vs_hybrid', 'rmse_gain_vs_E2']].mean()
    mean.to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    cols = ['period', 'model', 'rmse_bps', 'mae_bps', 'rank_ic', 'rmse_gain_vs_hybrid', 'rmse_gain_vs_E2']
    summary = ('E2 TECHNIQUE ON PACKAGE MEDIAN HYBRID\nSquared-error corrections; corrected output is mean-oriented, not a pure median.\n'
               'Exposed historical evaluation. No guaranteed improvement.\n' + table[cols].to_string(index=False) +
               '\n\nCycle2-4 unweighted means:\n' + mean.to_string())
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    return summary


def self_test():
    rng = np.random.default_rng(48)
    idx = pd.date_range('2020-01-01', periods=1700, freq='D', tz='UTC')
    signed = pd.Series(np.where(np.arange(len(idx)) % 2, 100., -100.), index=idx)
    y = pd.DataFrame({'label_end': idx+pd.Timedelta(hours=48)}, index=idx)
    for side in ['UP', 'DOWN']:
        plan = list(oof_plan(y, signed, side, idx[-1], minimum=100))
        assert plan
        for cut, tr, va in plan:
            assert y.loc[tr, 'label_end'].max() < cut <= va.min()
            assert y.loc[va, 'label_end'].max() < idx[-1]
            assert side_mask(signed.loc[tr], side).all()
        stop = idx[900]
        prefix = list(oof_plan(y.loc[:stop], signed.loc[:stop], side, stop, minimum=100))
        full = list(oof_plan(y, signed, side, stop, minimum=100))
        assert len(prefix) == len(full)
        for a, c in zip(prefix, full):
            assert a[0] == c[0] and a[1].equals(c[1]) and a[2].equals(c[2])
    package, cols, cards = median.package_spec()
    x = pd.DataFrame(rng.normal(size=(700, 7)), columns=cols)
    target = pd.Series(100 + 10*x.iloc[:, 0] + rng.exponential(5, len(x)))
    a = median.fit(x, target, x.index[:500], cols, cards[0], package['seeds'], 1)
    xf = pd.DataFrame(rng.normal(size=(700, 5)), index=pd.date_range('2020-01-01', periods=700, freq='D', tz='UTC'))
    residual = pd.Series(10*xf.iloc[:, 0] + rng.normal(size=700), index=xf.index)
    yy = pd.DataFrame({'label_end': xf.index+pd.Timedelta(hours=48)}, index=xf.index)
    cfg = dict(id='synthetic', pack='long', days=None, params=repair.old.schedule()[0])
    c = repair.fit_candidate(xf, residual, yy, xf.index[-1]+pd.Timedelta(days=3), cfg, 1, repair.old.lightgbm())
    assert c['early_train_max_label_end'] < c['early_validation_start']
    bundle = dict(base=a, correction=c)
    expected = np.maximum(0, median.components(a, x.iloc[-50:])['hybrid'] + repair.correction(c, xf.iloc[-50:]))
    np.testing.assert_allclose(predict(bundle, x.iloc[-50:], xf.iloc[-50:]), expected)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)/'bundle.joblib'
        joblib.dump(bundle, p)
        np.testing.assert_allclose(predict(joblib.load(p), x.iloc[-50:], xf.iloc[-50:]), expected)
    np.testing.assert_allclose(predict(dict(base=a, correction=None), x, xf), median.components(a, x)['hybrid'])
    print('PASS: monthly OOF side/maturity/prefix checks, correction early-stop purge, additive composition, exact fallback, saved bundle replay.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--resume', type=Path)
    p.add_argument('--check-data-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    b.check(args.threads > 0, 'Positive threads required')
    if args.self_test:
        self_test()
        return
    x, features, y, signed, ref, baseline, package, columns, cards, artifacts, audit = load()
    target = signed.abs()
    if args.check_data_only:
        print(f'PASS: {len(x)} rows, {len(features.columns)} E2 correction features, {len(ref)} origins; all eight frozen hybrid bundles replayed. No fitting.')
        return
    lgb = repair.old.lightgbm()
    configs = [dict(id=c['id'], pack='long', days=None, params=c) for c in repair.old.schedule()]
    modules = [median, median.data, b, repair, repair.old, repair.price, repair.price.canonical,
               repair.direction_source, repair.direction_source.old]
    files = [Path(__file__)] + [Path(m.__file__) for m in modules]
    files += [median.data.PREPARED, median.data.DATA/'X_train.parquet', median.data.DATA/'X_development.parquet',
              median.data.REF/'signed_return_predictions.csv', repair.price.WEEK1/'Dataset/canonical_returns_5m.parquet',
              median.E2/'COMPLETE.json', median.E2/'predictions.csv', BASE/'COMPLETE.json', BASE/'CONTRACT.json', BASE/'predictions.csv',
              median.PACKAGE/'config.json', median.PACKAGE/'features.csv']
    files += list(BASE.glob('*.joblib')) + list(BASE.glob('2024-*.json'))
    files += [median.PACKAGE/f'models/fold_{i}_selected.json' for i in range(1, 5)]
    files += [repair.direction_source.DIR/name for name in ['COMPLETE.json', 'validation_predictions.csv', 'development_predictions.csv']]
    identity = dict(hashes={str(q): b.sha(q) for q in files}, configs=configs, threads=args.threads,
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']}, lightgbm=lgb.__version__)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = args.resume or b.EXP/'package_median_hybrid_e2_repair_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not (out/'COMPLETE.json').exists(), 'Run already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature'] == signature, 'Resume signature mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        b.dump(out/'CONTRACT.json', dict(signature=signature, identity=identity, input_audit=audit,
            baseline=str(BASE), features=list(features.columns), package_cards=cards,
            oof='Monthly expanding side-specific package hybrid; minimum1024 rows; labels mature before month. For each outer issue, its fixed package card is used consistently for all OOF and inner fits. Shared identical cards reuse cache.',
            correction='Same E2 235 features,15 LightGBM L2 configurations,expanding monthly OOF errors,early-stop last20percent purged48h,1000cap/50patience; then refit matured residual history.',
            selection='Per-side pooled magnitude MSE in three earlier two-month windows at offsets9/6/3months. corrected always uses best correction; selected uses it only if earlier MSE beats unchanged hybrid. No outer selection or shrinkage search.',
            target='Positive ordinary-bps endpoint magnitude by realized side. Corrected magnitude=max(0,median hybrid+L2 correction); mean-oriented, no longer a pure median.',
            limitations='Historical package cards inherited, not recreated by nested selection. Monthly OOF is leakage-safe conditional on those fixed cards, not fresh confirmation of their original selection. Repeatedly exposed evaluation; no guarantee. No E3/E4 history or joint-policy changes.'))
        for q in files:
            if q.suffix == '.py':
                (out/q.name).write_bytes(q.read_bytes())
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
    frames, selections, ledger, oof_audit = [], [], [], []
    print('Output:', out, flush=True)
    with threadpool_limits(limits=args.threads):
        for issue, card in zip(b.ISSUES, cards):
            cid = card_id(card)
            windows = []
            for offset in [9, 6, 3]:
                cut = issue-pd.DateOffset(months=offset)
                va = y.index[(y.index >= cut) & (y.index < cut+pd.DateOffset(months=2))]
                b.check(y.loc[va, 'label_end'].max() < issue, 'Selection labels not mature')
                windows.append((cut, va))
            vk = windows[0][1].append([v for _, v in windows[1:]])
            te = ref.index[ref.issue.eq(issue)]
            frame = baseline.loc[te, ['p_up', 'p_down', 'actual_signed_return_bps',
                'UP_hybrid', 'DOWN_hybrid', 'hybrid_signed_bps', 'UP_E2', 'DOWN_E2', 'E2_signed_bps',
                'UP_ridge', 'DOWN_ridge', 'ridge_signed_bps']].copy()
            frame['issue'] = issue
            frame['symbol'] = 'BTCUSDT'
            frame['horizon_hours'] = 48
            frame['forecast_end'] = te+pd.Timedelta(hours=48)
            frame['zero_signed_bps'] = 0.
            for side in ['UP', 'DOWN']:
                mask = side_mask(signed, side)
                def base_at(cut):
                    keys = b.train_keys(y, mask, cut, None)
                    return cached(f'Base_{cid}_{side}_{cut:%Y%m%d}',
                        lambda: median.fit(x, target, keys, columns, card, package['seeds'], args.threads))
                residual = pd.Series(np.nan, index=x.index, dtype=float)
                for cut, tr, va in oof_plan(y, signed, side, issue):
                    print(f'OOF {issue.date()} {side} month={cut:%Y-%m}', flush=True)
                    monthly = cached(f'OOF_{cid}_{side}_{cut:%Y%m%d}', lambda cut=cut, va=va:
                        pd.Series(median.components(base_at(cut), x.loc[va])['hybrid'], index=va))
                    # A later issue can extend the last month's matured subset.
                    if not va.isin(monthly.index).all():
                        monthly = pd.Series(median.components(base_at(cut), x.loc[va])['hybrid'], index=va)
                    residual.loc[va] = target.loc[va]-monthly.loc[va]
                    oof_audit.append(dict(issue=issue, side=side, card_id=cid, cutoff=cut,
                        training_rows=len(tr), rows=len(va), max_training_label_end=y.loc[tr, 'label_end'].max()))
                b.dump(out/'OOF_AUDIT.json', oof_audit)
                raw = np.concatenate([median.components(base_at(cut), x.loc[va])['hybrid'] for cut, va in windows])
                take = mask.loc[vk].to_numpy()
                base_score = b.metric(target.loc[vk[take]], raw[take])['mse_bps2']
                scores = []
                for cfg in configs:
                    predictions = []
                    for cut, va in windows:
                        a = cached(f'Correction_{cid}_{side}_{cfg["id"]}_{cut:%Y%m%d}',
                            lambda cut=cut, cfg=cfg: repair.fit_candidate(features, residual, y, cut, cfg, args.threads, lgb))
                        predictions.append(repair.correction(a, features.loc[va]))
                    pred = np.maximum(0, raw+np.concatenate(predictions))
                    m = b.metric(target.loc[vk[take]], pred[take])
                    scores.append(m['mse_bps2'])
                    ledger.append(dict(issue=issue, side=side, candidate=cfg['id'], baseline_mse=base_score, **m))
                    pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv', index=False)
                    print(issue.date(), side, cfg['id'], 'validation MSE', scores[-1], flush=True)
                chosen = min(range(len(configs)), key=lambda i: (scores[i], configs[i]['id']))
                cfg = configs[chosen]
                correction = cached(f'Correction_{cid}_{side}_{cfg["id"]}_{issue:%Y%m%d}',
                    lambda: repair.fit_candidate(features, residual, y, issue, cfg, args.threads, lgb))
                use = scores[chosen] < base_score-1e-9
                for name, c in [('corrected', correction), ('selected', correction if use else None)]:
                    bundle = dict(base=artifacts[(issue, side)], correction=c, issue=issue, side=side,
                        card=card, candidate=cfg if c is not None else None)
                    path = out/f'{issue:%Y-%m}_{side}_{name}.joblib'
                    joblib.dump(bundle, path, compress=3)
                    pred = predict(bundle, x.loc[te], features.loc[te])
                    np.testing.assert_allclose(pred, predict(joblib.load(path), x.loc[te], features.loc[te]), atol=1e-7, rtol=0)
                    frame[f'{side}_{name}'] = pred
                    selections.append(dict(issue=issue, side=side, model=name, corrected=c is not None,
                        candidate=cfg['id'] if c is not None else 'unchanged', baseline_validation_mse=base_score,
                        correction_validation_mse=scores[chosen], best_iteration=c['best_iteration'] if c else 0,
                        max_residual_label_end=c['max_training_label_end'] if c else None,
                        artifact=path.name, sha256=b.sha(path), replay_passed=True))
                b.dump(out/'SELECTIONS_AND_FIT_AUDIT.json', selections)
            for name in ['corrected', 'selected']:
                frame[f'{name}_signed_bps'] = frame.p_up*frame[f'UP_{name}']-frame.p_down*frame[f'DOWN_{name}']
            frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv', index_label='decision_time')
            frames.append(frame)
    result = pd.concat(frames).sort_index()
    b.check(result.index.equals(ref.index), 'Final origin mismatch')
    result.to_csv(out/'predictions.csv', index_label='decision_time')
    summary = reports(result, y, target, out)
    b.dump(out/'COMPLETE.json', dict(status='COMPLETE', rows=len(result), replay_passed=True,
        prediction_sha256=b.sha(out/'predictions.csv'), signature=signature))
    print(summary+'\nSaved to '+str(out), flush=True)


if __name__ == '__main__':
    main()
