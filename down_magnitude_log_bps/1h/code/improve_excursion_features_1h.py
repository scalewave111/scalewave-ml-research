r"""Follow-up 1h excursion search: causal price blocks against saved F4Z controls.
python .\Code\improve_excursion_features_1h.py
Full training is user-run; --check-inputs only reads/validates inputs.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import discover_excursion_features_1h as base

WEEK1 = base.HERE.parents[1] / 'Week 1'
sys.path.insert(0, str(WEEK1/'Code'))
import canonical_return_features as canonical

HOURS = [1, 4, 12, 24, 48, 168]


def price_features(r, origins):
    """Endpoint-indexed 5m log returns; only endpoints strictly before origin.
    Path features use 5m closes, not intrabar highs/lows. Missing windows stay NaN.
    """
    values, groups = {}, {}
    log_level = r.fillna(0).cumsum().shift(1)
    for h in HOURS:
        n = h*12
        opts = dict(window=f'{h}h', closed='left', min_periods=n)
        valid = r.rolling(**opts).count().eq(n)
        rv = np.sqrt(r.pow(2).rolling(**opts).sum())
        net = r.rolling(**opts).sum()
        hi = log_level.rolling(n+1, min_periods=n+1).max()
        lo = log_level.rolling(n+1, min_periods=n+1).min()
        width = hi-lo
        stats = dict(rv=rv, up_rv=np.sqrt(r.clip(lower=0).pow(2).rolling(**opts).sum()),
                     down_rv=np.sqrt(r.clip(upper=0).pow(2).rolling(**opts).sum()),
                     net=net, past_range=width, distance_from_high=hi-log_level,
                     distance_from_low=log_level-lo,
                     max_abs_return=r.abs().rolling(**opts).max())
        groups[f'price_{h}h'] = []
        for name, v in stats.items():
            col = f'NEWPRICE__{name}__{h}h'
            values[col] = v.where(valid)
            groups[f'price_{h}h'].append(col)
    groups['volatility_ratios'] = []
    for short, long in [(1, 24), (4, 24), (24, 168)]:
        col = f'NEWPRICE__rv_ratio__{short}h_{long}h'
        values[col] = (values[f'NEWPRICE__rv__{short}h']/np.sqrt(short) /
                      (values[f'NEWPRICE__rv__{long}h']/np.sqrt(long)).clip(lower=1e-6))
        groups['volatility_ratios'].append(col)
    return pd.DataFrame(values).reindex(origins).replace([np.inf, -np.inf], np.nan), groups


def causal_check(r, origins):
    cut = origins[len(origins)//2]
    keys = origins[origins <= cut][-10:]
    original, _ = price_features(r, keys)
    prefix, _ = price_features(r.loc[r.index <= cut], keys)
    pd.testing.assert_frame_equal(original, prefix)
    changed = r.copy()
    changed.loc[changed.index >= cut] = 100000.
    a, _ = price_features(r, pd.DatetimeIndex([cut]))
    b, _ = price_features(changed, pd.DatetimeIndex([cut]))
    pd.testing.assert_frame_equal(a, b)


def gate(new, old, weights, min_gain):
    return (np.average(new, weights=weights) < np.average(old, weights=weights)*(1-min_gain)
            and int((np.asarray(new) < np.asarray(old)).sum()) >= 2)


def search(x, y, folds, anchor, groups, args, folder):
    """Selection reads inner rows only; controls retain the old model and inputs."""
    cache, trials, quality = {}, [], []
    allowed = {}
    for group, columns in groups.items():
        admitted = []
        for c in columns:
            ok = all(x.iloc[tr][c].notna().mean() >= .8 and x.iloc[tr][c].notna().sum() >= 1000
                     and x.iloc[tr][c].nunique() > 1 for tr, _ in folds)
            quality.append(dict(group=group, feature=c, eligible=ok))
            if ok:
                admitted.append(c)
        if admitted:
            allowed[group] = admitted
    pd.DataFrame(quality).to_csv(folder/'new_feature_quality.csv', index=False)
    weights = [len(va) for _, va in folds]

    def evaluate(cols, kind, name):
        key = (tuple(cols), kind)
        if key not in cache:
            a = x[cols].to_numpy(float)
            vals = []
            for tr, va in folds:
                m = base.fit(a[tr], y[tr], kind)
                vals.append(float(np.abs(y[va]-base.predict(m, a[va])).mean()))
            cache[key] = np.array(vals)
        v = cache[key]
        trials.append(dict(variant=name, model=kind, features=json.dumps(cols), pooled_mae=float(np.average(v, weights=weights)),
                           **{f'window_{i+1}_mae': float(z) for i, z in enumerate(v)}))
        pd.DataFrame(trials).to_csv(folder/'validation_trials.csv', index=False)
        return v

    old_cols, kind = anchor['features'], anchor['model']
    original = evaluate(old_cols, kind, 'original')
    median = evaluate([], 'median', 'median')
    variants = {'original': dict(features=old_cols, model=kind, errors=original)}
    current, selected_groups = original, []
    for step in range(args.max_blocks):
        candidates = []
        for group, cols in allowed.items():
            if group in selected_groups:
                continue
            chosen_groups = selected_groups+[group]
            names = old_cols + [c for g in chosen_groups for c in allowed[g]]
            tag = 'add_' + '+'.join(chosen_groups)
            val = evaluate(names, 'tree', tag)
            variants[tag] = dict(features=names, model='tree', errors=val)
            if gate(val, current, weights, args.min_gain):
                candidates.append((float(np.average(val, weights=weights)), group, val))
        if not candidates:
            break
        _, g, current = min(candidates, key=lambda t: (t[0], t[1]))
        selected_groups.append(g)
        print(f'  added block {g}, validation MAE {np.average(current, weights=weights):.4f}', flush=True)
    # Independent wide and replacement controls allow new features without old anchors.
    all_new = [c for columns in allowed.values() for c in columns]
    for tag, names in [('all_price_plus_original', old_cols+all_new), ('price_only', all_new)]:
        val = evaluate(names, 'tree', tag)
        variants[tag] = dict(features=names, model='tree', errors=val)
    # Group removal diagnostics on the forward-search endpoint.
    endpoint = old_cols + [c for g in selected_groups for c in allowed[g]]
    endpoint_mae = float(np.average(current, weights=weights))
    removals = []
    for group in ['original_features']+selected_groups:
        remove = old_cols if group == 'original_features' else allowed[group]
        names = [c for c in endpoint if c not in remove]
        tag = 'remove_' + group
        removal_kind = 'tree' if selected_groups else kind
        val = evaluate(names, removal_kind, tag)
        variants[tag] = dict(features=names, model=removal_kind, errors=val)
        removals.append(dict(group=group, removal_mae_change=float(np.average(val, weights=weights))-endpoint_mae))
    pd.DataFrame(removals).to_csv(folder/'removal_checks.csv', index=False)
    choices = [tag for tag, v in variants.items() if tag != 'original'
               and gate(v['errors'], original, weights, args.min_gain)
               and gate(v['errors'], median, weights, args.min_gain)]
    chosen = min(choices, key=lambda tag: (np.average(variants[tag]['errors'], weights=weights), tag)) if choices else 'original'
    specs = {tag: dict(features=v['features'], model=v['model'], validation_window_mae=v['errors'].tolist(),
                       validation_mae=float(np.average(v['errors'], weights=weights))) for tag, v in variants.items()}
    result = dict(selected_variant=chosen, forward_groups=selected_groups, variants=specs)
    base.dump(folder/'selection.json', result)
    return result


def inputs(args):
    x, y, dev, files = base.load(args)
    # Parquet/CSV readers can preserve different timestamp resolutions.
    x.index = x.index.as_unit('ns')
    y.index = y.index.as_unit('ns')
    dev = dev.as_unit('ns')
    done = json.loads((args.reference_run/'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and done['predictions_sha256'] == base.sha(args.reference_run/'predictions.csv'), 'Reference prediction integrity failed')
    contract = json.loads((args.reference_run/'CONTRACT.json').read_text())
    base.check(contract['horizon_minutes'] == 60, 'Reference must use 1-hour targets')
    base.check(contract['source_sha256'] == base.sha(Path(base.__file__)), 'Original runner changed; exact control replay requires original helper')
    old = pd.read_csv(args.reference_run/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    base.check(len(old) == 2*len(dev) and not old.duplicated(['decision_time', 'target']).any(), 'Reference coverage changed')
    for target in base.TARGETS:
        f = old[old.target.eq(target)].set_index('decision_time').sort_index()
        base.check(f.index.equals(dev), 'Reference keys differ')
        np.testing.assert_allclose(f.actual, y.loc[dev, target], rtol=0, atol=1e-10)
    r = canonical.load_returns(args.returns)
    causal_check(r, x.index)
    derived, groups = price_features(r, x.index)
    x = x.join(derived)
    anchors = json.loads((args.reference_run/'selections.json').read_text())
    # Validate reference cards against the saved fitted bundles before using them.
    for issue in base.ISSUES:
        for target in base.TARGETS:
            cards = [s for s in anchors if pd.Timestamp(s['issue']) == issue and s['target'] == target]
            base.check(len(cards) == 1 and cards[0]['model'] in ['tree', 'ridge', 'median'], 'Expected one saved 1h model card per issue/target')
            folder = args.reference_run/f'{issue:%Y-%m}_{target}'
            expected = json.loads((folder/'MODEL_HASH.json').read_text())['sha256']
            base.check(expected == base.sha(folder/'model.joblib'), 'Reference model integrity failed')
            bundle = joblib.load(folder/'model.joblib')
            base.check(bundle['features'] == cards[0]['features'] and bundle['target'] == target
                       and pd.Timestamp(bundle['issue']) == issue, 'Reference card/model mismatch')
    for issue in base.ISSUES:
        base.windows(x.index, issue)
    files += [args.returns, args.reference_run/'predictions.csv', args.reference_run/'selections.json',
              Path(base.__file__), Path(canonical.__file__), Path(__file__)]
    print(f'Checked {len(x)} rows, {len(dev)} origins and {len(derived.columns)} new price features.', flush=True)
    return x, y, dev, old, anchors, groups, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=base.DATA)
    p.add_argument('--y-dir', type=Path, default=base.CANONICAL/'y/horizon_minutes=60/symbol=BTCUSDT')
    p.add_argument('--returns', type=Path, default=WEEK1/'Dataset/canonical_returns_5m.parquet')
    p.add_argument('--reference-run', type=Path, required=True, help='Completed 1h feature-search run directory')
    p.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/excursion_price_followup_1h_v1')
    p.add_argument('--max-blocks', type=int, default=2)
    p.add_argument('--min-gain', type=float, default=.001)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args()
    base.check(1 <= args.max_blocks <= 7 and 0 <= args.min_gain < 1 and args.threads > 0, 'Invalid settings')
    x, y, dev, old, anchors, groups, files = inputs(args)
    if args.check_inputs:
        print('Input and causal checks passed. No training performed.')
        return
    out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    base.dump(out/'CONTRACT.json', dict(arguments=vars(args), targets=base.TARGETS, horizon_minutes=60,
        groups=groups, source_hashes={str(f): base.sha(f) for f in files},
        method='Same original quarterly expanding walk-forward keys and 1h purges; original selected-model control; greedy additions up to2 blocks by default, all-price plus original, price-only, and group removal controls; earlier-only MAE selection',
        gate='At least min_gain pooled MAE reduction versus original AND median, wins in>=2/3 earlier windows against each; otherwise original',
        model='Original absolute-error HistGradientBoosting settings unchanged; all tried variants evaluated on outer rows only after selection frozen',
        timing='5m return endpoints strictly before origin; historical ranges reconstructed from5m close log levels, not intrabar extremes',
        limitations=['Historical exposed history; new search also reuses inner validation; no independent confirmation',
                    'Original feature lists reused as historically selected anchors; no new nested selection claim',
                    'Outer scores for tried variants are diagnostic, never used by selector',
                    'Original X masking/provisional timing limits inherited',
                    'No guarantee that earlier validation improvement transfers', 'No automatic resume'],
        instruction_hashes={str(f):base.sha(f) for f in [base.HERE.parent/'PROJECT_CONTEXT.md',
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=base.sklearn.__version__)))
    for path in [Path(__file__), Path(base.__file__), Path(canonical.__file__)]:
        (out/path.name).write_bytes(path.read_bytes())
    frames, selected_rows, ledger = [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            folds = base.windows(x.index, issue)
            tr = np.flatnonzero(x.index + pd.Timedelta(hours=1) < issue)
            for target in base.TARGETS:
                print(f'{issue:%Y-%m} {target}', flush=True)
                ref = old[old.issue.eq(issue) & old.target.eq(target)].set_index('decision_time').sort_index()
                te = x.index.get_indexer(ref.index)
                base.check(len(te) > 0 and (te >= 0).all(), 'Missing reference outer rows')
                anchor = next(s for s in anchors if pd.Timestamp(s['issue']) == issue and s['target'] == target)
                folder = out/f'{issue:%Y-%m}_{target}'
                folder.mkdir()
                truth = y[target].to_numpy()
                plan = search(x, truth, folds, anchor, groups, args, folder)
                f = pd.DataFrame(dict(decision_time=ref.index, issue=issue, target=target, actual=truth[te],
                                     training_mean=float(truth[tr].mean()), training_median=float(np.median(truth[tr]))))
                predictions = {}
                for num, (tag, spec) in enumerate(plan['variants'].items()):
                    a = x[spec['features']].to_numpy(float)
                    model = base.fit(a[tr], truth[tr], spec['model'])
                    prediction = base.predict(model, a[te])
                    path = folder/f'model_{num:02d}.joblib'
                    joblib.dump(dict(model=model, features=spec['features'], variant=tag), path)
                    bundle = joblib.load(path)
                    np.testing.assert_allclose(prediction, base.predict(bundle['model'], a[te]), rtol=0, atol=1e-10)
                    if tag == 'original':
                        np.testing.assert_allclose(prediction, ref.prediction, rtol=0, atol=1e-7, err_msg='Original model failed replay')
                    predictions[tag] = prediction
                    f[tag] = prediction
                    spec['artifact'], spec['artifact_sha256'] = path.name, base.sha(path)
                f['selected'] = predictions[plan['selected_variant']]
                base.dump(folder/'selection.json', plan)
                f.to_csv(folder/'predictions.csv', index=False)
                frames.append(f)
                selected_rows.append(dict(issue=str(issue), target=target, variant=plan['selected_variant'],
                    features=plan['variants'][plan['selected_variant']]['features']))
                for window, train, validation in [('outer', tr, te)] + [(str(i+1), a, b) for i, (a, b) in enumerate(folds)]:
                    ledger.append(dict(issue=issue, target=target, window=window, training_rows=len(train),
                        training_label_end_max=(x.index[train]+pd.Timedelta(hours=1)).max(),
                        validation_start=x.index[validation].min(), evaluation_rows=len(validation)))
    f = pd.concat(frames, ignore_index=True)
    base.check(len(f) == 36962 and not f.duplicated(['decision_time', 'target']).any(), 'Output coverage changed')
    f['period'] = f.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    f.loc[f.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    f.to_csv(out/'predictions.csv', index=False)
    rows = []
    models = [c for c in f if c not in ['decision_time', 'issue', 'target', 'actual', 'period']]
    for target, g in f.groupby('target'):
        for period, sub in [('ALL', g)] + list(g.groupby('period')):
            for model in models:
                good = sub[model].notna()
                if not good.any():
                    continue
                h = sub.loc[good]
                measured = base.metric(h.actual.to_numpy(), h[model].to_numpy())
                control = base.metric(h.actual.to_numpy(), h.original.to_numpy())
                rows.append(dict(target=target, period=period, model=model, **measured,
                    full_period_coverage=bool(good.all()), mae_gain_vs_original=control['mae']-measured['mae'],
                    rmse_gain_vs_original=control['rmse']-measured['rmse']))
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'metrics_all_variants.csv', index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv', index=False)
    base.dump(out/'selections.json', selected_rows)
    pd.DataFrame([dict(issue=s['issue'], target=s['target'], feature=c) for s in selected_rows for c in s['features']]).to_csv(out/'selected_features.csv', index=False)
    summary = ('1h excursion price-feature follow-up â€” log bps\n' + scores[scores.period.eq('ALL') &
        scores.model.isin(['original', 'selected', 'training_mean', 'training_median'])].to_string(index=False) +
        '\nPositive gains mean lower error. Selected was chosen using earlier data only.\n'
        'Adaptive combination variants can have partial coverage; compare their matched-row gains, not unmatched pooled errors.\n'
        'Historical exposed evaluation; no improvement guaranteed.\n')
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    base.dump(out/'COMPLETE.json', dict(status='COMPLETE', development_origins=len(dev), replay_passed=True,
        original_replay_passed=True, predictions_sha256=base.sha(out/'predictions.csv')))
    print(summary)
    print('Saved:', out)


if __name__ == '__main__':
    main()

