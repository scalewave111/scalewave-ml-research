r"""User-run MAE tuning of completed 15-minute excursions. No training on import.
python .\Code\tune_excursions_15m.py
python .\Code\tune_excursions_15m.py --check-inputs
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import platform
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import improve_excursion_features_15m as prior

base = prior.base
PIPELINE = base.HERE/'experiments/excursion_pipeline_15m_v1/20261113T104418654180Z'
REFERENCE = PIPELINE/'price_followup/20261113T110533531198Z'
CONFIGS = {
    'unchanged': dict(max_depth=3, max_leaf_nodes=7, min_samples_leaf=128, max_iter=80, learning_rate=.05, l2_regularization=1.),
    'shallow_strong': dict(max_depth=2, max_leaf_nodes=4, min_samples_leaf=256, max_iter=80, learning_rate=.05, l2_regularization=10.),
    'shallow_flexible': dict(max_depth=2, max_leaf_nodes=4, min_samples_leaf=64, max_iter=160, learning_rate=.05, l2_regularization=5.),
    'smaller_leaf': dict(max_depth=3, max_leaf_nodes=7, min_samples_leaf=64, max_iter=80, learning_rate=.05, l2_regularization=1.),
    'more_iterations': dict(max_depth=3, max_leaf_nodes=7, min_samples_leaf=128, max_iter=160, learning_rate=.05, l2_regularization=1.),
    'slow_regularized': dict(max_depth=3, max_leaf_nodes=7, min_samples_leaf=128, max_iter=240, learning_rate=.025, l2_regularization=10.),
    'deeper_regularized': dict(max_depth=4, max_leaf_nodes=12, min_samples_leaf=128, max_iter=160, learning_rate=.05, l2_regularization=10.),
    'deeper_flexible': dict(max_depth=4, max_leaf_nodes=12, min_samples_leaf=64, max_iter=160, learning_rate=.025, l2_regularization=5.),
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def short_features(r, origins):
    """Same eight past-price statistics as prior, now at 15 and 30 minutes."""
    values = {}
    level = r.fillna(0).cumsum().shift(1)
    for minutes in [15, 30]:
        n = minutes//5
        opts = dict(window=f'{minutes}min', closed='left', min_periods=n)
        valid = r.rolling(**opts).count().eq(n)
        hi = level.rolling(n+1, min_periods=n+1).max()
        lo = level.rolling(n+1, min_periods=n+1).min()
        stats = dict(rv=np.sqrt(r.pow(2).rolling(**opts).sum()),
            up_rv=np.sqrt(r.clip(lower=0).pow(2).rolling(**opts).sum()),
            down_rv=np.sqrt(r.clip(upper=0).pow(2).rolling(**opts).sum()),
            net=r.rolling(**opts).sum(), past_range=hi-lo,
            distance_from_high=hi-level, distance_from_low=level-lo,
            max_abs_return=r.abs().rolling(**opts).max())
        for name, v in stats.items():
            values[f'SHORTPRICE__{name}__{minutes}m'] = v.where(valid)
    return pd.DataFrame(values).reindex(origins).replace([np.inf, -np.inf], np.nan)


def check_causal(r, origins):
    cut = origins[len(origins)//2]
    keys = origins[origins <= cut][-10:]
    pd.testing.assert_frame_equal(short_features(r, keys), short_features(r.loc[r.index <= cut], keys))
    changed = r.copy()
    changed.loc[changed.index >= cut] = 1e6
    keys = pd.DatetimeIndex([cut])
    pd.testing.assert_frame_equal(short_features(r, keys), short_features(changed, keys))


def fit(a, y, config):
    if a.shape[1] == 0:
        return {'constant': float(np.median(y))}
    prep = base.transform_fit(a)
    model = HistGradientBoostingRegressor(loss='absolute_error', early_stopping=False,
        random_state=48, **CONFIGS[config])
    model.fit(base.transform(a, prep), y)
    return {'prep': prep, 'model': model}


def choose(y, candidates, control, median, weights, min_gain):
    """Inputs are earlier validation arrays only; outer labels never enter."""
    errors = lambda pp: np.array([np.mean(np.abs(a-b)) for a, b in zip(y, pp)])
    old, med = errors(control), errors(median)
    records, eligible = [], []
    for name, predictions in candidates.items():
        for weight in [.5, 1.]:
            mixed = [(1-weight)*a+weight*b for a, b in zip(control, predictions)]
            err = errors(mixed)
            ok = prior.gate(err, old, weights, min_gain) and prior.gate(err, med, weights, min_gain)
            rec = dict(candidate=name, weight=weight, eligible=bool(ok),
                pooled_mae=float(np.average(err, weights=weights)),
                **{f'window_{i+1}_mae': float(e) for i, e in enumerate(err)})
            records.append(rec)
            if ok:
                eligible.append(rec)
    chosen = min(eligible, key=lambda r: (r['pooled_mae'], r['candidate'], r['weight'])) if eligible else dict(candidate='current', weight=0., pooled_mae=float(np.average(old, weights=weights)))
    return dict(chosen, control_window_mae=old.tolist(), median_window_mae=med.tolist()), records


def inputs(args):
    contract_path = args.followup_run/'CONTRACT.json'
    contract = read_json(contract_path)
    base.check(contract['horizon_minutes'] == 15, 'Reference horizon must be 15 minutes')
    args.reference_run = Path(contract['arguments']['reference_run'])
    x, y, dev, _, _, groups, files = prior.inputs(args)
    for module in [Path(base.__file__), Path(prior.__file__), Path(prior.canonical.__file__)]:
        base.check(contract['source_hashes'][str(module)] == base.sha(module), f'Reference helper changed: {module}')
    done = read_json(args.followup_run/'COMPLETE.json')
    path = args.followup_run/'predictions.csv'
    base.check(done['status'] == 'COMPLETE' and done['predictions_sha256'] == base.sha(path), 'Follow-up prediction hash mismatch')
    old = pd.read_csv(path)
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    base.check(len(old) == 2*len(dev) and not old.duplicated(['decision_time','target']).any(), 'Reference coverage mismatch')
    for target in base.TARGETS:
        f = old[old.target.eq(target)].set_index('decision_time').sort_index()
        base.check(f.index.equals(dev), 'Reference timestamp mismatch')
        np.testing.assert_allclose(f.actual, y.loc[dev,target], atol=1e-10, rtol=0)
        expected_issues = base.ISSUES[base.ISSUES.searchsorted(f.index, side='right')-1]
        base.check((f.issue.to_numpy() == expected_issues.to_numpy()).all(), 'Reference issuance mismatch')
    r = prior.canonical.load_returns(args.returns)
    check_causal(r, x.index)
    extra = short_features(r, x.index)
    x = x.join(extra)
    cards = {}
    files += [contract_path, path, args.followup_run/'COMPLETE.json', Path(__file__)]
    for issue in base.ISSUES:
        for target in base.TARGETS:
            directory = args.followup_run/f'{issue:%Y-%m}_{target}'
            card_path = directory/'selection.json'
            card = read_json(card_path)
            spec = card['variants'][card['selected_variant']]
            base.check(spec['model'] == 'tree', 'This tuning run expects prior selected boosting models')
            ref = old[old.issue.eq(issue) & old.target.eq(target)].set_index('decision_time').sort_index()
            for tag, col in [(card['selected_variant'], 'selected'), ('all_price_plus_original','all_price_plus_original')]:
                saved = card['variants'][tag]
                modelpath = directory/saved['artifact']
                base.check(base.sha(modelpath) == saved['artifact_sha256'], 'Saved reference model hash mismatch')
                bundle = joblib.load(modelpath)
                base.check(bundle['features'] == saved['features'], 'Saved feature order mismatch')
                pred = base.predict(bundle['model'], x.loc[ref.index,saved['features']].to_numpy(float))
                np.testing.assert_allclose(pred, ref[col], rtol=0, atol=1e-7)
                files.append(modelpath)
            files.append(card_path)
            cards[(issue,target)] = card
    print(f'Checked {len(x)} rows, {len(dev)} origins, 16 causal short-price inputs; saved reference models replayed.', flush=True)
    return x, y, dev, old, cards, list(extra), list(dict.fromkeys(files))


def feature_packs(x, folds, card, short_cols):
    selected = card['variants'][card['selected_variant']]['features']
    wide = card['variants']['all_price_plus_original']['features']
    allowed = [c for c in short_cols if all(x.iloc[tr][c].notna().mean() >= .8 and
        x.iloc[tr][c].notna().sum() >= 1000 and x.iloc[tr][c].nunique() > 1 for tr, _ in folds)]
    return dict(current_features=selected, all_price=wide, all_price_short=wide+allowed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=base.DATA)
    parser.add_argument('--y-dir', type=Path, default=base.CANONICAL/'y/horizon_minutes=15/symbol=BTCUSDT')
    parser.add_argument('--returns', type=Path, default=prior.WEEK1/'Dataset/canonical_returns_5m.parquet')
    parser.add_argument('--followup-run', type=Path, default=REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/excursion_tuning_15m_v1')
    parser.add_argument('--resume', type=Path, help='Existing tuning run folder; completed target/quarter blocks are hash-checked and reused')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--min-gain', type=float, default=.001)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    base.check(args.threads > 0 and 0 <= args.min_gain < 1, 'Invalid options')
    with threadpool_limits(limits=args.threads):
        x, y, dev, old, cards, short_cols, files = inputs(args)
    if args.check_inputs:
        print('Input checks and saved-model inference only. No fitting.')
        return
    identities = {str(p):base.sha(p) for p in files}
    settings = dict(configs=CONFIGS, input_hashes=identities, min_gain=args.min_gain,
        threads=args.threads, numpy=np.__version__, pandas=pd.__version__, sklearn=base.sklearn.__version__)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    out = args.resume or args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        base.check(read_json(out/'CONTRACT.json')['signature'] == signature, 'Resume inputs/code/settings changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        guidance = [base.HERE.parent/'PROJECT_CONTEXT.md',
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
        base.dump(out/'CONTRACT.json', dict(signature=signature, arguments=vars(args), **settings,
            horizon_minutes=15, targets=base.TARGETS, python=platform.python_version(),
            instruction_hashes={str(p):base.sha(p) for p in guidance},
            method='Three feature packs crossed independently with eight fixed MAE boosting configurations. Same three earlier quarterly validation windows; strict 15m maturity. Candidate weights 0.5/1 relative to current selected forecast; current retained unless >=0.1% MAE gain (default) and 2/3 window wins versus current AND median.',
            limitations=['Exposed historical evaluation; no fresh confirmation or guaranteed error reduction',
                'Reused validation windows and prior selected feature sets; adaptive search, not fully nested selection',
                'All-price pack included following exposed results; post-hoc research design disclosed',
                'Inherited X availability limitations and original 48h boundary gap; 30-minute origins',
                'No automatic early stopping; fixed iteration counts selected by earlier validation',
                'Resume granularity is one target/quarter; an unfinished block is recomputed']))
        for p in [Path(__file__), Path(prior.__file__), Path(base.__file__), Path(prior.canonical.__file__)]:
            shutil.copyfile(p, out/p.name)
    print('Output:', out, flush=True)
    frames, selections = [], []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            folds = base.windows(x.index, issue)
            tr = np.flatnonzero(x.index + pd.Timedelta(minutes=15) < issue)
            for target in base.TARGETS:
                folder = out/f'{issue:%Y-%m}_{target}'
                folder.mkdir(exist_ok=True)
                receipt = folder/'COMPLETE.json'
                if receipt.exists():
                    for filename, digest in read_json(receipt)['hashes'].items():
                        base.check(base.sha(folder/filename) == digest, 'Checkpoint hash mismatch')
                    frame = pd.read_csv(folder/'predictions.csv')
                    frame['decision_time'] = pd.to_datetime(frame.decision_time, utc=True)
                    frames.append(frame)
                    selections.append(read_json(folder/'selection.json'))
                    print('Reused completed:', folder.name, flush=True)
                    continue
                print('Tuning:', folder.name, flush=True)
                card = cards[(issue,target)]
                packs = feature_packs(x, folds, card, short_cols)
                specs = {f'{pack}__{config}':dict(features=cols, config=config) for pack, cols in packs.items() for config in CONFIGS}
                baseline_name = 'current_features__unchanged'
                truth = y[target].to_numpy()
                weights = [len(va) for _,va in folds]
                vy = [truth[va] for _,va in folds]
                median = [np.full(len(va), np.median(truth[itr])) for itr,va in folds]
                cache, inner = {}, {}
                for name, spec in specs.items():
                    key = (tuple(spec['features']), spec['config'])
                    if key not in cache:
                        a = x[spec['features']].to_numpy(float)
                        cache[key] = [base.predict(fit(a[itr], truth[itr], spec['config']), a[va]) for itr,va in folds]
                    inner[name] = cache[key]
                    print('  earlier fits:', name, flush=True)
                control = inner[baseline_name]
                old_errors = [np.abs(a-b).mean() for a,b in zip(vy, control)]
                np.testing.assert_allclose(old_errors, card['variants'][card['selected_variant']]['validation_window_mae'], rtol=0, atol=1e-7,
                    err_msg='Current baseline earlier-validation reproduction failed')
                chosen, trials = choose(vy, inner, control, median, weights, args.min_gain)
                pd.DataFrame(trials).to_csv(folder/'validation_trials.csv', index=False)
                ref = old[old.issue.eq(issue) & old.target.eq(target)].set_index('decision_time').sort_index()
                te = x.index.get_indexer(ref.index)
                f = pd.DataFrame(dict(decision_time=ref.index, issue=issue, target=target, actual=truth[te],
                    current=ref.selected.to_numpy(), previous_all_price=ref.all_price_plus_original.to_numpy(),
                    training_median=float(np.median(truth[tr])), training_mean=float(truth[tr].mean())))
                cache, artifacts = {}, {}
                for number, (name, spec) in enumerate(specs.items()):
                    key = (tuple(spec['features']), spec['config'])
                    a = x[spec['features']].to_numpy(float)
                    if key not in cache:
                        model = fit(a[tr], truth[tr], spec['config'])
                        modelpath = folder/f'model_{number:02d}.joblib'
                        joblib.dump(dict(model=model, **spec), modelpath)
                        replay = joblib.load(modelpath)
                        pred = base.predict(model, a[te])
                        np.testing.assert_allclose(pred, base.predict(replay['model'], a[te]), rtol=0, atol=1e-10)
                        cache[key] = (pred, modelpath.name)
                        artifacts[modelpath.name] = base.sha(modelpath)
                    pred, artifact = cache[key]
                    spec['artifact'] = artifact
                    f[name] = pred
                np.testing.assert_allclose(f[baseline_name], f.current, rtol=0, atol=1e-7)
                np.testing.assert_allclose(f['all_price__unchanged'], f.previous_all_price, rtol=0, atol=1e-7)
                name, weight = chosen['candidate'], chosen['weight']
                f['selected'] = (1-weight)*f.current + weight*f[name]
                chosen.update(issue=str(issue), target=target, packs=packs, specs=specs,
                    current_artifact=specs[baseline_name]['artifact'],
                    candidate_artifact=specs[name]['artifact'] if weight else specs[baseline_name]['artifact'])
                base.dump(folder/'selection.json', chosen)
                f.to_csv(folder/'predictions.csv', index=False)
                ledger = [dict(window='outer', latest_training_label_end=str((x.index[tr]+pd.Timedelta(minutes=15)).max()), evaluation_start=str(ref.index.min()))]
                ledger += [dict(window=i+1, latest_training_label_end=str((x.index[itr]+pd.Timedelta(minutes=15)).max()), validation_start=str(x.index[va].min()), validation_label_end_max=str((x.index[va]+pd.Timedelta(minutes=15)).max())) for i,(itr,va) in enumerate(folds)]
                base.dump(folder/'fit_windows.json', ledger)
                for filename in ['predictions.csv','validation_trials.csv','selection.json','fit_windows.json']:
                    artifacts[filename] = base.sha(folder/filename)
                base.dump(receipt, dict(status='COMPLETE', hashes=artifacts))
                frames.append(f)
                selections.append(chosen)
    f = pd.concat(frames, ignore_index=True)
    base.check(len(f) == 2*len(dev) and not f.duplicated(['decision_time','target']).any(), 'Output coverage mismatch')
    f['period'] = f.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    f.loc[f.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    f.to_csv(out/'predictions.csv', index=False)
    rows = []
    models = [c for c in f if c not in ['decision_time','issue','target','actual','period']]
    for target,g in f.groupby('target'):
        for period,sub in [('ALL',g)]+list(g.groupby('period')):
            baseline = base.metric(sub.actual.to_numpy(), sub.current.to_numpy())
            for name in models:
                measured = base.metric(sub.actual.to_numpy(), sub[name].to_numpy())
                rows.append(dict(target=target, period=period, model=name, **measured,
                    mae_gain_vs_current=baseline['mae']-measured['mae'], rmse_gain_vs_current=baseline['rmse']-measured['rmse']))
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'metrics.csv', index=False)
    base.dump(out/'selections.json', selections)
    summary = '15-minute excursion tuning - log bps\n'+scores[scores.period.eq('ALL') & scores.model.isin(['current','previous_all_price','selected','training_median'])].to_string(index=False)
    summary += '\nSelection uses earlier data only. Other variants are diagnostics, not automatic promotions. Historical exposed evaluation.\n'
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    base.dump(out/'COMPLETE.json', dict(status='COMPLETE', development_origins=len(dev), prediction_rows=len(f),
        replay_passed=True, current_reproduced=True, predictions_sha256=base.sha(out/'predictions.csv')))
    print(summary, '\nSaved:', out)


if __name__ == '__main__':
    main()
