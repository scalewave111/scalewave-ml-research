r"""User-run 120h canonical excursion feature search. No fits on import.

python .\Code\discover_excursion_features_120h.py
python .\Code\discover_excursion_features_120h.py --check-inputs
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data'
CANONICAL = Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1')
TARGETS = ['up_magnitude_log_bps', 'down_magnitude_log_bps']
ISSUES = pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'], utc=True)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def read_x(path):
    with duckdb.connect() as con:
        f = con.execute('SELECT * FROM read_parquet(?)', [str(path)]).fetchdf()
    key = next((k for k in ['decision_time', '__index_level_0__'] if k in f), None)
    check(key is not None, 'No timestamp column in X')
    f = f.set_index(key)
    f.index = pd.to_datetime(f.index, utc=True)
    check(f.index.is_unique and not f.index.hasnans and f.columns.is_unique, 'Invalid X keys')
    return f.sort_index()


def load(args):
    paths = [args.data_dir / n for n in ['X_train.parquet', 'X_development.parquet']]
    train, dev = [read_x(p) for p in paths]
    check(len(train) == 29615 and len(dev) == 18481, 'Original row counts changed')
    check(train.index.min() == pd.Timestamp('2022-04-22 00:30', tz='UTC') and
          train.index.max() == pd.Timestamp('2023-12-29 23:30', tz='UTC'), 'Initial training dates changed')
    check(dev.index.min() == ISSUES[0] and dev.index.max() < pd.Timestamp('2025-01-23', tz='UTC'), 'Development dates changed')
    check(train.columns.equals(dev.columns), 'Feature schemas differ')
    cols = [c for c in train if c.startswith('F4Z__') and c.endswith('__value')]
    check(len(cols) == 522, 'Expected 522 original masked F4Z values')
    x = pd.concat([train[cols], dev[cols]]).sort_index().replace([np.inf, -np.inf], np.nan)
    check(x.index.is_unique, 'Overlapping train/development timestamps')
    files = sorted(args.y_dir.rglob('*.parquet'))
    check(bool(files), 'No canonical Y partitions found')
    with duckdb.connect() as con:
        con.register('keys', pd.DataFrame({'decision_time': x.index}))
        y = con.execute('''SELECT y.* FROM read_parquet(?, hive_partitioning=false) y
            INNER JOIN keys k ON y.decision_time=k.decision_time''', [[str(p) for p in files]]).fetchdf()
    y['decision_time'] = pd.to_datetime(y.decision_time, utc=True)
    check(y.decision_time.is_unique, 'Duplicate canonical Y timestamps')
    y = y.set_index('decision_time').reindex(x.index)
    valid = (y.label_status.eq('VALID') & y.canonical_symbol.eq('BTCUSDT') &
             pd.to_numeric(y.horizon_minutes).eq(7200))
    valid &= np.isfinite(y[TARGETS]).all(axis=1) & y[TARGETS].ge(0).all(axis=1)
    check(y.canonical_symbol.eq('BTCUSDT').all() and pd.to_numeric(y.horizon_minutes).eq(7200).all(),
          'Missing canonical keys or incorrect symbol/horizon')
    check(y.label_status.isin(['VALID', 'INVALID_PRICE_PATH', 'OUTSIDE_RELEASE']).all(), 'Unexpected label status')
    check(valid.equals(y.label_status.eq('VALID')), 'Malformed targets marked VALID')
    excluded = y.loc[~valid, ['label_status']].reset_index()
    print('Excluded canonical labels:', excluded.label_status.value_counts().to_dict(), flush=True)
    x = x.loc[valid].copy()
    x.attrs['excluded_labels'] = excluded
    development = dev.index[dev.index.isin(x.index)]
    return x, y.loc[valid, TARGETS].astype(float), development, paths + files


def windows(index, issue):
    result = []
    for offset in [9, 6, 3]:
        start = issue - pd.DateOffset(months=offset)
        end = start + pd.DateOffset(months=3)
        tr = np.flatnonzero(index + pd.Timedelta(hours=120) < start)
        va = np.flatnonzero((index >= start) & (index < end) &
                            (index + pd.Timedelta(hours=120) < end))
        check(len(tr) >= 1000 and len(va) >= 100, 'Insufficient earlier validation data')
        check((index[tr] + pd.Timedelta(hours=120) < index[va].min()).all(), 'Purging failed')
        result.append((tr, va))
    return result


def transform_fit(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(a, axis=0)
    med = np.nan_to_num(med)
    b = np.column_stack([np.where(np.isnan(a), med, a), np.isnan(a).astype(float)])
    center, scale = b.mean(axis=0), b.std(axis=0)
    scale[scale < 1e-8] = 1
    return (med, center, scale)


def transform(a, prep):
    med, center, scale = prep
    return (np.column_stack([np.where(np.isnan(a), med, a), np.isnan(a).astype(float)]) - center) / scale


def fit(a, y, kind):
    if a.shape[1] == 0:
        return {'constant': float(np.median(y))}
    prep = transform_fit(a)
    if kind == 'screen':
        model = DecisionTreeRegressor(criterion='absolute_error', max_depth=2,
                                      min_samples_leaf=128, random_state=48)
    elif kind == 'ridge':
        model = Ridge(alpha=.01 * len(y))
    else:
        model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=80,
            learning_rate=.05, max_leaf_nodes=7, max_depth=3, min_samples_leaf=128,
            l2_regularization=1., early_stopping=False, random_state=48)
    model.fit(transform(a, prep), y)
    return {'prep': prep, 'model': model}


def predict(model, a):
    if 'constant' in model:
        return np.full(len(a), model['constant'])
    return np.maximum(0., model['model'].predict(transform(a, model['prep'])))


def metric(y, p):
    ic = pd.Series(y).corr(pd.Series(p), method='spearman') if np.std(p) > 1e-12 and np.std(y) > 1e-12 else np.nan
    return dict(rows=len(y), mae=float(np.mean(np.abs(y-p))), rmse=float(np.sqrt(np.mean((y-p)**2))),
                ic=float(ic) if np.isfinite(ic) else None)


def search(x, y, folds, args, folder):
    names = list(x.columns)
    a = x.to_numpy(float)
    eligible, quality = [], []
    for j, name in enumerate(names):
        counts = [int(np.isfinite(a[tr, j]).sum()) for tr, _ in folds]
        coverage = [float(np.isfinite(a[tr, j]).mean()) for tr, _ in folds]
        varies = all(pd.Series(a[tr, j]).nunique() > 1 for tr, _ in folds)
        ok = min(counts) >= 1000 and min(coverage) >= .8 and varies
        quality.append(dict(feature=name, min_count=min(counts), min_coverage=min(coverage), eligible=ok))
        if ok:
            eligible.append(j)
    pd.DataFrame(quality).to_csv(folder/'quality.csv', index=False)
    cache, trials = {}, []

    def evaluate(cols, kind, stage):
        key = (tuple(cols), kind)
        if key not in cache:
            errors = []
            for tr, va in folds:
                # Screening budget only: chronological evenly spaced training subset.
                use = tr[np.unique(np.linspace(0, len(tr)-1, min(args.screen_rows, len(tr)), dtype=int))] if kind == 'screen' else tr
                model = fit(a[np.ix_(use, cols)], y[use], kind)
                errors.append(np.abs(y[va] - predict(model, a[np.ix_(va, cols)])))
            cache[key] = np.array([np.mean(e) for e in errors])
            trials.append(dict(stage=stage, model=kind, features=json.dumps([names[j] for j in cols]),
                pooled_mae=float(np.average(cache[key], weights=[len(va) for _, va in folds])),
                **{f'window_{i+1}_mae': float(v) for i, v in enumerate(cache[key])}))
        return cache[key]

    weights = [len(va) for _, va in folds]
    score = lambda v: float(np.average(v, weights=weights))
    improves = lambda new, old: score(new) < score(old)*(1-args.min_gain) and int((new < old).sum()) >= 2
    baseline = evaluate([], 'tree', 'baseline')
    ranked = sorted(eligible, key=lambda j: (score(evaluate([j], 'screen', 'screen')), names[j]))
    # Redundancy decisions use only the earliest training window, never outer data.
    short = []
    for j in ranked:
        redundant = False
        for k in short:
            pair = x.iloc[folds[0][0], [j, k]].dropna()
            if len(pair) >= 1000 and abs(pair.iloc[:, 0].corr(pair.iloc[:, 1], method='spearman')) >= .9:
                redundant = True
                break
        if not redundant:
            short.append(j)
        if len(short) >= args.shortlist:
            break
    dump(folder/'shortlist.json', [names[j] for j in short])
    winners = []
    removals = []
    for kind in ['ridge', 'tree']:
        selected, current = [], baseline
        for step in range(args.max_features):
            candidates = [(j, evaluate(selected+[j], kind, 'forward')) for j in short if j not in selected]
            candidates = [(j, v) for j, v in candidates if improves(v, current)]
            if not candidates:
                break
            j, current = min(candidates, key=lambda item: (score(item[1]), names[item[0]]))
            selected.append(j)
            print(f'  {kind}: added {names[j]} (validation MAE {score(current):.4f})', flush=True)
        # Refit each removal, then remove a feature only if earlier validation is no worse.
        changed = True
        while changed and selected:
            changed = False
            tested = []
            for j in selected:
                cols = [k for k in selected if k != j]
                val = evaluate(cols, kind, 'removal')
                removals.append(dict(model=kind, feature=names[j], feature_set=json.dumps([names[k] for k in selected]),
                                     removal_mae_change=score(val)-score(current)))
                if score(val) <= score(current) and int((val <= current).sum()) >= 2:
                    tested.append((j, val))
            if tested:
                j, current = min(tested, key=lambda item: (score(item[1]), names[item[0]]))
                selected.remove(j)
                changed = True
        winners.append((kind, selected, current))
    candidates = [v for v in winners if v[1] and improves(v[2], baseline)]
    chosen = min(candidates, key=lambda v: score(v[2])) if candidates else ('median', [], baseline)
    pd.DataFrame(trials).to_csv(folder/'validation_trials.csv', index=False)
    pd.DataFrame(removals, columns=['model', 'feature', 'feature_set', 'removal_mae_change']).to_csv(folder/'removal_checks.csv', index=False)
    result = dict(model=chosen[0], features=[names[j] for j in chosen[1]], validation_mae=score(chosen[2]),
                  baseline_mae=score(baseline), validation_window_mae=chosen[2].tolist())
    dump(folder/'selection.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--y-dir', type=Path, default=CANONICAL/'y/horizon_minutes=7200/symbol=BTCUSDT')
    p.add_argument('--output-dir', type=Path, default=HERE/'experiments/excursion_feature_search_120h_v1')
    p.add_argument('--shortlist', type=int, default=16)
    p.add_argument('--max-features', type=int, default=6)
    p.add_argument('--screen-rows', type=int, default=4096)
    p.add_argument('--min-gain', type=float, default=.001)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args()
    check(args.shortlist >= args.max_features > 0 and args.screen_rows >= 1000 and
          args.threads > 0 and 0 <= args.min_gain < 1, 'Invalid search settings')
    print('Loading original X keys and exact canonical 120-hour labels...', flush=True)
    x, targets, dev, inputs = load(args)
    folds_by_issue = {issue: windows(x.index, issue) for issue in ISSUES}
    print(f'Inputs verified: {len(x)} rows; {len(dev)} development origins; {len(x.columns)} candidates.', flush=True)
    for issue in ISSUES:
        mature = x.index[x.index + pd.Timedelta(hours=120) < issue]
        print(f'{issue:%Y-%m}: {len(mature)} mature training rows; latest label end {mature.max()+pd.Timedelta(hours=120)}', flush=True)
    if args.check_inputs:
        print('Input checks only. No model training or feature selection performed.')
        return
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv', index=False)
    dump(out/'CONTRACT.json', dict(arguments=vars(args), targets=TARGETS, horizon_minutes=7200,
        population='Original 29615 initial and 18481 development input origins; exclude canonical INVALID_PRICE_PATH/OUTSIDE_RELEASE labels with audit; retain all valid rows including zeros; no direction filtering',
        excluded_label_counts=x.attrs['excluded_labels'].label_status.value_counts().to_dict(),
        eligible_development_origins=len(dev),
        refits=[str(i) for i in ISSUES], january_2025='October model retained; separately scored',
        selection='Three earlier quarterly windows; strict 120h maturity at every boundary; greedy MAE gain >= min_gain and wins in >=2/3 windows; median fallback',
        search='522 masked F4Z only; no mandatory features; 80% training coverage, 1000 readings, variation in every inner fit; depth2 median-tree screen; earliest-training redundancy; Ridge and small absolute-error boosting forward/removal search',
        limitations=['Previously exposed history, not independent confirmation', 'Adaptive reuse of inner validation is not nested confirmation',
                     'Original X availability masking inherited, not newly certified; original PROVISIONAL/funding timing limits remain',
                     'Half-hour sampled population; original 48h boundary gap is preserved',
                     'No newly derived price features or old eight-feature comparator; this runner searches the 522 supplied F4Z values',
                     'Ridge fits squared error but is selected by MAE; tree fits absolute error; predictions clipped at zero',
                     'Bounded greedy search can miss interactions; no significance or guaranteed feature-count claim'],
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        source_sha256=sha(Path(__file__)), inputs={str(v): sha(v) for v in inputs},
        guidance_hashes={str(v): sha(v) for v in [HERE.parent/'PROJECT_CONTEXT.md',
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),
            Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md'),
            CANONICAL/'team_tools/TEAM_DATASET_SPEC.md',
            args.data_dir/'feature_dictionary_and_admission.csv'] if v.exists()}))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    predictions, selections, ledger = [], [], []
    with threadpool_limits(limits=args.threads):
        for i, issue in enumerate(ISSUES):
            end = ISSUES[i+1] if i+1 < len(ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te = x.index.get_indexer(dev[(dev >= issue) & (dev < end)])
            tr = np.flatnonzero(x.index + pd.Timedelta(hours=120) < issue)
            for target in TARGETS:
                print(f'{issue:%Y-%m} {target}: searching earlier windows...', flush=True)
                folder = out/f'{issue:%Y-%m}_{target}'
                folder.mkdir()
                y = targets[target].to_numpy()
                selected = search(x, y, folds_by_issue[issue], args, folder)
                cols = selected['features']
                a = x[cols].to_numpy(float)
                model = fit(a[tr], y[tr], selected['model'])
                pred = predict(model, a[te])
                joblib.dump(dict(model=model, features=cols, target=target, issue=issue), folder/'model.joblib')
                replay = joblib.load(folder/'model.joblib')
                np.testing.assert_allclose(pred, predict(replay['model'], a[te]), rtol=0, atol=1e-10)
                dump(folder/'MODEL_HASH.json', {'sha256': sha(folder/'model.joblib')})
                f = pd.DataFrame(dict(decision_time=x.index[te], issue=issue, target=target,
                    actual=y[te], prediction=pred, training_mean=np.mean(y[tr]), training_median=np.median(y[tr])))
                f.to_csv(folder/'predictions.csv', index=False)
                predictions.append(f)
                selections.append(dict(issue=str(issue), target=target, **selected))
                ledger.append(dict(issue=issue, target=target, training_rows=len(tr), evaluation_rows=len(te),
                    latest_training_origin=x.index[tr].max(), latest_training_label_end=(x.index[tr]+pd.Timedelta(hours=120)).max()))
                for n, (itr, iva) in enumerate(folds_by_issue[issue]):
                    ledger.append(dict(issue=issue, target=target, inner_window=n+1, training_rows=len(itr),
                        evaluation_rows=len(iva), latest_training_origin=x.index[itr].max(),
                        latest_training_label_end=(x.index[itr]+pd.Timedelta(hours=120)).max(),
                        validation_start=x.index[iva].min(), validation_label_end_max=(x.index[iva]+pd.Timedelta(hours=120)).max()))
    pred = pd.concat(predictions, ignore_index=True)
    check(len(pred) == 2*len(dev) and not pred.duplicated(['decision_time', 'target']).any(), 'Prediction coverage failed')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    pred.to_csv(out/'predictions.csv', index=False)
    metrics = []
    for target, g in pred.groupby('target'):
        for period, f in [('ALL', g)] + list(g.groupby('period')):
            for col in ['prediction', 'training_mean', 'training_median']:
                metrics.append(dict(target=target, period=period, model=col, **metric(f.actual.to_numpy(), f[col].to_numpy())))
    scores = pd.DataFrame(metrics)
    scores.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv', index=False)
    dump(out/'selections.json', selections)
    rows = [dict(target=s['target'], issue=s['issue'], feature=c) for s in selections for c in s['features']]
    selected_table = pd.DataFrame(rows, columns=['target', 'issue', 'feature'])
    selected_table.to_csv(out/'selected_features_by_cycle.csv', index=False)
    stability = selected_table.groupby(['target', 'feature']).size().rename('selected_cycles').reset_index()
    stability['total_cycles'] = len(ISSUES)
    dictionary = args.data_dir/'feature_dictionary_and_admission.csv'
    if dictionary.exists():
        meanings = pd.read_csv(dictionary)[['feature', 'family', 'measurement', 'lookback']]
        stability = stability.merge(meanings, on='feature', how='left', validate='many_to_one')
    stability.to_csv(out/'feature_stability.csv', index=False)
    summary = ('120-HOUR UP/DOWN EXCURSION FEATURE SEARCH â€” log bps\n\n' +
        scores[scores.period.eq('ALL')].to_string(index=False) +
        '\n\nSelected features by cycle are in selections.json. Empty lists mean median fallback.\n'
        'Selection frequency is descriptive; never select a new pooled feature set using outer scores.\n'
        'January 2025 uses the October model. Historical exposed evaluation, not fresh confirmation.\n')
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    dump(out/'COMPLETE.json', dict(status='COMPLETE', prediction_rows=len(pred), development_origins=len(dev),
        replay_passed=True, predictions_sha256=sha(out/'predictions.csv')))
    print(summary)
    print('Saved:', out)


if __name__ == '__main__':
    main()

