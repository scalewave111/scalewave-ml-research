"""User-run 120h UP/DOWN feature-pack and HGB/MLP/GRU search."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import discover_long_features_120h as source
import market_state_features_120h as market_features
import magnitude_models_120h as models
import torch
from magnitude_models_120h import windows

b = models.b


REFERENCE=b.HERE/'experiments/long_features_120h_v1/20261113T123515172594Z'
FRAMEWORK=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
GUIDANCE=[b.HERE.parent/'PROJECT_CONTEXT.md',FRAMEWORK/'AGENTS.md',FRAMEWORK/'docs/methods/WORKFLOW.md',FRAMEWORK/'docs/methods/DATA_AND_METRICS.md']


def read_verified(path):
    done=json.loads((path/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(path/'predictions.csv')==done['predictions_sha256'],'120h reference hash')
    frame=pd.read_csv(path/'predictions.csv')
    frame.decision_time=pd.to_datetime(frame.decision_time,utc=True).dt.as_unit('ns')
    frame.issue=pd.to_datetime(frame.issue,utc=True)
    b.check(not frame.duplicated(['decision_time','target']).any(),'Duplicate reference keys')
    return frame


def load(args):
    price, y, dev, old, dictionary, files = source.load(args)
    market = pd.concat([b.read_x(args.data_dir / n) for n in
                        ['X_train.parquet', 'X_development.parquet']]).sort_index()
    market.index = market.index.as_unit('ns')
    # Sequence history must not depend on whether an earlier future label is valid.
    eligible_index = price.index
    excluded_labels = price.attrs['excluded_labels']
    contract = json.loads((source.REFERENCE/'CONTRACT.json').read_text())
    returns = source.prior.canonical.load_returns(Path(contract['arguments']['returns']))
    price, dictionary = source.build(returns, market.index, 120)
    state, definitions = market_features.build(market, price, price.index)
    cut = price.index[len(price) // 2]
    keys = price.index[price.index <= cut][-10:]
    changed = market.copy()
    changed.loc[changed.index >= cut, list(market_features.SOURCES.values())] = 1e6
    pd.testing.assert_frame_equal(state.loc[keys], market_features.build(changed, price.loc[keys], keys)[0])
    pd.testing.assert_frame_equal(state.loc[keys], market_features.build(market.loc[market.index <= cut], price.loc[keys], keys)[0])
    x = models.augment(price.join(state)).loc[eligible_index].copy()
    x.attrs['excluded_labels'] = excluded_labels
    sequence_definitions = [dict(feature=f'SEQ{lag}__{c}', family='sequence',
        formula=f'Exact historical snapshot at origin minus {lag} hours; no filling', sources=c)
        for lag in models.LAGS for c in models.pack_columns(price.join(state), 'combined')]
    definitions = pd.concat([definitions, pd.DataFrame(sequence_definitions)], ignore_index=True)
    for pack in models.PACKS:
        models.pack_columns(x, pack)
    control=read_verified(REFERENCE)
    keys=['decision_time','issue','target']
    old=old.merge(control[keys+['actual','new_features']].rename(columns={'actual':'long_actual','new_features':'previous_long'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==2*len(dev),'120h control alignment')
    np.testing.assert_allclose(old.actual,old.long_actual,rtol=0,atol=1e-10)
    files += [REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv']
    return x, y, dev, old, pd.concat([dictionary, definitions], ignore_index=True), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=b.HERE / 'experiments/magnitude_models_120h_v1')
    args = parser.parse_args()
    b.check(args.threads > 0, 'Invalid thread count')
    torch.set_num_threads(args.threads)
    x, truth, dev, old, dictionary, files = load(args)
    b.check(truth.index.equals(x.index), 'Target alignment')
    folds = {issue: windows(x.index, issue) for issue in b.ISSUES}
    print(f'Checked {len(x)} rows, {len(dev)} origins per target; Independent UP/DOWN feature-pack and HGB/MLP/GRU search.', flush=True)
    if args.check_inputs:
        return
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    dictionary.to_csv(out / 'feature_dictionary.csv', index=False)
    x.attrs['excluded_labels'].to_csv(out / 'excluded_labels.csv', index=False)
    paths = [Path(__file__), Path(models.__file__), Path(market_features.__file__),
             Path(source.__file__), Path(source.feature_module.__file__),
             Path(source.prior.__file__),
             Path(source.prior.canonical.__file__), Path(b.__file__),
             b.HERE / 'focused_market_trees_120h.py', b.HERE / 'neural_market_120h.py']
    b.dump(out / 'CONTRACT.json', dict(arguments=vars(args), targets=b.TARGETS,
        horizon_hours=120, config=models.CONFIG, loss='absolute_error',
        packs={p: models.pack_columns(x, p) for p in models.PACKS}, weights=models.WEIGHTS,
        estimator_row_caps=dict(hgb=6000, neural=models.CAP), epochs=models.EPOCHS, sequence_lags_hours=models.LAGS, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=b.sklearn.__version__, torch=torch.__version__),
        method='Separate target/quarter fits; four earlier purged quarters; selected requires .1% MAE gain, 3/4 wins or ties, no window >2% worse. Otherwise median. Best validation candidate saved separately. Training-only preprocessing and coverage admission.',
        limitations=['Repeatedly exposed historical evaluation, not independent confirmation',
                     'Candidate may fail acceptance gates; not automatically promoted',
                     'Three 38/30/62 feature packs; GRU uses four causal snapshots; source masking limitations inherited'],
        guidance_hashes={str(p): b.sha(p) for p in GUIDANCE}, input_hashes={str(p): b.sha(p) for p in files}, source_hashes={str(p): b.sha(p) for p in paths}))
    for p in paths:
        (out / p.name).write_bytes(p.read_bytes())
    frames, selections, ledger = [], [], []
    with threadpool_limits(limits=args.threads):
        for target in b.TARGETS:
            y = truth[target].to_numpy(float)
            control = old[old.target.eq(target)].set_index('decision_time')
            for i, issue in enumerate(b.ISSUES):
                end = b.ISSUES[i+1] if i+1 < len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
                te = x.index.get_indexer(dev[(dev >= issue) & (dev < end)])
                tr = np.flatnonzero(x.index + pd.Timedelta(hours=120) < issue)
                folder = out / target / f'{issue:%Y-%m}'
                folder.mkdir(parents=True)
                print(f'{target} {issue:%Y-%m}', flush=True)
                plan = models.search(x, y, folds[issue], folder)
                bundle = dict(target=target, issue=issue, plan=plan)
                forecasts = {}
                for name in ['candidate', 'selected']:
                    choice = plan[name]
                    model = (bundle['candidate'] if name == 'selected' and choice == plan['candidate']
                             else models.fit(x.iloc[tr], y[tr], choice))
                    bundle[name] = model
                    forecasts[name] = models.predict(model, x.iloc[te], choice.get('weight', 1.))
                    b.dump(folder / f'{name}_quality.json', model.get('quality', []))
                joblib.dump(bundle, folder / 'model.joblib')
                saved = joblib.load(folder / 'model.joblib')
                for name in forecasts:
                    np.testing.assert_allclose(forecasts[name], models.predict(saved[name], x.iloc[te], saved['plan'][name].get('weight', 1.)), rtol=0, atol=1e-10)
                b.dump(folder / 'MODEL_HASH.json', dict(sha256=b.sha(folder / 'model.joblib')))
                refs = control.loc[x.index[te]]
                median = np.median(y[tr])
                np.testing.assert_allclose(refs.actual, y[te], rtol=0, atol=1e-10)
                np.testing.assert_allclose(refs.training_median, median, rtol=0, atol=1e-8)
                frames.append(pd.DataFrame(dict(decision_time=x.index[te], target=target, issue=issue,
                    actual=y[te], **forecasts, training_median=median, training_mean=np.mean(y[tr]),
                    previous_price=refs.selected.to_numpy(), previous_long=refs.previous_long.to_numpy())))
                selections.append(dict(target=target, issue=str(issue), **{
                    name: {**plan[name], 'features': bundle[name]['features']} for name in forecasts}))
                for name, (train, val) in [('outer', (tr, te))] + [(f'inner_{j+1}', pair) for j, pair in enumerate(folds[issue])]:
                    ledger.append(dict(target=target, issue=issue, window=name, training_rows=len(train),
                        evaluation_rows=len(val), training_label_end_max=x.index[train].max()+pd.Timedelta(hours=120),
                        evaluation_start=x.index[val].min()))
    pred = pd.concat(frames, ignore_index=True)
    b.check(len(pred) == 2*len(dev) and not pred.duplicated(['decision_time', 'target']).any(), 'Output coverage')
    b.check(np.isfinite(pred[['selected', 'candidate']]).all().all(), 'Nonfinite forecasts')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    pred.to_csv(out / 'predictions.csv', index=False)
    rows = []
    for target, frame in pred.groupby('target'):
        for period, part in [('ALL', frame)] + list(frame.groupby('period')):
            for name in ['selected', 'candidate', 'training_median', 'training_mean', 'previous_price', 'previous_long']:
                rows.append(dict(target=target, period=period, model=name, **b.metric(part.actual.to_numpy(), part[name].to_numpy())))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'metrics.csv', index=False)
    b.dump(out / 'selections.json', selections)
    pd.DataFrame(ledger).to_csv(out / 'fit_windows.csv', index=False)
    summary = 'UP/DOWN120h MAE HGB/MLP/GRU; historically exposed evaluation.\n' + scores[scores.period.eq('ALL')].to_string(index=False)
    (out / 'SUMMARY.txt').write_text(summary)
    b.dump(out / 'COMPLETE.json', dict(status='COMPLETE', targets=b.TARGETS, development_origins_per_target=len(dev),
        replay_passed=True, predictions_sha256=b.sha(out / 'predictions.csv')))
    print(summary)
    print('Saved:', out)


if __name__ == '__main__':
    main()
