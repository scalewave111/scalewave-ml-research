"""User-run fast DOWN72h MLP and GRU architecture search."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import discover_long_features_72h as source
import market_state_features_72h as market_features
import neural_market_down_72h as models
import torch
from improve_best_models_72h import read_verified, REFERENCE
from fit_focused_market_down_72h import windows

b = models.b


def load(args):
    price, y, dev, old, dictionary, files = source.load(args)
    market = pd.concat([b.read_x(args.data_dir / n) for n in
                        ['X_train.parquet', 'X_development.parquet']]).sort_index()
    market.index = market.index.as_unit('ns')
    state, definitions = market_features.build(market, price, price.index)
    cut = price.index[len(price) // 2]
    keys = price.index[price.index <= cut][-10:]
    changed = market.copy()
    changed.loc[changed.index >= cut, list(market_features.SOURCES.values())] = 1e6
    pd.testing.assert_frame_equal(state.loc[keys], market_features.build(changed, price.loc[keys], keys)[0])
    pd.testing.assert_frame_equal(state.loc[keys], market_features.build(market.loc[market.index <= cut], price.loc[keys], keys)[0])
    x = models.augment(price.join(state))
    x.attrs['excluded_labels'] = price.attrs['excluded_labels']
    for pack in models.PACKS:
        models.pack_columns(x, pack)
    control=read_verified(REFERENCE)
    control=control[control.target.eq('down_magnitude_log_bps')]
    keys=['decision_time','issue','target']
    old=old[old.target.eq('down_magnitude_log_bps')].merge(control[keys+['actual','candidate']].rename(columns={'actual':'best_actual','candidate':'previous_best'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==len(dev),'Best HGB alignment')
    np.testing.assert_allclose(old.actual,old.best_actual,rtol=0,atol=1e-10)
    files += [REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv']
    return x, y, dev, old, pd.concat([dictionary, definitions], ignore_index=True), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=b.HERE / 'experiments/neural_market_down_72h_v1')
    args = parser.parse_args()
    b.check(args.threads > 0, 'Invalid thread count')
    torch.set_num_threads(args.threads)
    x, truth, dev, old, dictionary, files = load(args)
    b.check(truth.index.equals(x.index), 'Target alignment')
    folds = {issue: windows(x.index, issue) for issue in b.ISSUES}
    print(f'Checked {len(x)} rows, {len(dev)} origins per target; DOWN MLP/GRU search.', flush=True)
    if args.check_inputs:
        return
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    dictionary.to_csv(out / 'feature_dictionary.csv', index=False)
    x.attrs['excluded_labels'].to_csv(out / 'excluded_labels.csv', index=False)
    paths = [Path(__file__), Path(models.__file__), Path(market_features.__file__),
             Path(source.__file__), Path(source.feature_module.__file__),
             Path(source.feature_module.shared.__file__), Path(source.prior.__file__),
             Path(source.prior.canonical.__file__), Path(b.__file__),
             b.HERE / 'fit_focused_market_down_72h.py', b.HERE / 'focused_market_trees_72h.py', b.HERE / 'improve_best_models_72h.py']
    b.dump(out / 'CONTRACT.json', dict(arguments=vars(args), targets=['down_magnitude_log_bps'],
        horizon_hours=72, config=models.CONFIG, loss='absolute_error',
        packs={p: models.pack_columns(x, p) for p in models.PACKS}, weights=models.WEIGHTS,
        estimator_row_cap=models.CAP, epochs=models.EPOCHS, sequence_lags_hours=models.LAGS, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=b.sklearn.__version__, torch=torch.__version__),
        method='Separate target/quarter fits; four earlier purged quarters; selected requires .1% MAE gain, 3/4 wins or ties, no window >2% worse. Otherwise median. Best validation candidate saved separately. Training-only preprocessing and coverage admission.',
        limitations=['Repeatedly exposed historical evaluation, not independent confirmation',
                     'Candidate may fail acceptance gates; not automatically promoted',
                     'Flow/liquidation 30 features; GRU uses 4 causal snapshots; source masking limitations inherited'],
        input_hashes={str(p): b.sha(p) for p in files}, source_hashes={str(p): b.sha(p) for p in paths}))
    for p in paths:
        (out / p.name).write_bytes(p.read_bytes())
    frames, selections, ledger = [], [], []
    with threadpool_limits(limits=args.threads):
        for target in ['down_magnitude_log_bps']:
            y = truth[target].to_numpy(float)
            control = old[old.target.eq(target)].set_index('decision_time')
            for i, issue in enumerate(b.ISSUES):
                end = b.ISSUES[i+1] if i+1 < len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
                te = x.index.get_indexer(dev[(dev >= issue) & (dev < end)])
                tr = np.flatnonzero(x.index + pd.Timedelta(hours=72) < issue)
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
                    previous_neural=refs.previous_neural.to_numpy(), previous_best=refs.previous_best.to_numpy())))
                selections.append(dict(target=target, issue=str(issue), **{
                    name: {**plan[name], 'features': bundle[name]['features']} for name in forecasts}))
                for name, (train, val) in [('outer', (tr, te))] + [(f'inner_{j+1}', pair) for j, pair in enumerate(folds[issue])]:
                    ledger.append(dict(target=target, issue=issue, window=name, training_rows=len(train),
                        evaluation_rows=len(val), training_label_end_max=x.index[train].max()+pd.Timedelta(hours=72),
                        evaluation_start=x.index[val].min()))
    pred = pd.concat(frames, ignore_index=True)
    b.check(len(pred) == len(dev) and not pred.duplicated(['decision_time', 'target']).any(), 'Output coverage')
    b.check(np.isfinite(pred[['selected', 'candidate']]).all().all(), 'Nonfinite forecasts')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    pred.to_csv(out / 'predictions.csv', index=False)
    rows = []
    for target, frame in pred.groupby('target'):
        for period, part in [('ALL', frame)] + list(frame.groupby('period')):
            for name in ['selected', 'candidate', 'training_median', 'training_mean', 'previous_neural', 'previous_best']:
                rows.append(dict(target=target, period=period, model=name, **b.metric(part.actual.to_numpy(), part[name].to_numpy())))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'metrics.csv', index=False)
    b.dump(out / 'selections.json', selections)
    pd.DataFrame(ledger).to_csv(out / 'fit_windows.csv', index=False)
    summary = 'DOWN72h MAE MLP/GRU; historically exposed evaluation.\n' + scores[scores.period.eq('ALL')].to_string(index=False)
    (out / 'SUMMARY.txt').write_text(summary)
    b.dump(out / 'COMPLETE.json', dict(status='COMPLETE', targets=['down_magnitude_log_bps'], development_origins_per_target=len(dev),
        replay_passed=True, predictions_sha256=b.sha(out / 'predictions.csv')))
    print(summary)
    print('Saved:', out)


if __name__ == '__main__':
    main()
