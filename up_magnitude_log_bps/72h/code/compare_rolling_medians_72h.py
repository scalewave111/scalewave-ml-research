"""User-run causal rolling-median comparison. No tree/neural fitting."""
from pathlib import Path
import argparse
import platform
import numpy as np
import pandas as pd
import discover_excursion_features_72h as base
from tune_excursions_72h import read_json

DEFAULT_RUN = base.HERE/'experiments/excursion_tuning_72h_v1/20261113T120003142401Z'
DAYS = [30, 90, 180, 365]
MIN_ROWS = 48


def causal_medians(labels, origins, days=DAYS, minimum=MIN_ROWS):
    """Window covers label MATURITY times in [origin-days, origin).

    Label at origin+72h is usable only strictly AFTER that maturity timestamp.
    Missing/ineligible source labels are never filled with zero.
    """
    values = labels.copy().sort_index()
    base.check(values.index.is_unique and origins.is_unique, 'Duplicate keys')
    base.check(np.isfinite(values).all() and (values >= 0).all(), 'Invalid target')
    values.index = values.index+pd.Timedelta(hours=72)
    grid = values.index.union(origins).sort_values()
    history = values.reindex(grid)
    output = pd.DataFrame(index=origins)
    counts = pd.DataFrame(index=origins)
    output['online_expanding'] = history.expanding(min_periods=minimum).median().shift(1).reindex(origins)
    counts['online_expanding'] = history.expanding().count().shift(1).reindex(origins).fillna(0)
    for window in days:
        roll = history.rolling(f'{window}D', closed='left', min_periods=minimum)
        output[f'rolling_{window}d'] = roll.median().reindex(origins)
        counts[f'rolling_{window}d'] = history.rolling(f'{window}D', closed='left', min_periods=0).count().reindex(origins)
    return output, counts


def choose(truths, controls, forecasts):
    """Earlier-only MAE gate; deterministic tie order, fallback frozen median."""
    weights = np.array([len(y) for y in truths])
    baseline = np.array([np.abs(y-p).mean() for y, p in zip(truths, controls)])
    records = []
    selected, best = 'quarterly_median', np.average(baseline, weights=weights)
    for name, predictions in forecasts.items():
        errors = np.array([np.abs(y-p).mean() for y, p in zip(truths, predictions)])
        score = float(np.average(errors, weights=weights))
        eligible = score < np.average(baseline, weights=weights)*.999 and (errors < baseline).sum() >= 2
        records.append(dict(model=name, mae=score, median_mae=float(np.average(baseline, weights=weights)),
            window_mae=errors.tolist(), baseline_window_mae=baseline.tolist(), wins=int((errors < baseline).sum()), eligible=bool(eligible)))
        if eligible and score < best:
            selected, best = name, score
    return selected, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/rolling_medians_72h_v1')
    args = parser.parse_args()
    ref = args.reference_run.resolve()
    contract, done = read_json(ref/'CONTRACT.json'), read_json(ref/'COMPLETE.json')
    base.check(done['status'] == 'COMPLETE' and done['replay_passed'] and done['original_replay_passed'], 'Incomplete reference')
    base.check(contract['horizon_minutes'] == 4320, 'Wrong horizon')
    base.check(base.sha(ref/'predictions.csv') == done['predictions_sha256'], 'Predictions changed')
    for path, digest in contract['input_hashes'].items():
        base.check(base.sha(Path(path)) == digest, 'Reference input/source changed: '+path)
    prior = read_json(Path(contract['arguments']['reference_run'])/'CONTRACT.json')
    load_args = argparse.Namespace(data_dir=Path(prior['arguments']['data_dir']), y_dir=Path(prior['arguments']['y_dir']))
    x, labels, dev, files = base.load(load_args)
    labels.index, dev = labels.index.as_unit('ns'), dev.as_unit('ns')
    del x
    old = pd.read_csv(ref/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    base.check(len(old) == 2*len(dev) and len(dev) == done['development_origins'] and not old.duplicated(['target', 'decision_time']).any(), 'Bad coverage')
    base.check(set(old.target) == set(base.TARGETS), 'Wrong targets')
    frames, trials, selections, ledger = [], [], [], []
    names = ['online_expanding']+[f'rolling_{d}d' for d in DAYS]
    for target in base.TARGETS:
        source = labels[target]
        f = old[old.target.eq(target)].set_index('decision_time').sort_index()
        base.check(f.index.equals(dev), 'Evaluation timestamps differ')
        np.testing.assert_allclose(f.actual, source.loc[dev], rtol=0, atol=1e-10)
        queries = labels.index.union(base.ISSUES).sort_values()
        values, counts = causal_medians(source, queries)
        for i, issue in enumerate(base.ISSUES):
            end = base.ISSUES[i+1] if i+1 < len(base.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            sub = f[(f.index >= issue) & (f.index < end)].copy()
            base.check(sub.issue.eq(issue).all(), 'Wrong issuance')
            folds = base.windows(labels.index, issue)
            truths, controls = [], []
            predictions = {name: [] for name in names}
            for n, (tr, va) in enumerate(folds):
                keys = labels.index[va]
                truth = source.iloc[va].to_numpy()
                median = float(source.iloc[tr].median())
                truths.append(truth)
                controls.append(np.full(len(va), median))
                for name in names:
                    p = values.loc[keys, name].to_numpy()
                    base.check(np.isfinite(p).all(), 'Insufficient mature labels for '+name)
                    predictions[name].append(p)
                    ledger.append(dict(target=target, issue=str(issue), inner_window=n+1, model=name,
                        start=str(keys.min()), end=str(keys.max()), minimum_mature_rows=int(counts.loc[keys, name].min())))
            selected, records = choose(truths, controls, predictions)
            trials.extend(dict(target=target, issue=str(issue), **record) for record in records)
            selections.append(dict(target=target, issue=str(issue), selected=selected))
            mature = source[source.index+pd.Timedelta(hours=72) < issue]
            np.testing.assert_allclose(sub.training_median, mature.median(), rtol=0, atol=1e-7)
            np.testing.assert_allclose(sub.training_mean, mature.mean(), rtol=0, atol=1e-7)
            frame = pd.DataFrame(dict(actual=sub.actual, issue=issue, target=target,
                quarterly_median=sub.training_median, quarterly_mean=sub.training_mean,
                previous_tree=sub.previous, tuned_tree=sub.selected))
            for name in names:
                frame[name] = values.loc[sub.index, name]
                frame[name+'_count'] = counts.loc[sub.index, name].astype(int)
            base.check(np.isfinite(frame[names].to_numpy()).all(), 'Missing forecast')
            frame['selected'] = frame[selected]
            frame['selected_method'] = selected
            # Prefix replay: future labels cannot change forecasts at this cutoff.
            cut = sub.index[len(sub)//2]
            replay, _ = causal_medians(source[source.index+pd.Timedelta(hours=72) < cut], pd.DatetimeIndex([cut]))
            np.testing.assert_allclose(replay.loc[cut, names], values.loc[cut, names], rtol=0, atol=1e-9)
            frame.index.name = 'decision_time'
            frames.append(frame.reset_index())
            print(f'{issue:%Y-%m} {target}: {selected}', flush=True)
    pred = pd.concat(frames, ignore_index=True)
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
    metrics = []
    for target, f in pred.groupby('target'):
        for period, part in [('ALL', f)]+list(f.groupby('period')):
            for model in ['quarterly_median', 'quarterly_mean', 'previous_tree', 'tuned_tree']+names+['selected']:
                score = base.metric(part.actual.to_numpy(), part[model].to_numpy())
                baseline = np.abs(part.actual-part.quarterly_median).mean()
                metrics.append(dict(target=target, period=period, model=model, **score, mae_gain_vs_quarterly_median=float(baseline-score['mae'])))
    out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    pred.to_csv(out/'predictions.csv', index=False)
    scores = pd.DataFrame(metrics)
    scores.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(trials).to_csv(out/'validation_trials.csv', index=False)
    pd.DataFrame(ledger).to_csv(out/'availability_audit.csv', index=False)
    base.dump(out/'selections.json', selections)
    guidance = [base.HERE.parent/'PROJECT_CONTEXT.md', base.HERE.parents[2]/'ml_framework/AGENTS.md',
        base.HERE.parents[2]/'ml_framework/docs/methods/WORKFLOW.md',base.HERE.parents[2]/'ml_framework/docs/methods/DATA_AND_METRICS.md']
    base.dump(out/'CONTRACT.json', dict(arguments=vars(args), horizon_minutes=4320, days=DAYS, minimum_rows=MIN_ROWS,
        window='Label maturity times in [forecast_time - days, forecast_time); strict72h maturity',
        selection='Quarterly earlier3window MAE; >=.1% gain and >=2/3wins vs quarterly median, otherwise quarterly median',
        online='Values update every forecast; later evaluation labels can enter only AFTER their 72h maturity. No outer-based window selection.',
        limitations=['Historically exposed data and overlapping labels; descriptive exploratory experiment',
            '30-minute eligible origin inventory, inherited48h initial gap and invalid label exclusions',
            'January2025 retains October window choice, but online median values keep updating',
            'No automatic promotion of hindsight best fixed window; no resume'],
        source_hashes={str(p):base.sha(p) for p in files+[Path(__file__), Path(base.__file__), ref/'CONTRACT.json', ref/'COMPLETE.json', ref/'predictions.csv']},
        guidance_hashes={str(p):base.sha(p) for p in guidance},
        versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__)))
    summary = '72h rolling medians; all errors in log bps. No tree/neural fitting.\n'+scores[scores.period.eq('ALL')].to_string(index=False)
    summary += '\n\nQuarterly earlier-selected policies:\n'+pd.DataFrame(selections).to_string(index=False)
    summary += '\nOnline forecasts use only labels strictly mature at issuance. Outer best fixed window is hindsight, not a selected policy.\n'
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    base.dump(out/'COMPLETE.json', dict(status='COMPLETE', development_origins=len(dev),
        prefix_replay_passed=True, predictions_sha256=base.sha(out/'predictions.csv')))
    print(summary)
    print('Saved:', out)


if __name__ == '__main__':
    main()
