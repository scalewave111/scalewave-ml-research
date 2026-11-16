"""Diagnose saved 72h forecasts using inference only; never fit a model."""
from pathlib import Path
import argparse
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_excursions_72h as tuning

base = tuning.base
DEFAULT_RUN = base.HERE/'experiments/excursion_tuning_72h_v1/20261113T120003142401Z'
MODELS = ['previous', 'selected', 'training_median', 'training_mean']


def no_fitting(*args, **kwargs):
    raise RuntimeError('Fitting is prohibited in this diagnostic runner')


def describe(frame, column):
    y = frame.actual.to_numpy(float)
    p = frame[column].to_numpy(float)
    median = frame.training_median.to_numpy(float)
    gain = np.abs(y-median)-np.abs(y-p)
    adjustment = p-median
    moved = np.abs(adjustment) > 1e-9
    return dict(**base.metric(y, p), actual_mean=float(y.mean()),
        predicted_mean=float(p.mean()), actual_median=float(np.median(y)),
        predicted_median=float(np.median(p)), mean_bias=float((p-y).mean()),
        mae_gain_vs_median=float(gain.mean()), total_absolute_error_gain=float(gain.sum()),
        fraction_helped=float((gain > 1e-9).mean()), fraction_hurt=float((gain < -1e-9).mean()),
        fraction_tied=float((np.abs(gain) <= 1e-9).mean()),
        mean_abs_adjustment=float(np.abs(adjustment).mean()),
        adjustment_toward_actual_fraction=float((adjustment[moved]*(y-median)[moved] > 0).mean()) if moved.any() else None)


def bins(values, low, high, labels):
    values = np.asarray(values, dtype=float)
    return np.where(~np.isfinite(values), 'missing',
                    np.where(values <= low, labels[0], np.where(values <= high, labels[1], labels[2])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/excursion_diagnostics_72h_v1')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    base.check(args.threads > 0, 'threads must be positive')
    # Guard imported training entry points, including accidental future calls.
    base.fit = no_fitting
    tuning.fit_candidate = no_fitting
    tuning.run_cycle = no_fitting
    run = args.run_dir.resolve()
    complete = tuning.read_json(run/'COMPLETE.json')
    contract = tuning.read_json(run/'CONTRACT.json')
    base.check(complete['status'] == 'COMPLETE' and complete['replay_passed'] and complete['original_replay_passed'], 'Incomplete reference')
    base.check(contract['horizon_minutes'] == 4320, 'Expected 72h run')
    base.check(base.sha(run/'predictions.csv') == complete['predictions_sha256'], 'Prediction hash mismatch')
    for path, expected in contract['input_hashes'].items():
        base.check(base.sha(Path(path)) == expected, 'Input/source changed: '+path)
    with threadpool_limits(limits=args.threads):
        inputs = argparse.Namespace(reference_run=Path(contract['arguments']['reference_run']))
        x, y, dev, old, cards, groups, files = tuning.load_inputs(inputs)
        pred = pd.read_csv(run/'predictions.csv')
        pred['decision_time'] = pd.to_datetime(pred.decision_time, utc=True).dt.as_unit('ns')
        pred['issue'] = pd.to_datetime(pred.issue, utc=True)
        base.check(len(dev) == complete['development_origins'], 'Origin count changed')
        base.check(len(pred) == 2*len(dev) and not pred.duplicated(['target', 'decision_time']).any(), 'Duplicate/incomplete coverage')
        base.check(set(pred.target) == set(base.TARGETS), 'Unexpected targets')
        base.check(np.isfinite(pred[['actual']+MODELS].to_numpy(float)).all(), 'Nonfinite forecasts')
        base.check((pred[['actual']+MODELS].to_numpy(float) >= 0).all(), 'Negative excursion')
        period = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
        period.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC')] = '2025-01'
        base.check(pred.period.equals(period), 'Period mapping differs')
        metrics, training, windows, transfer, thresholds, annotations = [], [], [], [], [], []
        evidence = [run/'COMPLETE.json', run/'CONTRACT.json', run/'predictions.csv', run/'metrics.csv']
        for target in base.TARGETS:
            f = pred[pred.target.eq(target)].set_index('decision_time').sort_index()
            prior = old[old.target.eq(target)].set_index('decision_time').sort_index()
            base.check(f.index.equals(dev), 'Timestamp mismatch')
            np.testing.assert_allclose(f.actual, y.loc[dev, target], rtol=0, atol=1e-10)
            np.testing.assert_allclose(f.previous, prior.selected, rtol=0, atol=1e-7)
            base.check(f.issue.equals(prior.issue), 'Issuance mismatch')
            for model in MODELS:
                for period_name, part in [('ALL', f)]+list(f.groupby('period')):
                    metrics.append(dict(target=target, period=period_name, model=model, **describe(part, model)))
            for issue, part in f.groupby('issue'):
                folder = run/f'{issue:%Y-%m}_{target}'
                path = folder/'model.joblib'
                base.check(base.sha(path) == tuning.read_json(folder/'MODEL_HASH.json')['sha256'], 'Model hash mismatch')
                bundle = joblib.load(path)
                plan = tuning.read_json(folder/'selection.json')
                chosen = plan['selected']
                base.check(bundle['selection'] == chosen and bundle['issue'] == issue and bundle['target'] == target, 'Bundle metadata mismatch')
                np.testing.assert_allclose(tuning.predict_bundle(bundle, x.loc[part.index]), part.selected, rtol=0, atol=1e-7)
                np.testing.assert_allclose(base.predict(bundle['control'], x.loc[part.index, bundle['control_features']].to_numpy(float)), part.previous, rtol=0, atol=1e-7)
                expanding = tuning.training_rows(x.index, issue)
                tr = tuning.training_rows(x.index, issue, chosen['days'] if chosen else 0)
                median = float(np.median(y[target].iloc[expanding]))
                np.testing.assert_allclose(part.training_median, median, rtol=0, atol=1e-7)
                np.testing.assert_allclose(bundle['median'], median, rtol=0, atol=1e-7)
                np.testing.assert_allclose(part.training_mean, y[target].iloc[expanding].mean(), rtol=0, atol=1e-7)
                for population, rows in [('training_candidate_history', x.index[tr]), ('training_expanding', x.index[expanding]), ('evaluation', part.index)]:
                    frame = pd.DataFrame({'actual': y.loc[rows, target], 'training_median': median})
                    frame['selected'] = tuning.predict_bundle(bundle, x.loc[rows])
                    frame['previous'] = base.predict(bundle['control'], x.loc[rows, bundle['control_features']].to_numpy(float))
                    for model in ['previous', 'selected', 'training_median']:
                        training.append(dict(target=target, issue=str(issue), population=population, model=model, **describe(frame, model)))
                # Thresholds use expanding mature training only, never evaluation labels.
                vol = x['NEWPRICE__rv__72h']
                finite = vol.iloc[expanding].dropna()
                base.check(len(finite) >= 1000, 'Too little training volatility data')
                vlo, vhi = finite.quantile([1/3, 2/3]).to_numpy()
                alo, ahi = y[target].iloc[expanding].quantile([.5, .9]).to_numpy()
                thresholds.append(dict(target=target, issue=str(issue), volatility_low=vlo, volatility_high=vhi,
                    actual_training_p50=alo, actual_training_p90=ahi, training_rows=len(expanding),
                    max_training_label_end=str(x.index[expanding].max()+pd.Timedelta(hours=72))))
                tagged = part.reset_index()
                tagged['past_72h_volatility'] = vol.loc[part.index].to_numpy()
                tagged['volatility_group'] = bins(tagged.past_72h_volatility, vlo, vhi, ['low', 'medium', 'high'])
                tagged['actual_size_group'] = bins(tagged.actual, alo, ahi, ['small_le_training_p50', 'medium_to_training_p90', 'large_gt_training_p90'])
                annotations.append(tagged)
                folds = base.windows(x.index, issue)
                med_errors = np.array(plan['median_window_errors'])[:, 0]
                selected_errors = np.array(chosen['window_mae']) if chosen else med_errors
                control_errors = np.array(plan['control_window_errors'])[:, 0]
                for n, (_, va) in enumerate(folds):
                    windows.append(dict(target=target, issue=str(issue), window=n+1, rows=len(va),
                        start=str(x.index[va].min()), end=str(x.index[va].max()), median_mae=med_errors[n],
                        previous_mae=control_errors[n], selected_mae=selected_errors[n],
                        selected_gain_vs_median=med_errors[n]-selected_errors[n]))
                weights = [len(va) for _, va in folds]
                gains = med_errors-selected_errors
                positive = np.maximum(gains, 0)*weights
                transfer.append(dict(target=target, issue=str(issue),
                    inner_gain_vs_median=float(np.average(gains, weights=weights)),
                    inner_windows_won=int((gains > 0).sum()),
                    largest_window_share_of_positive_gain=float(positive.max()/positive.sum()) if positive.sum() else None,
                    outer_gain_vs_median=describe(part, 'selected')['mae_gain_vs_median'],
                    history_days=chosen['days'] if chosen else 0, weight=chosen['weight'] if chosen else 0,
                    pack=chosen['pack'] if chosen else 'median_fallback'))
                evidence += [path, folder/'MODEL_HASH.json', folder/'selection.json', folder/'validation_trials.csv']
        result = pd.DataFrame(metrics)
        saved = pd.read_csv(run/'metrics.csv')
        keys = ['target', 'period', 'model']
        a = result.set_index(keys).sort_index()
        b = saved.set_index(keys).sort_index()
        base.check(a.index.equals(b.index), 'Metric rows differ')
        np.testing.assert_allclose(a[['rows', 'mae', 'rmse', 'ic']], b[['rows', 'mae', 'rmse', 'ic']], rtol=1e-9, atol=1e-7, equal_nan=True)
        tagged = pd.concat(annotations, ignore_index=True)
        grouped = []
        for dimension in ['volatility_group', 'actual_size_group']:
            for target, target_frame in tagged.groupby('target'):
                for period_name, part in [('ALL', target_frame)]+list(target_frame.groupby('period')):
                    for group, sub in part.groupby(dimension):
                        for model in ['previous', 'selected', 'training_median']:
                            grouped.append(dict(target=target, period=period_name, dimension=dimension, group=group,
                                model=model, **describe(sub, model)))
        out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        out.mkdir(parents=True, exist_ok=False)
        for name, data in [('period_metrics', result), ('training_vs_evaluation', training),
                           ('validation_windows', windows), ('validation_transfer', transfer),
                           ('group_metrics', grouped), ('training_thresholds', thresholds)]:
            pd.DataFrame(data).to_csv(out/f'{name}.csv', index=False)
        tagged.to_csv(out/'annotated_predictions.csv', index=False)
        guidance = [base.HERE.parent/'PROJECT_CONTEXT.md', base.HERE.parents[2]/'ml_framework/AGENTS.md',
            base.HERE.parents[2]/'ml_framework/docs/methods/WORKFLOW.md', base.HERE.parents[2]/'ml_framework/docs/methods/DATA_AND_METRICS.md']
        base.dump(out/'AUDIT.json', dict(run=str(run), training=False, metrics_reproduced=True,
            model_replay=True, rows=len(pred), development_origins=len(dev), units='log bps',
            source_hashes={str(p):base.sha(p) for p in set(files+evidence+[Path(__file__)])},
            guidance_hashes={str(p):base.sha(p) for p in guidance},
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=base.sklearn.__version__)))
        summary = '\n'.join(['72h saved-forecast diagnostics; NO TRAINING; log bps.',
            'Prediction/model hashes, saved-model replay and all reported metrics checked.',
            result[result.period.eq('ALL')][['target', 'model', 'mae', 'rmse', 'ic', 'mae_gain_vs_median', 'fraction_helped', 'fraction_hurt']].to_string(index=False),
            '\nValidation-to-evaluation transfer (positive gains mean lower MAE):', pd.DataFrame(transfer).to_string(index=False),
            '\nTraining errors are in-sample, not evidence of future performance.',
            'Actual-size groups use FUTURE outcomes for diagnosis only; never use them as prediction gates.',
            'Volatility groups use past inputs and training-only thresholds. Group results are exploratory.',
            'Inner validation errors are saved records, not independently replayed (inner models were not saved).',
            'Overlapping 72h labels are dependent; frequencies are descriptive, not significance tests.',
            'Historical evaluation is exposed. No automatic model promotion or causal explanation.'])
        (out/'SUMMARY.txt').write_text(summary+'\n', encoding='utf-8')
        (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        base.dump(out/'COMPLETE.json', dict(status='COMPLETE', training=False, replay_passed=True,
            metrics_reproduced=True, summary_sha256=base.sha(out/'SUMMARY.txt')))
        print(summary)
        print('Saved:', out)


if __name__ == '__main__':
    main()
