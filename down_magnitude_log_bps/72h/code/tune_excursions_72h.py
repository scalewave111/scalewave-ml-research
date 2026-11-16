"""User-run 72h MAE refinement. --check-inputs performs no fitting.

Uses the completed price-follow-up as control. No reports are generated.
"""
from pathlib import Path
from itertools import product
import argparse
import json
import platform
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import improve_excursion_features_72h as prior

base = prior.base
DEFAULT_REFERENCE = base.HERE / 'experiments/excursion_pipeline_72h_v1/20261113T111815030467Z/price_followup/20261113T113500606206Z'
CONFIGS = {'original': dict(max_iter=80, learning_rate=.05, max_leaf_nodes=7, max_depth=3, min_samples_leaf=128, l2_regularization=1.)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def extended_price(r, origins):
    # Reuse the tested formulas without changing the original helper file.
    hours = prior.HOURS
    try:
        prior.HOURS = [1, 4, 12, 24, 48, 72, 168, 336, 720]
        return prior.price_features(r, origins)
    finally:
        prior.HOURS = hours


def training_rows(index, boundary, days=0):
    mask = index + pd.Timedelta(hours=72) < boundary
    if days:
        mask &= index >= boundary - pd.Timedelta(days=days)
    rows = np.flatnonzero(mask)
    base.check(len(rows) >= 1000, 'Too few mature training observations')
    return rows


def fit_candidate(a, y, config):
    return base.fit(a, y, 'tree')


def predict_candidate(model, a):
    return base.predict(model, a)


def errors(truths, predictions):
    return np.array([[np.abs(y-p).mean(), np.square(y-p).mean()]
                     for y, p in zip(truths, predictions)])


def admissible(candidate, control, median, weights):
    # Compare with the expanding training median; no RMSE objective substitution.
    return (np.average(candidate[:, 0], weights=weights) <
            np.average(median[:, 0], weights=weights)*.999 and
            int((candidate[:, 0] < median[:, 0]).sum()) >= 2)


def load_inputs(args):
    ref = args.reference_run
    done, contract = read_json(ref/'COMPLETE.json'), read_json(ref/'CONTRACT.json')
    base.check(done['status'] == 'COMPLETE' and done['replay_passed'] and
               done['original_replay_passed'], 'Reference is not complete/replay-verified')
    base.check(done['predictions_sha256'] == base.sha(ref/'predictions.csv'), 'Reference predictions changed')
    base.check(contract['horizon_minutes'] == 4320, 'Expected 72h reference')
    for path in [Path(base.__file__), Path(prior.__file__), Path(prior.canonical.__file__)]:
        base.check(contract['source_hashes'][str(path)] == base.sha(path), 'Reference helper changed: '+str(path))
    args.data_dir = Path(contract['arguments']['data_dir'])
    args.y_dir = Path(contract['arguments']['y_dir'])
    x, y, dev, files = base.load(args)
    x.index, y.index, dev = x.index.as_unit('ns'), y.index.as_unit('ns'), dev.as_unit('ns')
    returns_path = Path(contract['arguments']['returns'])
    base.check(contract['source_hashes'][str(returns_path)] == base.sha(returns_path), 'Returns changed')
    for path in files:
        base.check(contract['source_hashes'][str(path)] == base.sha(path), 'Reference input changed: '+str(path))
    r = prior.canonical.load_returns(returns_path)
    price, groups = extended_price(r, x.index)
    cut = x.index[len(x)//2]
    keys = x.index[x.index <= cut][-10:]
    pd.testing.assert_frame_equal(extended_price(r, keys)[0], extended_price(r[r.index <= cut], keys)[0])
    changed = r.copy()
    changed.loc[changed.index >= cut] = 100000.
    pd.testing.assert_frame_equal(extended_price(r, pd.DatetimeIndex([cut]))[0],
                                  extended_price(changed, pd.DatetimeIndex([cut]))[0])
    x = x.join(price)
    old = pd.read_csv(ref/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    base.check(len(old) == 2*len(dev) and not old.duplicated(['target', 'decision_time']).any(), 'Reference coverage differs')
    cards = {}
    for target in base.TARGETS:
        f = old[old.target.eq(target)].set_index('decision_time').sort_index()
        base.check(f.index.equals(dev), 'Reference timestamp mismatch')
        np.testing.assert_allclose(f.actual, y.loc[dev, target], rtol=0, atol=1e-10)
        for i, issue in enumerate(base.ISSUES):
            end = base.ISSUES[i+1] if i+1 < len(base.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            sub = f[(f.index >= issue) & (f.index < end)]
            base.check(sub.issue.eq(issue).all(), 'Reference issuance mismatch')
            folder = ref/f'{issue:%Y-%m}_{target}'
            plan = read_json(folder/'selection.json')
            spec = plan['variants'][plan['selected_variant']]
            path = folder/spec['artifact']
            base.check(base.sha(path) == spec['artifact_sha256'], 'Reference model changed')
            bundle = joblib.load(path)
            base.check(bundle['features'] == spec['features'] and bundle['variant'] == plan['selected_variant'], 'Model card mismatch')
            np.testing.assert_allclose(base.predict(bundle['model'], x.loc[sub.index, spec['features']].to_numpy(float)),
                                       sub.selected, rtol=0, atol=1e-7)
            cards[(issue, target)] = spec
            files += [path, folder/'selection.json']
    files += [returns_path, ref/'CONTRACT.json', ref/'COMPLETE.json', ref/'predictions.csv',
              Path(base.__file__), Path(prior.__file__), Path(prior.canonical.__file__), Path(__file__)]
    return x, y, dev, old, cards, groups, files


def run_cycle(x, truth, issue, control_spec, groups, folder):
    folds = base.windows(x.index, issue)
    cols = control_spec['features']
    control, median, truths, ledger = [], [], [], []
    for tr, va in folds:
        model = base.fit(x.iloc[tr][cols].to_numpy(float), truth[tr], control_spec['model'])
        control.append(base.predict(model, x.iloc[va][cols].to_numpy(float)))
        median.append(np.full(len(va), np.median(truth[tr])))
        truths.append(truth[va])
    weights = [len(v) for _, v in folds]
    control_errors, median_errors = errors(truths, control), errors(truths, median)
    if 'validation_window_mae' in control_spec:
        np.testing.assert_allclose(control_errors[:, 0], control_spec['validation_window_mae'],
                                   rtol=0, atol=1e-7, err_msg='Earlier control scores failed replay')
    long_columns = [c for h in [72, 168, 336, 720] for c in groups[f'price_{h}h']]
    packs = {'current': cols, 'current_plus_long': list(dict.fromkeys(cols+long_columns))}
    trials, best = [], None
    for pack, config, days in product(packs, CONFIGS, [0, 180, 365]):
        columns = packs[pack]
        predictions = []
        for n, (_, va) in enumerate(folds):
            boundary = issue - pd.DateOffset(months=[9, 6, 3][n])
            tr = training_rows(x.index, boundary, days)
            a, b = x.iloc[tr][columns].to_numpy(float), x.iloc[va][columns].to_numpy(float)
            model = fit_candidate(a, truth[tr], config)
            predictions.append(predict_candidate(model, b))
            ledger.append(dict(pack=pack, config=config, days=days, window=n+1,
                training_rows=len(tr), training_label_end_max=x.index[tr].max()+pd.Timedelta(hours=72),
                validation_start=x.index[va].min(), validation_label_end_max=x.index[va].max()+pd.Timedelta(hours=72)))
        for weight in [0., .25, .5, .75, 1.]:
            blended = [weight*p+(1-weight)*c for p, c in zip(predictions, median)]
            e = errors(truths, blended)
            eligible = admissible(e, control_errors, median_errors, weights)
            branch = ('long_features' if pack != 'current' and days == 0 and weight == 1 else
                      'history' if pack == 'current' and days != 0 and weight == 1 else
                      'median_blend' if pack == 'current' and days == 0 else 'combined')
            row = dict(branch=branch, pack=pack, config=config, days=days, weight=weight,
                mae=float(np.average(e[:, 0], weights=weights)),
                rmse=float(np.sqrt(np.average(e[:, 1], weights=weights))), eligible=bool(eligible),
                window_mae=e[:, 0].tolist(), window_mse=e[:, 1].tolist())
            trials.append(row)
            if eligible and (best is None or row['mae'] < best['mae']):
                best = dict(**row, features=columns)
        print(f'  {pack}/{config}/history={days or "all"} complete', flush=True)
    pd.DataFrame(trials).to_csv(folder/'validation_trials.csv', index=False)
    pd.DataFrame(ledger).to_csv(folder/'inner_fit_windows.csv', index=False)
    base.dump(folder/'selection.json', dict(selected=best, control_features=cols,
        control_window_errors=control_errors.tolist(), median_window_errors=median_errors.tolist()))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/excursion_tuning_72h_v1')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    base.check(args.threads > 0, 'threads must be positive')
    with threadpool_limits(limits=args.threads):
        x, y, dev, old, cards, groups, files = load_inputs(args)
        for issue in base.ISSUES:
            base.windows(x.index, issue)
        print(f'Inputs/reference replay passed: {len(x)} rows; {len(dev)} evaluation origins; 75 price features.', flush=True)
        if args.check_inputs:
            return
        out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        out.mkdir(parents=True, exist_ok=False)
        guidance = [base.HERE.parent/'PROJECT_CONTEXT.md', base.HERE.parents[2]/'ml_framework/AGENTS.md',
                    base.HERE.parents[2]/'ml_framework/docs/methods/WORKFLOW.md',
                    base.HERE.parents[2]/'ml_framework/docs/methods/DATA_AND_METRICS.md']
        base.dump(out/'CONTRACT.json', dict(arguments=vars(args), horizon_minutes=4320, configs=CONFIGS,
            history_days=[0, 180, 365], blend_weights=[0., .25, .5, .75, 1.], objective='MAE',
            gate='>=0.1% pooled MAE improvement versus expanding median and >=2/3 window wins; otherwise expanding median',
            preprocessing='Original training-only median imputation/scaling/missing flags; original fixed MAE boosting; no early stopping; nonnegative predictions',
            limitations=['Previously exposed historical evaluation; no independent confirmation',
                        'Reference feature lists already selected using these earlier windows; adaptive reuse, not nested validation',
                        'Original masked X availability/provisional limitations inherited; original 48h boundary gap retained',
                        'Validation guardrail does not guarantee lower outer MAE or RMSE', 'No automatic resume'],
            input_hashes={str(p):base.sha(p) for p in files}, guidance_hashes={str(p):base.sha(p) for p in guidance},
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=base.sklearn.__version__)))
        for path in [Path(__file__), Path(base.__file__), Path(prior.__file__)]:
            (out/path.name).write_bytes(path.read_bytes())
        frames, selections, ledger, diagnostics = [], [], [], []
        for issue in base.ISSUES:
            for target in base.TARGETS:
                print(f'{issue:%Y-%m} {target}', flush=True)
                folder = out/f'{issue:%Y-%m}_{target}'
                folder.mkdir()
                truth = y[target].to_numpy(float)
                spec = cards[(issue, target)]
                chosen = run_cycle(x, truth, issue, spec, groups, folder)
                ref = old[old.issue.eq(issue) & old.target.eq(target)].set_index('decision_time').sort_index()
                control = ref.selected.to_numpy()
                tr = training_rows(x.index, issue)
                original_model = base.fit(x.iloc[tr][spec['features']].to_numpy(float), truth[tr], spec['model'])
                np.testing.assert_allclose(base.predict(original_model, x.loc[ref.index, spec['features']].to_numpy(float)), control, rtol=0, atol=1e-7)
                bundle = dict(control=original_model, control_features=spec['features'], candidate=None,
                              selection=chosen, issue=issue, target=target, median=float(np.median(truth[tr])))
                prediction = np.full(len(ref), bundle['median'])
                if chosen is not None:
                    tr = training_rows(x.index, issue, chosen['days'])
                    model = fit_candidate(x.iloc[tr][chosen['features']].to_numpy(float), truth[tr], chosen['config'])
                    prediction = chosen['weight']*predict_candidate(model, x.loc[ref.index, chosen['features']].to_numpy(float))+(1-chosen['weight'])*bundle['median']
                    bundle['candidate'] = model
                for population, rows in [('training', x.index[tr]), ('evaluation', ref.index)]:
                    actual = y.loc[rows, target].to_numpy(float)
                    current = base.predict(original_model, x.loc[rows, spec['features']].to_numpy(float))
                    selected = predict_bundle(bundle, x.loc[rows])
                    for name, values in [('current', current), ('selected', selected),
                                         ('median', np.full(len(rows), bundle['median']))]:
                        diagnostics.append(dict(issue=issue, target=target, population=population,
                            model=name, actual_mean=float(actual.mean()), predicted_mean=float(values.mean()),
                            actual_median=float(np.median(actual)), predicted_median=float(np.median(values)),
                            **base.metric(actual, values)))
                joblib.dump(bundle, folder/'model.joblib')
                replay = predict_bundle(joblib.load(folder/'model.joblib'), x.loc[ref.index])
                np.testing.assert_allclose(prediction, replay, rtol=0, atol=1e-7)
                base.dump(folder/'MODEL_HASH.json', dict(sha256=base.sha(folder/'model.joblib')))
                frames.append(pd.DataFrame(dict(decision_time=ref.index, issue=issue, target=target,
                    actual=ref.actual.to_numpy(), previous=control, selected=prediction,
                    training_mean=ref.training_mean.to_numpy(), training_median=ref.training_median.to_numpy())))
                selections.append(dict(issue=str(issue), target=target, selected=chosen))
                ledger.append(dict(issue=issue, target=target, candidate_training_rows=len(tr),
                    candidate_training_label_end_max=x.index[tr].max()+pd.Timedelta(hours=72)))
        pred = pd.concat(frames, ignore_index=True)
        base.check(len(pred) == 2*len(dev) and not pred.duplicated(['decision_time', 'target']).any(), 'Output coverage failed')
        pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
        pred.loc[pred.decision_time >= pd.Timestamp('2025-01-01', tz='UTC'), 'period'] = '2025-01'
        pred.to_csv(out/'predictions.csv', index=False)
        scores = []
        for target, g in pred.groupby('target'):
            for period, f in [('ALL', g)]+list(g.groupby('period')):
                for model in ['previous', 'selected', 'training_mean', 'training_median']:
                    scores.append(dict(target=target, period=period, model=model,
                        **base.metric(f.actual.to_numpy(), f[model].to_numpy())))
        metrics = pd.DataFrame(scores)
        metrics.to_csv(out/'metrics.csv', index=False)
        pd.DataFrame(ledger).to_csv(out/'outer_fit_windows.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(out/'level_and_fit_diagnostics.csv', index=False)
        base.dump(out/'selections.json', selections)
        summary = metrics[metrics.period.eq('ALL')].to_string(index=False)
        (out/'SUMMARY.txt').write_text('72h MAE refinement; all errors in log bps.\n'+summary+'\nHistorical exploratory evaluation; no guaranteed gain.\n', encoding='utf-8')
        base.dump(out/'COMPLETE.json', dict(status='COMPLETE', development_origins=len(dev),
            replay_passed=True, original_replay_passed=True, predictions_sha256=base.sha(out/'predictions.csv')))
        print(summary)
        print('Saved:', out)


def predict_bundle(bundle, x):
    if bundle['candidate'] is None:
        return np.full(len(x), bundle['median'])
    s = bundle['selection']
    candidate = predict_candidate(bundle['candidate'], x[s['features']].to_numpy(float))
    return s['weight']*candidate+(1-s['weight'])*bundle['median']


if __name__ == '__main__':
    main()
