r"""Focused Ridge conditional mean magnitudes. User-run CPU training.
python .\Code\tune_ridge_mean_magnitudes_48h.py
Both UP and DOWN outputs are nonnegative ordinary-bps endpoint magnitudes.
No signed-return forecasts. Historical evaluation is already exposed.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as base

HERE = Path(__file__).resolve().parent
ALPHAS = [.0001, .001, .01, .1, 1., 10.]
WEIGHTS = [0., .25, .5, .75, 1.]
PACKS = {
    'UP': [base.COLS[i] for i in [0, 1, 3, 5]],
    'DOWN': [base.COLS[i] for i in [0, 1, 3, 5, 7]],
}
REFERENCE_NAMES = {
    'UP': ['ridge_core4', 'ridge_core4_drop_liq_breadth_1h', 'mlp_selected'],
    'DOWN': ['ridge_core4_add_basis_1h', 'ridge_full8_drop_liq_1h'],
}


def choose(rows, ridge_only=False):
    eligible = [r for r in rows if not ridge_only or r['ridge_weight'] == 1.]
    # Exact ties prefer more shrinkage and stronger regularization.
    return min(eligible, key=lambda r: (r['mse_bps2'], r['ridge_weight'], -r['alpha_per_row']))


def predict_bundle(bundle, features):
    """Label-free inference; one saved side bundle supplies all its forecast origins."""
    base.check(features.index.is_unique and features.index.tz is not None,
               'Inference needs unique timezone-aware timestamps')
    base.check((features.index >= pd.Timestamp(bundle['issue'])).all(), 'Forecast precedes model issue')
    x = features.replace([np.inf, -np.inf], np.nan)
    p = base.predict(bundle['ridge'], x)
    return bundle['ridge_weight'] * p + (1. - bundle['ridge_weight']) * bundle['training_mean_bps']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepared', type=Path, default=HERE.parent.parent /
                        'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    parser.add_argument('--reference-run', type=Path, default=base.EXP /
                        'controlled_mean_ablation_48h_v1/20261112T082233470776Z')
    parser.add_argument('--output-dir', type=Path, default=base.EXP / 'focused_ridge_mean_48h_v1')
    args = parser.parse_args()
    base.check(args.threads > 0, 'Threads must be positive')
    base.PACKS.update(PACKS)
    ref_path = args.reference_run / 'magnitude_predictions.csv'
    done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and done['predictions_sha256'] == base.sha(ref_path),
               'Reference integrity failed')
    ref = pd.read_csv(ref_path, index_col='decision_time')
    ref.index = pd.to_datetime(ref.index, utc=True)
    ref['issue'] = pd.to_datetime(ref.issue, utc=True)
    ref = ref.sort_index()
    x, y, dev, _ = joblib.load(args.prepared)
    base.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing
               and x.index.tz is not None and x.columns.is_unique, 'Invalid prepared indexes')
    dev = pd.DatetimeIndex(dev).sort_values()
    base.check(len(dev) == 18481 and ref.index.is_unique and dev.equals(ref.index)
               and dev.isin(y.index).all(), 'Exact historical comparison population required')
    x = x[base.COLS].replace([np.inf, -np.inf], np.nan)
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    base.check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Unexpected horizon')
    log_target = y.signed_return_log_bps.astype(float)
    target = pd.Series(np.abs(10000. * np.expm1(log_target.to_numpy() / 10000.)), index=y.index)
    base.check(np.isfinite(target).all() and (log_target.loc[dev] != 0).all(), 'Invalid or flat outcomes')
    np.testing.assert_allclose(target.loc[dev], ref.actual_magnitude_bps, rtol=0, atol=1e-7)
    base.check(np.array_equal(np.where(log_target.loc[dev] > 0, 'UP', 'DOWN'), ref.actual_direction),
               'Side labels differ')
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    for path, name in [(Path(__file__), 'source_snapshot.py'),
                       (Path(base.__file__), 'tuning_helper_snapshot.py')]:
        (out / name).write_bytes(path.read_bytes())
    base.dump(out / 'CONTRACT.json', dict(
        task='Focused Ridge UP/DOWN conditional mean magnitudes', symbol='BTCUSDT', horizon_hours=48,
        units='Nonnegative ordinary basis points on BOTH sides; DOWN is a magnitude, not a signed return',
        target='abs(10000 * expm1(signed_return_log_bps / 10000))',
        feature_sets=PACKS, alpha_per_row=ALPHAS, ridge_weights=WEIGHTS, seed=48,
        history='Expanding; training label_end strictly before fit cutoff; no outcome trimming',
        validation='Offsets 9,6,3 months; first two months of origins; labels mature before next quarter',
        selection='Row-weighted earlier-validation MSE; fixed feature set per side; no family switching',
        blend='weight * nonnegative Ridge prediction + (1-weight) * eligible training-side mean',
        zero_weight_policy='Declared training-average baseline; may be selected; never called a learned forecast',
        preprocessing='Train-only median imputation, standardization, +/-8 feature clipping, missing flags; affine target scaling',
        evaluation='Repeatedly exposed historical development period; feature sets proposed after prior results',
        fresh_test='None verified or evaluated by this runner; saved policy needs untouched later data for confirmation',
        reference_price='Inherited canonical endpoint-return labels; exact exchange price construction not re-established here',
        availability='Both side predictions at each reference origin; historical eligible population only; not a live availability certification',
        signed_return_forecasts=False, reference_sha256=base.sha(ref_path), prepared_sha256=base.sha(args.prepared),
        source_sha256=base.sha(Path(__file__)), helper_sha256=base.sha(Path(base.__file__)),
        packages={k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        instructions={str(p): base.sha(p) for p in [HERE.parent / 'PROJECT_CONTEXT.md',
            HERE.parent.parent.parent / 'ml_framework/AGENTS.md'] if p.exists()}))
    frames, validations, scores, selections, audits = [], [], [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            base.check(len(te) > 0 and (te >= issue).all(), 'Invalid evaluation fold')
            f = ref.loc[te, ['issue', 'actual_direction', 'actual_magnitude_bps']].copy()
            f['symbol'], f['horizon_hours'] = 'BTCUSDT', 48
            for side in ['UP', 'DOWN']:
                mask = log_target > 0 if side == 'UP' else log_target < 0
                print(f'START {issue.date()} {side}: 6 Ridge penalties x 3 validation windows', flush=True)
                candidate_scores = []
                for alpha in ALPHAS:
                    totals = {w: 0. for w in WEIGHTS}
                    count = 0
                    config = dict(family='ridge', pack=side, history_days=None, alpha_per_row=alpha)
                    for offset in [9, 6, 3]:
                        cutoff = issue - pd.DateOffset(months=offset)
                        tr = base.train_keys(y, mask, cutoff, None)
                        va = y.index[(y.index >= cutoff) & (y.index < cutoff + pd.DateOffset(months=2))
                                     & (y.label_end < cutoff + pd.DateOffset(months=3)) & mask]
                        base.check(len(va) >= 100 and y.loc[va, 'label_end'].max() < issue,
                                   'Invalid validation window')
                        artifact = base.fit(x, target, tr, config, 48)
                        vp = base.predict(artifact, x.loc[va])
                        average = float(target.loc[tr].mean())
                        count += len(va)
                        for weight in WEIGHTS:
                            m = base.metric(target.loc[va], weight * vp + (1-weight) * average)
                            totals[weight] += m['mse_bps2'] * len(va)
                            validations.append(dict(issue=issue, side=side, alpha_per_row=alpha,
                                ridge_weight=weight, offset_months=offset, cutoff=cutoff,
                                training_rows=len(tr), training_mean_bps=average,
                                max_training_label_end=y.loc[tr, 'label_end'].max(),
                                max_validation_label_end=y.loc[va, 'label_end'].max(), **m))
                    for weight in WEIGHTS:
                        candidate_scores.append(dict(alpha_per_row=alpha, ridge_weight=weight,
                                                     mse_bps2=totals[weight]/count, rows=count))
                choices = {'tuned_ridge': choose(candidate_scores, ridge_only=True),
                           'selected_blend': choose(candidate_scores)}
                scores.extend(dict(issue=issue, side=side, **r) for r in candidate_scores)
                # Freeze choices before outer fitting/scoring. Alternate masks remain comparisons only.
                selections.append(dict(issue=issue, side=side, features=PACKS[side], choices=choices))
                base.dump(out / 'SELECTIONS.json', selections)
                tr = base.train_keys(y, mask, issue, None)
                average = float(target.loc[tr].mean())
                f[f'{side}_training_mean_bps'] = average
                f[f'{side}_tail_threshold_bps'] = float(target.loc[tr].quantile(.9))
                fitted = {}
                for name, choice in choices.items():
                    alpha = choice['alpha_per_row']
                    if alpha not in fitted:
                        fitted[alpha] = base.fit(x, target, tr, dict(family='ridge', pack=side,
                            history_days=None, alpha_per_row=alpha), 48)
                    bundle = dict(issue=issue, side=side, ridge=fitted[alpha],
                        ridge_weight=choice['ridge_weight'], training_mean_bps=average,
                        policy='training_side_average' if choice['ridge_weight'] == 0 else 'ridge_blend',
                        max_training_label_end=y.loc[tr, 'label_end'].max(), units='ordinary magnitude bps')
                    pred = predict_bundle(bundle, x.loc[te])
                    f[f'{side}_{name}_bps'] = pred
                    path = out / f'{issue:%Y-%m}_{side}_{name}.joblib'
                    joblib.dump(bundle, path, compress=3)
                    np.testing.assert_allclose(pred, predict_bundle(joblib.load(path), x.loc[te]), rtol=0, atol=1e-6)
                    audits.append(dict(issue=issue, side=side, name=name, training_rows=len(tr),
                        max_training_label_end=bundle['max_training_label_end'], artifact=path.name,
                        sha256=base.sha(path), replay_passed=True, policy=bundle['policy']))
                for number, name in enumerate(REFERENCE_NAMES[side]):
                    f[f'{side}_reference_{number}_bps'] = ref.loc[te, f'{side}_{name}']
                best = choices['selected_blend']
                print(f'DONE {issue.date()} {side}: alpha/row={best["alpha_per_row"]}, '
                      f'Ridge weight={best["ridge_weight"]}, validation RMSE={np.sqrt(best["mse_bps2"]):.3f}', flush=True)
            frames.append(f)
            f.to_csv(out / f'{issue:%Y-%m}_magnitude_predictions.csv', index_label='decision_time')
            pd.DataFrame(validations).to_csv(out / 'validation_metrics.csv', index=False)
            pd.DataFrame(scores).to_csv(out / 'validation_candidates.csv', index=False)
            base.dump(out / 'FIT_AUDIT.json', audits)
    result = pd.concat(frames).sort_index()
    base.check(result.index.equals(dev), 'Final timestamp mismatch')
    result.to_csv(out / 'magnitude_predictions.csv', index_label='decision_time')
    metrics, tails = [], []
    for period, group in [('ALL', result)] + [(str(k), v) for k, v in result.groupby('issue')]:
        for side in ['UP', 'DOWN']:
            g = group.loc[group.actual_direction.eq(side)]
            names = ['training_mean', 'tuned_ridge', 'selected_blend']
            names += [f'reference_{i}' for i in range(len(REFERENCE_NAMES[side]))]
            for name in names:
                column = f'{side}_{name}_bps'
                m = base.metric(g.actual_magnitude_bps, g[column])
                label = REFERENCE_NAMES[side][int(name[-1])] if name.startswith('reference_') else name
                metrics.append(dict(period=period, side=side, model=label, **m))
                large = g.actual_magnitude_bps > g[f'{side}_tail_threshold_bps']
                for tag, mask in [('ordinary', ~large), ('large', large)]:
                    h = g.loc[mask]
                    if len(h):
                        tm = base.metric(h.actual_magnitude_bps, h[column])
                        tails.append(dict(period=period, side=side, model=label, group=tag,
                            squared_error_share=tm['mse_bps2']*len(h)/max(m['mse_bps2']*len(g), 1e-30), **tm))
    table = pd.DataFrame(metrics)
    table.to_csv(out / 'magnitude_metrics.csv', index=False)
    pd.DataFrame(tails).to_csv(out / 'tail_errors.csv', index=False)
    base.dump(out / 'REFERENCE_COLUMNS.json', REFERENCE_NAMES)
    summary = ('FOCUSED RIDGE CONDITIONAL MEAN MAGNITUDES - ordinary bps\n'
        'Both sides are positive magnitudes. No signed-return forecasts.\n'
        'Selection uses earlier validation MSE. Zero Ridge weight explicitly means training-average baseline.\n'
        'Historical exposed evaluation; no fresh-data confirmation or automatic model promotion.\n\n'
        + table[table.period.eq('ALL')].to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    base.dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(result), folds=4,
        training_performed=True, signed_return_forecasts=False, model_replay_passed=True,
        contract_sha256=base.sha(out / 'CONTRACT.json'), predictions_sha256=base.sha(out / 'magnitude_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    main()
