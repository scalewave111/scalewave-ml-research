r"""Controlled UP/DOWN conditional-mean feature experiment. NEW TRAINING.

python .\Code\controlled_mean_feature_ablation_48h.py
Fixed expanding history and model settings; single-feature additions/removals.
No direction probabilities or signed-return forecasts. Requires the tuning helper
tune_conditional_mean_magnitudes_48h.py alongside this script.
"""
from pathlib import Path
import argparse
import json
import gc

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_conditional_mean_magnitudes_48h as base

HERE = Path(__file__).resolve().parent
IDS = ['liq_15m', 'liq_5m', 'liq_1h', 'vol_ratio_4h_24h', 'vol_168h',
       'liq_breadth_1h', 'funding_5m', 'basis_1h']
FEATURES = dict(zip(IDS, base.COLS))
CORE = ['liq_15m', 'liq_5m', 'vol_ratio_4h_24h', 'liq_breadth_1h']
SETS = {'core4': CORE, 'full8': IDS}
CONTRASTS = []
for name in CORE:
    key = 'core4_drop_' + name
    SETS[key] = [c for c in CORE if c != name]
    CONTRASTS.append(dict(feature=name, experiment='remove_from_core4', without=key, with_feature='core4'))
for name in IDS:
    if name not in CORE:
        key = 'core4_add_' + name
        SETS[key] = CORE + [name]
        CONTRASTS.append(dict(feature=name, experiment='add_to_core4', without='core4', with_feature=key))
    key = 'full8_drop_' + name
    SETS[key] = [c for c in IDS if c != name]
    CONTRASTS.append(dict(feature=name, experiment='remove_from_full8', without=key, with_feature='full8'))
# Register explicit feature masks in the shared fit helper. No helper file edits.
base.PACKS = {key: [FEATURES[c] for c in values] for key, values in SETS.items()}


def recipes(include_mlp):
    result = dict(ridge=dict(family='ridge', alpha_per_row=.01),
                  rf=dict(family='rf', trees=128, depth=3, leaf=128, max_features=1., sample=4096))
    if include_mlp:
        result['mlp'] = dict(family='mlp', hidden=[8, 4], alpha=.1,
                            learning_rate=.001, iterations=160, activation='relu')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-mlp', action='store_true', help='Also run fixed8/4 MLP; increases fits by50percent')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--prepared', type=Path, default=HERE.parent.parent / 'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    p.add_argument('--reference-run', type=Path, default=base.EXP / 'conditional_mean_magnitude_tuning_v1/20261112T072620594416Z')
    p.add_argument('--output-dir', type=Path, default=base.EXP / 'controlled_mean_ablation_48h_v1')
    args = p.parse_args()
    base.check(args.threads > 0, 'Threads must be positive')
    models = recipes(args.include_mlp)
    ref_path = args.reference_run / 'magnitude_predictions.csv'
    done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    base.check(done['status'] == 'COMPLETE' and done['predictions_sha256'] == base.sha(ref_path), 'Reference integrity failed')
    ref = pd.read_csv(ref_path, index_col='decision_time', parse_dates=True)
    ref.index = pd.to_datetime(ref.index, utc=True)
    ref['issue'] = pd.to_datetime(ref.issue, utc=True)
    base.check(ref.index.is_unique and not ref.index.hasnans, 'Invalid reference timestamps')
    ref = ref.sort_index()
    x, y, dev, _ = joblib.load(args.prepared)
    base.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing
               and x.index.tz is not None and x.columns.is_unique, 'Invalid prepared indexes')
    dev = pd.DatetimeIndex(dev).sort_values()
    base.check(len(dev) == 18481 and dev.equals(ref.index) and dev.isin(y.index).all(), 'Exact evaluation population required')
    x = x[base.COLS].replace([np.inf, -np.inf], np.nan)
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    base.check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Unexpected target horizon')
    log_target = y.signed_return_log_bps.astype(float)
    target = pd.Series(np.abs(10000. * np.expm1(log_target.to_numpy() / 10000.)), index=y.index)
    base.check(np.isfinite(target).all() and (log_target.loc[dev] != 0).all(), 'Nonfinite or flat outcomes')
    np.testing.assert_allclose(target.loc[dev], ref.actual_magnitude_bps, rtol=0, atol=1e-7)
    base.check(np.array_equal(np.where(log_target.loc[dev] > 0, 'UP', 'DOWN'), ref.actual_direction), 'Side labels differ')
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'tuning_helper_snapshot.py').write_bytes(Path(base.__file__).read_bytes())
    base.dump(out / 'CONTRACT.json', dict(task='Controlled mean-magnitude feature ablation only',
        seed=48, feature_names=FEATURES, feature_sets=SETS, contrasts=CONTRASTS, model_recipes=models,
        history='Expanding only. Identical side-specific rows across all feature/model comparisons per cutoff.',
        loss='Original-scale squared error through train-only affine target normalization; no target clipping',
        hyperparameters='Frozen; no retuning inside feature contrasts',
        selection='Lowest row-weighted MSE across3 earlier windows; separate choice for each side and model family',
        tail_definition='Magnitude above P90 of eligible expanding side training labels at the applicable cutoff',
        calibration='Selected model low/medium/high boundaries from pooled earlier-validation predictions; no calibration fitting',
        contribution_sign='positive MSE_without_feature minus MSE_with_feature means feature reduced error',
        feature_effects='Retrained-model predictive comparisons conditional on other features, not causal effects; historical exploratory, no significance claim',
        inner_fits=4 * 2 * 3 * len(SETS) * len(models), outer_fits=4 * 2 * len(SETS) * len(models),
        baseline_is_benchmark_only=True, signed_return_forecasts=False,
        source_sha256=base.sha(Path(__file__)), helper_sha256=base.sha(Path(base.__file__)),
        prepared_sha256=base.sha(args.prepared), reference_sha256=base.sha(ref_path)))
    frames, validation, audits, selections, tail_rows, calibration = [], [], [], [], [], []
    diagnostics = []
    with threadpool_limits(limits=args.threads):
        for issue in base.ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            base.check(len(te) > 0, 'Empty outer fold')
            f = pd.DataFrame(dict(issue=str(issue), symbol='BTCUSDT', horizon_hours=48,
                actual_direction=np.where(log_target.loc[te] > 0, 'UP', 'DOWN'),
                actual_magnitude_bps=target.loc[te]), index=te)
            for side in ['UP', 'DOWN']:
                mask = log_target > 0 if side == 'UP' else log_target < 0
                outer = base.train_keys(y, mask, issue, None)
                baseline, threshold = float(target.loc[outer].mean()), float(target.loc[outer].quantile(.9))
                f[f'{side}_training_mean_bps'], f[f'{side}_tail_threshold_bps'] = baseline, threshold
                f[f'{side}_previous_best_bps'] = ref.loc[te, f'{side}_best_mean_bps']
                f[f'{side}_previous_ensemble_bps'] = ref.loc[te, f'{side}_ensemble_mean_bps']
                all_scores = []
                for family, recipe in models.items():
                    candidates = []
                    for feature_set in SETS:
                        config = dict(recipe, pack=feature_set, history_days=None)
                        losses, count, vp_all = 0., 0, []
                        for months in [9, 6, 3]:
                            cutoff = issue - pd.DateOffset(months=months)
                            tr = base.train_keys(y, mask, cutoff, None)
                            va = y.index[(y.index >= cutoff) & (y.index < cutoff + pd.DateOffset(months=2))
                                         & (y.label_end < cutoff + pd.DateOffset(months=3)) & mask]
                            base.check(len(va) >= 100 and y.loc[va, 'label_end'].max() < issue, 'Invalid validation rows')
                            print(f'{issue.date()} {side} {family}/{feature_set} validation offset={months}', flush=True)
                            artifact = base.fit(x, target, tr, config, 48)
                            vp = base.predict(artifact, x.loc[va])
                            m = base.metric(target.loc[va], vp)
                            losses += m['mse_bps2'] * len(va)
                            count += len(va)
                            vp_all.extend(vp.tolist())
                            validation.append(dict(issue=str(issue), side=side, family=family, feature_set=feature_set,
                                offset_months=months, training_rows=len(tr), max_training_label_end=str(y.loc[tr, 'label_end'].max()),
                                fit_warnings=json.dumps(artifact['fit_warnings']), **m))
                        candidates.append(dict(family=family, feature_set=feature_set, mse=losses/count,
                                               rows=count, cuts=np.quantile(vp_all, [1/3, 2/3]).tolist()))
                        # Same fixed settings/expanding rows, fitted before outer outcomes are scored.
                        artifact = base.fit(x, target, outer, config, 48)
                        artifact.update(issue=issue, side=side, max_training_label_end=y.loc[outer, 'label_end'].max())
                        diagnostics.append(dict(issue=str(issue), side=side, family=family, feature_set=feature_set,
                            stage='train_in_sample', **base.metric(target.loc[outer], base.predict(artifact, x.loc[outer]))))
                        column = f'{side}_{family}_{feature_set}'
                        pred = base.predict(artifact, x.loc[te])
                        f[column] = pred
                        path = out / f'{issue:%Y-%m}_{side}_{family}_{feature_set}.joblib'
                        joblib.dump(artifact, path, compress=3)
                        np.testing.assert_allclose(pred, base.predict(joblib.load(path), x.loc[te]), rtol=0, atol=1e-6)
                        audits.append(dict(issue=issue, side=side, family=family, feature_set=feature_set,
                            training_rows=len(outer), max_training_label_end=y.loc[outer, 'label_end'].max(),
                            artifact=path.name, sha256=base.sha(path), replay_passed=True, fit_warnings=artifact['fit_warnings']))
                        pd.DataFrame(validation).to_csv(out / 'validation_metrics.csv', index=False)
                        del artifact
                    best = min(candidates, key=lambda c: (c['mse'], len(SETS[c['feature_set']]), c['feature_set']))
                    f[f'{side}_{family}_selected'] = f[f'{side}_{family}_{best["feature_set"]}']
                    all_scores.extend(candidates)
                    selections.append(dict(issue=issue, side=side, policy=family, **best))
                best = min(all_scores, key=lambda c: (c['mse'], len(SETS[c['feature_set']]), c['family'], c['feature_set']))
                f[f'{side}_selected_mean_bps'] = f[f'{side}_{best["family"]}_{best["feature_set"]}']
                f[f'{side}_selected_mean_pct'] = f[f'{side}_selected_mean_bps'] / 100.
                selections.append(dict(issue=issue, side=side, policy='selected', **best))
                # Descriptive calibration using earlier validation cutoffs, no test-fit adjustment.
                g = f.loc[f.actual_direction.eq(side)]
                col = f'{side}_selected_mean_bps'
                lo, hi = best['cuts']
                groups = np.where(g[col] <= lo, 'low', np.where(g[col] <= hi, 'medium', 'high'))
                for label in ['low', 'medium', 'high']:
                    h = g.loc[groups == label]
                    if len(h):
                        calibration.append(dict(issue=str(issue), side=side, group=label, lower_cut_bps=lo,
                            upper_cut_bps=hi, **base.metric(h.actual_magnitude_bps, h[col])))
                base.dump(out / 'SELECTIONS.json', selections)
                base.dump(out / 'FIT_AUDIT.json', audits)
                print(f'DONE {issue.date()} {side}: selected {best["family"]}/{best["feature_set"]}', flush=True)
                gc.collect()
            frames.append(f)
            f.to_csv(out / f'{issue:%Y-%m}_magnitude_predictions.csv', index_label='decision_time')
    result = pd.concat(frames).sort_index()
    base.check(result.index.equals(dev), 'Final timestamp mismatch')
    result.to_csv(out / 'magnitude_predictions.csv', index_label='decision_time')
    metrics, effects = [], []
    for period, f in [('ALL', result)] + [(str(k), g) for k, g in result.groupby('issue')]:
        for side in ['UP', 'DOWN']:
            g = f.loc[f.actual_direction.eq(side)]
            if g.empty:
                continue
            names = [f'{family}_{s}' for family in models for s in list(SETS) + ['selected']]
            names += ['selected_mean_bps', 'training_mean_bps', 'previous_best_bps', 'previous_ensemble_bps']
            scores = {}
            for name in names:
                column = f'{side}_{name}'
                m = base.metric(g.actual_magnitude_bps, g[column])
                scores[name] = m
                metrics.append(dict(period=period, side=side, model=name, **m))
                tail = g.actual_magnitude_bps > g[f'{side}_tail_threshold_bps']
                for label, tail_mask in [('ordinary', ~tail), ('large', tail)]:
                    h = g.loc[tail_mask]
                    if len(h):
                        hm = base.metric(h.actual_magnitude_bps, h[column])
                        tail_rows.append(dict(period=period, side=side, model=name, group=label,
                            squared_error_share=(hm['mse_bps2'] * len(h)) / max(m['mse_bps2'] * len(g), 1e-30),
                            **hm))
            for family in models:
                for c in CONTRASTS:
                    without, present = scores[f'{family}_{c["without"]}'], scores[f'{family}_{c["with_feature"]}']
                    effects.append(dict(period=period, side=side, family=family, **c,
                        feature_column=FEATURES[c['feature']], rows=len(g),
                        mse_reduction_from_feature=without['mse_bps2']-present['mse_bps2'],
                        rmse_reduction_from_feature=without['rmse_bps']-present['rmse_bps'],
                        mae_reduction_from_feature=without['mae_bps']-present['mae_bps']))
    # Paired validation contrasts use exactly matching rows/cutoffs and fixed recipes.
    v = pd.DataFrame(validation)
    v_effects = []
    for (issue, side, family, offset), group in v.groupby(['issue', 'side', 'family', 'offset_months']):
        by_set = group.set_index('feature_set')
        for c in CONTRASTS:
            a, b = by_set.loc[c['without']], by_set.loc[c['with_feature']]
            base.check(a['rows'] == b['rows'], 'Feature contrast validation rows differ')
            v_effects.append(dict(issue=issue, side=side, family=family, offset_months=offset, **c,
                rows=int(a['rows']), mse_reduction_from_feature=float(a.mse_bps2-b.mse_bps2),
                rmse_reduction_from_feature=float(a.rmse_bps-b.rmse_bps)))
    e = pd.DataFrame(effects)
    per_fold = e[e.period.ne('ALL')]
    stability = per_fold.groupby(['side', 'family', 'feature', 'experiment']).agg(
        folds=('period', 'size'), folds_with_lower_mse=('mse_reduction_from_feature', lambda s: int((s > 0).sum()))).reset_index()
    pooled = e[e.period.eq('ALL')].merge(stability, on=['side', 'family', 'feature', 'experiment'], validate='one_to_one')
    table = pd.DataFrame(metrics)
    for name, data in [('magnitude_metrics', table), ('feature_effects_by_fold', e),
                       ('feature_effects_pooled', pooled), ('feature_effects_validation', pd.DataFrame(v_effects)),
                       ('tail_errors', pd.DataFrame(tail_rows)), ('calibration', pd.DataFrame(calibration)),
                       ('training_diagnostics', pd.DataFrame(diagnostics))]:
        data.to_csv(out / f'{name}.csv', index=False)
    primary = ['selected_mean_bps', 'training_mean_bps', 'previous_best_bps', 'previous_ensemble_bps']
    primary += [f'{family}_selected' for family in models]
    summary = ('CONTROLLED CONDITIONAL MEAN MAGNITUDES - fixed expanding history\n'
        'No signed-return forecasts. Selection uses only earlier validation MSE.\n'
        'Feature effects: positive reduction means inclusion helped, conditional on the compared feature set/model.\n'
        'Feature effects are historical/descriptive, not causal or independently confirmed; no automatic test-based pruning.\n\n'
        + table[table.period.eq('ALL') & table.model.isin(primary)].to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    base.dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(result), folds=4,
        training_performed=True, signed_return_forecasts=False, model_replay_passed=True,
        contract_sha256=base.sha(out / 'CONTRACT.json'), predictions_sha256=base.sha(out / 'magnitude_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    main()
