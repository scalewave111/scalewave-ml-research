r"""Compare learned UP/DOWN conditional-mean MLP and RF models. NEW TRAINING.

Run: python .\Code\fit_mlp_rf_conditional_means_48h.py
Uses existing mean MLP architecture, fixed RF architecture, 4/5/8 feature packs.
Both side forecasts are made for every timestamp; no outcome-based routing.
"""
from pathlib import Path
import argparse
import gc
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

import fit_conditional_48h_mean_features as mlp
import fit_conditional_means_signed_48h as common

HERE = Path(__file__).resolve().parent
EXP = HERE / 'experiments'
PACKS = {k: list(v) for k, v in mlp.PACKS.items()}
SEEDS = list(mlp.SEEDS)


def rf_fit(x, target, keys, columns):
    # Same RF capacity as the established recipe; squared-error leaf means.
    a = x.loc[keys, columns]
    median = a.median().fillna(0.)
    filled = a.fillna(median)
    p = dict(columns=columns, median=median, center=filled.mean(),
             scale=filled.std(ddof=0).replace(0, 1.).fillna(1.))
    common.check(len(keys) >= 4096, 'RF needs at least 4096 eligible training rows')
    model = RandomForestRegressor(n_estimators=128, criterion='squared_error',
        max_depth=3, min_samples_leaf=128, max_features=1., bootstrap=True,
        max_samples=4096, n_jobs=1, random_state=48)
    model.fit(common.design(x.loc[keys], p), target.loc[keys].to_numpy(float))
    return dict(model=model, prep=p)


def predict_artifact(artifact, x, keys):
    """Only feature history/origins required, not future direction or returns."""
    common.check(len(keys) > 0 and keys.isin(x.index).all()
                 and (keys >= pd.Timestamp(artifact['issue'])).all(),
                 'Missing feature origins or model issuance after origin')
    if artifact['family'] == 'mlp':
        result = np.mean([mlp.predict_mlp(a, x, keys) for a in artifact['models']], axis=0)
    else:
        a = artifact['model']
        result = a['model'].predict(common.design(x.loc[keys], a['prep']))
    common.check(np.isfinite(result).all() and (result >= 0).all(), 'Invalid mean-magnitude predictions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=common.WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    p.add_argument('--direction-run', type=Path, default=EXP / 'direction_v45_style_v1/20261112T045009318177Z')
    p.add_argument('--reference-run', type=Path, default=EXP / 'conditional_means_signed_48h_v1/20261112T063535834150Z')
    p.add_argument('--output-dir', type=Path, default=EXP / 'mlp_rf_conditional_means_48h_v1')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    direction_path = args.direction_run / 'development_predictions.csv'
    reference_path = args.reference_run / 'signed_return_predictions.csv'
    direction_done = json.loads((args.direction_run / 'COMPLETE.json').read_text())
    reference_done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    common.check(direction_done.get('status') == reference_done.get('status') == 'COMPLETE',
                 'Direction/reference runs must be complete')
    common.check(direction_done['artifacts']['development_predictions.csv'] == common.sha(direction_path)
                 and reference_done['prediction_sha256'] == common.sha(reference_path), 'Input hash mismatch')
    direction, reference = common.read_csv(direction_path), common.read_csv(reference_path)
    pipeline_path = args.direction_run / 'final/pipeline.joblib'
    hashes = {k.replace('\\', '/'): v for k, v in direction_done['artifacts'].items()}
    common.check(hashes.get('final/pipeline.joblib') == common.sha(pipeline_path), 'Direction artifact hash mismatch')
    direction_cutoff = pd.Timestamp(joblib.load(pipeline_path)['max_label_end'])
    common.check(direction_cutoff < direction.index.min(), 'Direction training overlaps forecast origins')
    x, y, dev, _ = joblib.load(args.prepared)
    common.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing
                 and x.columns.is_unique and x.index.tz is not None, 'Invalid prepared feature/label indexes')
    dev = pd.DatetimeIndex(dev).sort_values()
    common.check(len(dev) == 18481 and dev.equals(direction.index) and dev.equals(reference.index)
                 and dev.isin(y.index).all(), 'Exact 18,481-row alignment required; no row dropping')
    x = x[common.COLS].replace([np.inf, -np.inf], np.nan)
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    common.check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Expected 48-hour endpoints')
    signed = pd.Series(10000. * np.expm1(y.signed_return_log_bps.to_numpy(float) / 10000.), index=y.index)
    target = signed.abs()
    common.check(np.isfinite(signed).all() and (signed.loc[dev] != 0).all(), 'Invalid/flat development outcomes')
    np.testing.assert_allclose(signed.loc[dev], reference.actual_signed_return_bps, rtol=0, atol=1e-7)
    common.check(np.array_equal(direction.y.to_numpy(), (signed.loc[dev] > 0).astype(int).to_numpy()),
                 'Direction labels disagree with magnitude labels')
    common.check(np.isfinite(direction[['p_up', 'p_down']]).all().all()
                 and direction.p_up.between(0, 1).all() and direction.p_down.between(0, 1).all(),
                 'Invalid direction probabilities')
    sum_error = float((direction.p_up + direction.p_down - 1.).abs().max())
    common.check(sum_error <= 2 * np.finfo(np.float32).eps, 'Probability complements disagree')
    direction['p_up'] = direction.p_up.astype(float)
    direction['p_down'] = 1. - direction.p_up
    np.testing.assert_allclose(direction[['p_up', 'p_down']], reference[['p_up', 'p_down']], rtol=0, atol=1e-7)
    reference_issue = pd.to_datetime(reference.issue, utc=True)

    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(mlp.__file__), Path(common.__file__)]
    for source in sources:
        (out / ('source_' + source.name)).write_bytes(source.read_bytes())
    common.dump(out / 'CONTRACT.json', dict(
        task='Learn separate UP/DOWN conditional mean magnitude models and combine with frozen direction',
        units='ordinary bps, percent=bps/100; both conditional magnitude outputs positive',
        target='abs(10000*expm1(signed_return_log_bps/10000)); positive/negative historical sides',
        packs=PACKS, seeds=SEEDS, horizon_hours=48, symbol='BTCUSDT',
        mlp='Ridge alpha1 frozen skip + 8/4 SiLU correction, dropout .05, AdamW .001, MSE, 3 seeds; up to48 epochs, patience8',
        rf='128 trees, squared_error, depth3, leaf128, bootstrap4096, seed48; standalone RF, not hybrid',
        selection='Per side/fold, earlier validation MSE selects epochs and 4/5/8 pack for each family; then selects MLP versus RF. Tie: fewer features, then name.',
        validation='First two months after issue minus3 months; training label_end strictly before inner cutoff; validation label_end before issue',
        outer='Four quarterly expanding fits; all training label_end strictly before issue',
        constant_fallback=False, baseline_role='Benchmarks only; not candidates for model selection',
        direction_run=str(args.direction_run), direction_max_training_label_end=direction_cutoff,
        reference_run=str(args.reference_run), direction_retrained=False,
        input_hashes={str(args.prepared): common.sha(args.prepared), str(direction_path): common.sha(direction_path),
                      str(reference_path): common.sha(reference_path)},
        source_hashes={str(s): common.sha(s) for s in sources},
        probability_sum_error_before_correction=sum_error,
        limitation='New experiment on historically exposed data. Mean objective does not guarantee accuracy/calibration. No test-based automatic promotion or significance claim.'))
    frames, validations, selections, audits = [], [], [], []
    for issue in common.STARTS:
        inner = issue - pd.DateOffset(months=3)
        end = issue + pd.DateOffset(months=3) if issue != common.STARTS[-1] else dev.max() + pd.Timedelta(minutes=30)
        te = dev[(dev >= issue) & (dev < end)]
        common.check(len(te) > 0 and reference_issue.loc[te].eq(issue).all(), 'Reference folds disagree')
        outer = y.index[y.label_end < issue]
        f = direction.loc[te, ['p_up', 'p_down']].copy()
        f['issue'], f['symbol'], f['horizon_hours'] = issue, 'BTCUSDT', 48
        f['forecast_end'] = te + pd.Timedelta(hours=48)
        f['actual_signed_return_bps'] = signed.loc[te]
        f['training_signed_mean_bps'] = signed.loc[outer].mean()
        for side in ['UP', 'DOWN']:
            mask = signed > 0 if side == 'UP' else signed < 0
            ti = y.index[(y.label_end < inner) & mask]
            va = y.index[(y.index >= inner) & (y.index < issue - pd.DateOffset(months=1))
                         & (y.label_end < issue) & mask]
            tr = y.index[(y.label_end < issue) & mask]
            common.check(min(len(ti), len(va), len(tr)) >= 100, 'Insufficient side-specific rows')
            f[f'{side}_training_mean_bps'] = target.loc[tr].mean()
            source_base = 'up_train_mean_bps' if side == 'UP' else 'down_train_mean_bps'
            np.testing.assert_allclose(f[f'{side}_training_mean_bps'], reference.loc[te, source_base], rtol=0, atol=1e-7)
            trials = []
            for pack, columns in PACKS.items():
                print(f'START {issue.date()} {side} {pack}: MLP and RF means', flush=True)
                mlp.COLS = list(columns)
                inner_models = []
                for seed in SEEDS:
                    model = mlp.fit_mlp(x, target, ti, inner, seed, valid=va)
                    inner_models.append(model)
                    print(f'  MLP seed {seed}: selected {model["epoch"]} epochs', flush=True)
                vp = np.mean([mlp.predict_mlp(a, x, va) for a in inner_models], axis=0)
                trials.append(dict(issue=str(issue), side=side, family='mlp', pack=pack,
                    feature_count=len(columns), **common.score(target.loc[va], vp)))
                epochs = [a['epoch'] for a in inner_models]
                del inner_models
                models = [mlp.fit_mlp(x, target, tr, issue, seed, epochs=ep) for seed, ep in zip(SEEDS, epochs)]
                artifacts = {'mlp': dict(family='mlp', models=models, issue=issue, side=side, pack=pack,
                    epochs=epochs, max_training_label_end=y.loc[tr, 'label_end'].max())}
                rf_inner = rf_fit(x, target, ti, columns)
                rp = rf_inner['model'].predict(common.design(x.loc[va], rf_inner['prep']))
                trials.append(dict(issue=str(issue), side=side, family='rf', pack=pack,
                    feature_count=len(columns), **common.score(target.loc[va], rp)))
                artifacts['rf'] = dict(family='rf', model=rf_fit(x, target, tr, columns),
                    issue=issue, side=side, pack=pack, max_training_label_end=y.loc[tr, 'label_end'].max())
                for family, artifact in artifacts.items():
                    path = out / f'{issue:%Y-%m}_{side}_{family}_{pack}.joblib'
                    pred = predict_artifact(artifact, x, te)
                    joblib.dump(artifact, path, compress=3)
                    replay = predict_artifact(joblib.load(path), x, te)
                    np.testing.assert_allclose(pred, replay, rtol=0, atol=1e-6)
                    f[f'{side}_{family}_{pack}'] = pred
                    audits.append(dict(issue=issue, side=side, family=family, pack=pack,
                        training_rows=len(tr), max_training_label_end=artifact['max_training_label_end'],
                        replay_passed=True, artifact=path.name, sha256=common.sha(path)))
                del artifacts, models, model, artifact, rf_inner
                gc.collect()
            validations.extend(trials)
            def order(r):
                return r['mse_bps2'], r['feature_count'], r['family'], r['pack']
            for family in ['mlp', 'rf', 'selected']:
                winner = min(trials if family == 'selected' else [r for r in trials if r['family'] == family], key=order)
                column = f'{side}_{winner["family"]}_{winner["pack"]}'
                f[f'{side}_{family}_mean_bps'] = f[column]
                selections.append(dict(issue=issue, side=side, policy=family, family=winner['family'],
                    pack=winner['pack'], validation_mse_bps2=winner['mse_bps2'],
                    model_file=f'{issue:%Y-%m}_{side}_{winner["family"]}_{winner["pack"]}.joblib'))
            f[f'{side}_reference_mean_bps'] = reference.loc[te,
                'up_mean_bps' if side == 'UP' else 'down_mean_magnitude_bps']
        for family in ['mlp', 'rf', 'selected']:
            up, down = f[f'UP_{family}_mean_bps'], f[f'DOWN_{family}_mean_bps']
            f[f'{family}_up_contribution_bps'] = f.p_up * up
            f[f'{family}_down_contribution_bps'] = -f.p_down * down
            f[f'{family}_signed_return_bps'] = f[f'{family}_up_contribution_bps'] + f[f'{family}_down_contribution_bps']
            f[f'{family}_signed_return_pct'] = f[f'{family}_signed_return_bps'] / 100.
        f['reference_signed_return_bps'] = reference.loc[te, 'signed_return_forecast_bps']
        f['side_mean_baseline_bps'] = f.p_up * f.UP_training_mean_bps - f.p_down * f.DOWN_training_mean_bps
        f['zero_baseline_bps'] = 0.
        f['up_mean_bps'], f['down_mean_magnitude_bps'] = f.UP_selected_mean_bps, f.DOWN_selected_mean_bps
        f['up_mean_pct'], f['down_mean_magnitude_pct'] = f.up_mean_bps / 100., f.down_mean_magnitude_bps / 100.
        f['signed_return_forecast_bps'] = f.selected_signed_return_bps
        f['signed_return_forecast_pct'] = f.selected_signed_return_pct
        f['actual_signed_return_pct'] = f.actual_signed_return_bps / 100.
        f.to_csv(out / f'{issue:%Y-%m}_predictions.csv', index_label='decision_time')
        frames.append(f)
        common.dump(out / 'SELECTIONS.json', selections)
        common.dump(out / 'FIT_AUDIT.json', audits)
        pd.DataFrame(validations).to_csv(out / 'validation_scores.csv', index=False)
        print(f'DONE {issue.date()}', flush=True)

    predictions = pd.concat(frames).sort_index()
    common.check(predictions.index.equals(dev), 'Final timestamp set mismatch')
    predictions.to_csv(out / 'signed_return_predictions.csv', index_label='decision_time')
    side_scores, signed_scores = [], []
    for period, f in [('ALL', predictions)] + [(str(k), g) for k, g in predictions.groupby('issue')]:
        for side in ['UP', 'DOWN']:
            g = f.loc[f.actual_signed_return_bps > 0 if side == 'UP' else f.actual_signed_return_bps < 0]
            if g.empty:
                continue
            names = [(f'{family}_{pack}', f'{side}_{family}_{pack}') for family in ['mlp', 'rf'] for pack in PACKS]
            names += [(family, f'{side}_{family}_mean_bps') for family in ['mlp', 'rf', 'selected', 'reference']]
            names += [('training_mean', f'{side}_training_mean_bps')]
            for model, column in names:
                side_scores.append(dict(period=period, side=side, model=model,
                    **common.score(g.actual_signed_return_bps.abs(), g[column])))
        columns = {family: f'{family}_signed_return_bps' for family in ['mlp', 'rf', 'selected', 'reference']}
        columns.update(side_means='side_mean_baseline_bps', training_signed_mean='training_signed_mean_bps', zero='zero_baseline_bps')
        for model, column in columns.items():
            signed_scores.append(dict(period=period, model=model, **common.score(f.actual_signed_return_bps, f[column])))
    side_table, signed_table = pd.DataFrame(side_scores), pd.DataFrame(signed_scores)
    side_table.to_csv(out / 'magnitude_metrics.csv', index=False)
    signed_table.to_csv(out / 'signed_metrics.csv', index=False)
    shown = side_table[side_table.period.eq('ALL') & side_table.model.isin(['mlp', 'rf', 'selected', 'reference', 'training_mean'])]
    summary = ('UP/DOWN CONDITIONAL MEAN MODELS - ordinary bps\n'
        'MLP and RF rows use feature packs chosen using earlier validation MSE.\n'
        'selected chooses a learned family/pack separately per side using earlier validation MSE.\n'
        'reference = previous UP mean GRU / DOWN mean hybrid. training_mean is a benchmark only.\n'
        'No performance guarantee; historically exposed evaluation. Lower RMSE is the primary mean-target metric.\n\n'
        + shown.to_string(index=False) + '\n\nSIGNED RETURN\n'
        + signed_table[signed_table.period.eq('ALL')].to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    common.dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(predictions), folds=4,
        training_performed=True, direction_retrained=False, saved_model_replay_passed=True,
        contract_sha256=common.sha(out / 'CONTRACT.json'), prediction_sha256=common.sha(out / 'signed_return_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
