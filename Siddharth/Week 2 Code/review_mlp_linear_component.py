"""Read-only UP MLP ablation. No fitting, training, tuning, or input changes.

Save beside fit_mlp_regularization_mae.py in Week 2/Code.
Reads the completed mlp_regularization_mae_v1 artifacts and Week 1 prepared data.
Compares the full original MLP against its OWN frozen median-linear component.
The linear-only forecast keeps the original nonnegative output clamp.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits
import fit_mlp_regularization_mae as recipe


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--source', type=Path, help='Completed regularization experiment folder')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    workspace = args.workspace.resolve()
    source = args.source or workspace/'Code/experiments/mlp_regularization_mae_v1'
    out = args.out or workspace/'Code/experiments/mlp_linear_component_review_v1'
    check(not out.exists(), f'Preserving existing output {out}; use --out with a new folder')
    check((source/'COMPLETE.json').exists(), 'Regularization run must be complete first')
    complete = json.loads((source/'COMPLETE.json').read_text())
    check(complete['status'] == 'COMPLETED', 'Source run is incomplete')
    x, y, dev, ref = joblib.load(workspace.parent/'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing, 'Invalid input index')
    check(dev.equals(ref.index) and len(dev) == 18481, 'Unexpected evaluation universe')
    check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Unexpected outcome horizon')
    x = x[recipe.COLS].replace([np.inf, -np.inf], np.nan)
    saved = pd.read_csv(source/'predictions.csv', index_col='decision_time')
    saved.index = pd.to_datetime(saved.index, utc=True)
    check(saved.index.equals(dev.sort_values()), 'Saved prediction timestamps mismatch')
    signed = y.signed_return_log_bps.astype(float)
    target = pd.Series(np.abs(10000*np.expm1(signed/10000)), index=y.index)
    side = pd.Series(np.where(signed > 0, 'UP', np.where(signed < 0, 'DOWN', 'FLAT')), index=y.index)
    np.testing.assert_allclose(saved.actual_magnitude_bps, target.loc[saved.index], rtol=0, atol=1e-8)
    check(np.array_equal(saved.actual_direction, side.loc[saved.index]), 'Saved direction mismatch')
    check(int(side.loc[dev].eq('UP').sum()) == 10191, 'Unexpected UP row count')
    issues = pd.to_datetime(saved.issue, utc=True)
    check(set(issues.unique()) == set(recipe.STARTS), 'Unexpected test folds')
    frames, audit = [], []
    print('Reading saved UP MLP fits only. No fitting or training.', flush=True)
    for issue in recipe.STARTS:
        keys = saved.index[issues.eq(issue)]
        path = source/f'{issue:%Y-%m}_UP_original.joblib'
        art = joblib.load(path)
        check(pd.Timestamp(art['issue']) == issue, 'Artifact issue mismatch')
        expected_train = y.index[(y.label_end < issue) & side.eq('UP')]
        check(art['outer_train_keys'].equals(expected_train), 'Outer training population mismatch')
        check(y.loc[expected_train, 'label_end'].max() < issue, 'Training label cutoff violation')
        check([a['seed'] for a in art['outer']] == recipe.SEEDS, 'Unexpected seed list')
        z = recipe.design(x, keys, art['outer_prep'])
        full, linear = [], []
        for a in art['outer']:
            check(a['setting'] == 'original', 'Wrong setting in artifact')
            model = recipe.restore(a)
            full.append(recipe.predict(model, z))
            # Remove only the neural contribution; use the saved linear weights.
            with torch.no_grad():
                linear.append(np.concatenate([
                    model.linear(z[i:i+4096]).squeeze(1).clamp(min=0).numpy()*100.
                    for i in range(0, len(z), 4096)
                ]).astype(float))
        full = np.mean(full, axis=0)
        linear = np.mean(linear, axis=0)
        np.testing.assert_allclose(full, saved.loc[keys, 'UP_original'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(full, saved.loc[keys, 'UP_existing_mlp'], rtol=0, atol=.001)
        f = pd.DataFrame({'issue': str(issue), 'actual_direction': side.loc[keys],
                          'actual_bps': target.loc[keys], 'full_mlp': full,
                          'linear_only': linear}, index=keys)
        for base in ['mean', 'median']:
            f[f'baseline_{base}'] = saved.loc[keys, f'UP_baseline_{base}']
            expected = getattr(target.loc[expected_train], base)()
            np.testing.assert_allclose(f[f'baseline_{base}'], expected, rtol=0, atol=1e-8)
        f['adjustment_bps'] = f.full_mlp - f.linear_only
        f['absolute_error_gain_bps'] = (f.actual_bps-f.linear_only).abs() - (f.actual_bps-f.full_mlp).abs()
        # Only matching realized UP outcomes are scored, as in all prior runs.
        f = f.loc[f.actual_direction.eq('UP')].copy()
        frames.append(f)
        audit.append({'issue': str(issue), 'selected_epochs_by_seed': [a['epochs'] for a in art['outer']],
                      'max_replay_error_bps': float(np.max(np.abs(full-saved.loc[keys, 'UP_original']))),
                      'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'Replayed {issue.date()}: {len(f)} UP rows; epochs {audit[-1]["selected_epochs_by_seed"]}', flush=True)
    pred = pd.concat(frames).sort_index()
    check(pred.index.equals(dev.sort_values()[side.loc[dev.sort_values()].eq('UP')]), 'Final UP timestamp mismatch')
    rows, contributions = [], []
    for issue, f in [('POOLED', pred)] + list(pred.groupby('issue')):
        scores = {name: recipe.metrics(f.actual_bps, f[name]) for name in
                  ['full_mlp', 'linear_only', 'baseline_mean', 'baseline_median']}
        for name, m in scores.items():
            rows.append({'issue': issue, 'side': 'UP', 'model': name, **m,
                         'mae_gain_vs_mean_bps': scores['baseline_mean']['mae_bps']-m['mae_bps'],
                         'mae_gain_vs_median_bps': scores['baseline_median']['mae_bps']-m['mae_bps']})
        contributions.append({'issue': issue, 'rows': len(f),
            'mae_gain_from_neural_bps': scores['linear_only']['mae_bps']-scores['full_mlp']['mae_bps'],
            'linear_rank_ic': scores['linear_only']['rank_ic'], 'full_rank_ic': scores['full_mlp']['rank_ic'],
            'mean_abs_forecast_adjustment_bps': float(f.adjustment_bps.abs().mean()),
            'p90_abs_forecast_adjustment_bps': float(f.adjustment_bps.abs().quantile(.9)),
            'fraction_rows_error_improved': float((f.absolute_error_gain_bps > 1e-8).mean()),
            'fraction_rows_error_worsened': float((f.absolute_error_gain_bps < -1e-8).mean())})
    scores = pd.DataFrame(rows)
    contributions = pd.DataFrame(contributions)
    summary = ('READ-ONLY UP MLP COMPONENT REVIEW\n'
        'Positive mae_gain_from_neural_bps means full MLP beats its own frozen median-linear component.\n'
        'Linear-only retains the nonnegative output clamp; adjustment is the final forecast difference.\n'
        'This is median regression, not the Ridge component from the earlier MSE experiment.\n'
        'Pooled baseline IC is not within-fold ranking. Constant within-fold IC is undefined.\n'
        'Historical exposed results; no fitting, selection, confidence intervals, or significance claims.\n\n'
        + scores[scores.issue.eq('POOLED')].to_string(index=False)
        + '\n\nNEURAL CONTRIBUTION BY FOLD\n' + contributions.to_string(index=False))
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    pred.to_csv(out/'predictions.csv', index_label='decision_time')
    scores.to_csv(out/'scores.csv', index=False)
    contributions.to_csv(out/'neural_contribution.csv', index=False)
    (out/'replay_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETED','up_rows':len(pred),'replay_checks':4}), encoding='utf-8')
    print(summary)
    print(f'Outputs: {out}')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        main()
