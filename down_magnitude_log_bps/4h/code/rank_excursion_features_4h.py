"""Saved-model block-permutation reliance, October models on Oct2024-Jan2025."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import improve_excursion_features_4h as follow
base = follow.base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=base.HERE/'experiments/excursion_price_followup_4h_v1/20261113T093511890828Z')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    base.check(args.repeats > 0, 'Positive repeats required')
    done = json.loads((args.run_dir/'COMPLETE.json').read_text())
    base.check(done['status']=='COMPLETE' and done['predictions_sha256']==base.sha(args.run_dir/'predictions.csv'), 'Prediction hash mismatch')
    cfg = argparse.Namespace(data_dir=base.DATA, y_dir=base.CANONICAL/'y/horizon_minutes=240/symbol=BTCUSDT',
        returns=follow.WEEK1/'Dataset/canonical_returns_5m.parquet', reference_run=follow.DEFAULT_REFERENCE)
    x, y, dev, old, anchors, groups, files = follow.inputs(cfg)
    pred = pd.read_csv(args.run_dir/'predictions.csv')
    pred['decision_time'] = pd.to_datetime(pred.decision_time, utc=True).dt.as_unit('ns')
    pred['issue'] = pd.to_datetime(pred.issue, utc=True)
    issue = pd.Timestamp('2024-10-01', tz='UTC')
    out = args.run_dir/'feature_reliance'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    rows, draws, audits = [], [], []
    with threadpool_limits(limits=2):
        for target in base.TARGETS:
            folder = args.run_dir/f'{issue:%Y-%m}_{target}'
            plan = json.loads((folder/'selection.json').read_text())
            spec = plan['variants'][plan['selected_variant']]
            artifact = folder/spec['artifact']
            base.check(base.sha(artifact)==spec['artifact_sha256'], 'Model hash mismatch')
            bundle = joblib.load(artifact)
            cols = spec['features']
            base.check(cols==bundle['features'], 'Feature order mismatch')
            f = pred[pred.target.eq(target) & pred.issue.eq(issue)].set_index('decision_time').sort_index()
            a = x.loc[f.index, cols].to_numpy(float)
            truth = y.loc[f.index, target].to_numpy()
            np.testing.assert_allclose(truth, f.actual, rtol=0, atol=1e-10)
            model = bundle['model']
            original = base.predict(model, a)
            np.testing.assert_allclose(original, f.selected, rtol=0, atol=1e-7)
            loss = np.abs(original-truth)
            audits.append(dict(target=target, rows=len(f), features=len(cols), model_sha256=base.sha(artifact),
                baseline_mae=float(loss.mean()), start=str(f.index.min()), end=str(f.index.max())))
            # Blocks of 48 consecutive forecast rows (~24h), preserving within-block order.
            # Same permutations for all columns; any short final block remains a block.
            blocks = [np.arange(start, min(start+48, len(f))) for start in range(0, len(f), 48)]
            perms = [np.concatenate([blocks[i] for i in np.random.default_rng(48+r).permutation(len(blocks))]) for r in range(args.repeats)]
            sets = [('individual', c, [j]) for j, c in enumerate(cols)]
            sets += [('group', 'original_F4Z', [j for j,c in enumerate(cols) if c.startswith('F4Z__')])]
            sets += [('group', g, [j for j,c in enumerate(cols) if c in names]) for g,names in groups.items()]
            for kind, name, indices in sets:
                if not indices:
                    continue
                changes = []
                for r, perm in enumerate(perms):
                    altered = a.copy()
                    altered[:, indices] = a[np.ix_(perm, indices)]
                    delta = np.abs(base.predict(model, altered)-truth)-loss
                    changes.append(float(delta.mean()))
                    draws.append(dict(target=target, kind=kind, feature=name, repeat=r,
                                      mae_increase=float(delta.mean())))
                rows.append(dict(target=target, kind=kind, feature=name, columns=len(indices),
                    mean_mae_increase=float(np.mean(changes)), shuffle_sd=float(np.std(changes, ddof=1)) if len(changes)>1 else 0.,
                    positive_repeats=int((np.array(changes)>0).sum()), repeats=args.repeats))
                print(f'{target}: {kind} {name}: {np.mean(changes):.5f}', flush=True)
    result = pd.DataFrame(rows).sort_values(['target','kind','mean_mae_increase'], ascending=[True,True,False])
    result['rank'] = result.groupby(['target','kind']).cumcount()+1
    result.to_csv(out/'feature_ranking.csv', index=False)
    pd.DataFrame(draws).to_csv(out/'permutation_draws.csv', index=False)
    base.dump(out/'AUDIT.json', dict(models=audits, method='Permute raw feature values including NaN in48-row blocks; jointly recompute missing indicators;5 fixed-seed repetitions default',
        repeats=args.repeats, source_sha256=base.sha(Path(__file__)), prediction_sha256=base.sha(args.run_dir/'predictions.csv'),
        interpretation='Model reliance on October-Jan historical evaluation, not causal importance or additive contribution; correlated inputs and off-distribution combinations affect rankings; shuffle SD is not a confidence interval',
        training_performed=False, replay_passed=True))
    (out/'SUMMARY.txt').write_text(result.to_string(index=False), encoding='utf-8')
    base.dump(out/'COMPLETE.json', dict(status='COMPLETE', training_performed=False, ranking_sha256=base.sha(out/'feature_ranking.csv')))
    print('Saved:', out)


if __name__=='__main__':
    main()
