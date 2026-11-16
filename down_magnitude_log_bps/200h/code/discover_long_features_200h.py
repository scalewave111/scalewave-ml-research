"""User-run fresh 200h feature experiment. --check-inputs never fits models."""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import improve_excursion_features_200h as prior
import long_horizon_features_200h as features

base = prior.base
REFERENCE = base.HERE/'experiments/excursion_pipeline_200h_v1/20261113T115534387046Z/price_followup/20261113T121536996657Z'


def baseline_gate(selection, y, folds):
    weights = [len(va) for _, va in folds]
    means = np.array([np.abs(y[va]-np.mean(y[tr])).mean() for tr,va in folds])
    medians = np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr,va in folds])
    candidate = np.array(selection['validation_window_mae'])
    accepted = bool(selection['features']) and all(
        np.average(candidate, weights=weights) < .999*np.average(b, weights=weights)
        and (candidate < b).sum() >= 2 for b in [means, medians])
    fallback = 'mean' if np.average(means,weights=weights) < np.average(medians,weights=weights) else 'median'
    return dict(accepted=bool(accepted), fallback=fallback, mean_window_mae=means.tolist(),
                median_window_mae=medians.tolist())


def load(args):
    x, y, dev, files = base.load(args)
    excluded = x.attrs.get('excluded_labels')
    x.index, y.index, dev = x.index.as_unit('ns'), y.index.as_unit('ns'), dev.as_unit('ns')
    # Aggregate market activity on the original input inventory, independently of Y validity.
    market = pd.concat([base.read_x(args.data_dir/n) for n in ['X_train.parquet','X_development.parquet']]).sort_index()
    market.index = market.index.as_unit('ns')
    r = prior.canonical.load_returns(args.returns)
    new, definitions = features.make_features(r, x.index, market)
    cut = x.index[len(x)//2]
    keys = pd.DatetimeIndex([cut])
    expected = features.make_features(r, keys, market)[0]
    altered = r.copy()
    altered.loc[altered.index >= cut] = 100000.
    changed = market.copy()
    changed.loc[changed.index >= cut] = 99999.
    pd.testing.assert_frame_equal(expected, features.make_features(altered, keys, changed)[0])
    pd.testing.assert_frame_equal(expected, features.make_features(r.loc[:cut], keys, market.loc[:cut])[0])
    ref = args.reference_run
    done = json.loads((ref/'COMPLETE.json').read_text())
    contract = json.loads((ref/'CONTRACT.json').read_text())
    base.check(done['status']=='COMPLETE' and done['predictions_sha256']==base.sha(ref/'predictions.csv'), 'Reference integrity failed')
    base.check(contract['horizon_minutes']==12000, 'Wrong reference horizon')
    for path in files+[args.returns]:
        base.check(contract['source_hashes'][str(path)]==base.sha(path), 'Reference input changed: '+str(path))
    old = pd.read_csv(ref/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time, utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue, utc=True)
    base.check(len(old)==2*len(dev) and not old.duplicated(['decision_time','target']).any(), 'Reference coverage mismatch')
    for target in base.TARGETS:
        f = old[old.target.eq(target)].set_index('decision_time').sort_index()
        base.check(f.index.equals(dev), 'Reference keys mismatch')
        np.testing.assert_allclose(f.actual, y.loc[dev,target], rtol=0, atol=1e-10)
        for i,issue in enumerate(base.ISSUES):
            end=base.ISSUES[i+1] if i+1<len(base.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            sub=f[(f.index>=issue)&(f.index<end)]
            train=y.loc[y.index+pd.Timedelta(hours=200)<issue,target]
            base.check(sub.issue.eq(issue).all(), 'Reference issue mismatch')
            np.testing.assert_allclose(sub.training_mean,train.mean(),rtol=0,atol=1e-8)
            np.testing.assert_allclose(sub.training_median,train.median(),rtol=0,atol=1e-8)
    files += [args.returns, ref/'predictions.csv', ref/'COMPLETE.json', ref/'CONTRACT.json',
              Path(__file__), Path(features.__file__), Path(base.__file__), Path(prior.__file__), Path(prior.canonical.__file__)]
    return new, y, dev, old, definitions, excluded, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=base.DATA)
    p.add_argument('--y-dir', type=Path, default=base.CANONICAL/'y/horizon_minutes=12000/symbol=BTCUSDT')
    p.add_argument('--returns', type=Path, default=prior.WEEK1/'Dataset/canonical_returns_5m.parquet')
    p.add_argument('--reference-run', type=Path, default=REFERENCE)
    p.add_argument('--output-dir', type=Path, default=base.HERE/'experiments/long_features_200h_v1')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args()
    base.check(args.threads>0, 'Threads must be positive')
    args.shortlist, args.max_features, args.screen_rows, args.min_gain = 20, 8, 4096, .001
    with threadpool_limits(limits=args.threads):
        x,y,dev,old,definitions,excluded,files = load(args)
        folds = {issue:base.windows(x.index,issue) for issue in base.ISSUES}
        print(f'Input audit passed: {len(x)} rows, {len(dev)} evaluation origins, {x.shape[1]} new candidates', flush=True)
        if args.check_inputs:
            for issue,windows in folds.items():
                eligible=set(x.columns)
                for tr,_ in windows:
                    block=x.iloc[tr]
                    eligible &= set(block.columns[(block.notna().mean()>=.8)&(block.notna().sum()>=1000)&(block.nunique()>1)])
                print(f'{issue:%Y-%m}: {len(eligible)} candidates pass earlier-training quality; {sum(c.startswith("LONGMARKET") for c in eligible)} market summaries',flush=True)
            return
        out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        out.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(definitions).to_csv(out/'feature_definitions.csv',index=False)
        if excluded is not None:
            excluded.to_csv(out/'excluded_labels.csv',index=False)
        guidance = [base.HERE.parent/'PROJECT_CONTEXT.md'] + [Path('C:/Users/Siddh/Desktop/ML/ml_framework')/n for n in ['AGENTS.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md']]
        base.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=12000,
            features='96 new causal price features +20 market history summaries; no old selected feature reserved',
            selection='Independent UP/DOWN; earlier 3 purged quarters; fresh greedy Ridge/MAE-HGB search; shortlist20,max8; candidate must beat mean and median MAE by .1% and 2/3 windows; otherwise earlier-chosen mean/median',
            price_hours=features.HOURS,market_hours=[168,720],
            limitations=['Historical exposed evaluation, adaptive validation reuse; not independent confirmation',
                        'Long overlap reduces effective sample size; no guaranteed baseline win',
                        'Original F4Z point-in-time availability and funding limitations inherited',
                        'Price paths use canonical 5m endpoints; targets use canonical 1m paths',
                        'Old predictions compared as saved; old models not independently replayed here',
                        'No automatic resume; no external macro or on-chain inputs'],
            input_hashes={str(q):base.sha(q) for q in files},guidance_hashes={str(q):base.sha(q) for q in guidance},
            versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=base.sklearn.__version__)))
        for q in [Path(__file__),Path(features.__file__),Path(base.__file__),Path(prior.__file__),Path(prior.canonical.__file__)]:
            (out/q.name).write_bytes(q.read_bytes())
        frames, selections, ledger = [],[],[]
        for i,issue in enumerate(base.ISSUES):
            end = base.ISSUES[i+1] if i+1<len(base.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te = x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            tr = np.flatnonzero(x.index+pd.Timedelta(hours=200)<issue)
            for target in base.TARGETS:
                print(f'{issue:%Y-%m} {target}: fresh long-feature search',flush=True)
                folder=out/f'{issue:%Y-%m}_{target}'
                folder.mkdir()
                truth=y[target].to_numpy(float)
                choice=base.search(x,truth,folds[issue],args,folder)
                policy=baseline_gate(choice,truth,folds[issue])
                cols=choice['features']
                model=base.fit(x.iloc[tr][cols].to_numpy(float),truth[tr],choice['model'])
                raw=base.predict(model,x.iloc[te][cols].to_numpy(float))
                constant=float(np.mean(truth[tr]) if policy['fallback']=='mean' else np.median(truth[tr]))
                selected=raw if policy['accepted'] else np.full(len(te),constant)
                bundle=dict(model=model,features=cols,policy=policy,constant=constant,issue=issue,target=target)
                joblib.dump(bundle,folder/'model.joblib')
                loaded=joblib.load(folder/'model.joblib')
                replay=base.predict(loaded['model'],x.iloc[te][loaded['features']].to_numpy(float))
                np.testing.assert_allclose(raw,replay,rtol=0,atol=1e-10)
                np.testing.assert_allclose(selected,replay if loaded['policy']['accepted'] else np.full(len(te),loaded['constant']),rtol=0,atol=1e-10)
                base.dump(folder/'MODEL_HASH.json',dict(sha256=base.sha(folder/'model.joblib')))
                reference=old[old.target.eq(target)].set_index('decision_time').loc[x.index[te]]
                base.check(reference.issue.eq(issue).all(),'Reference issue mismatch')
                np.testing.assert_allclose(reference.training_mean,np.mean(truth[tr]),atol=1e-8,rtol=0)
                np.testing.assert_allclose(reference.training_median,np.median(truth[tr]),atol=1e-8,rtol=0)
                frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=target,actual=truth[te],
                    candidate=raw,selected=selected,previous=reference.selected.to_numpy(),
                    training_mean=np.mean(truth[tr]),training_median=np.median(truth[tr]))))
                selections.append(dict(issue=str(issue),target=target,search=choice,policy=policy))
                base.dump(folder/'policy.json',policy)
                for label,train,valid in [('outer',tr,te)]+[(f'inner{k+1}',a,b) for k,(a,b) in enumerate(folds[issue])]:
                    ledger.append(dict(issue=issue,target=target,window=label,training_rows=len(train),evaluation_rows=len(valid),
                        training_label_end_max=x.index[train].max()+pd.Timedelta(hours=200),validation_start=x.index[valid].min(),
                        evaluation_label_end_max=x.index[valid].max()+pd.Timedelta(hours=200)))
        pred=pd.concat(frames,ignore_index=True)
        base.check(len(pred)==2*len(dev) and not pred.duplicated(['decision_time','target']).any(),'Output coverage failed')
        pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
        pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
        pred.to_csv(out/'predictions.csv',index=False)
        scores=[]
        for target,g in pred.groupby('target'):
            for period,f in [('ALL',g)]+list(g.groupby('period')):
                for name in ['candidate','selected','previous','training_mean','training_median']:
                    scores.append(dict(target=target,period=period,model=name,**base.metric(f.actual.to_numpy(),f[name].to_numpy())))
        metrics=pd.DataFrame(scores)
        metrics.to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
        base.dump(out/'selections.json',selections)
        summary=metrics[metrics.period.eq('ALL')].to_string(index=False)
        (out/'SUMMARY.txt').write_text(summary+'\nAll errors log bps. Historical exposed evaluation. Candidate may equal median if search fails.\n',encoding='utf-8')
        base.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),replay_passed=True,predictions_sha256=base.sha(out/'predictions.csv')))
        print(summary)
        print('Saved:',out)


if __name__=='__main__':
    main()
