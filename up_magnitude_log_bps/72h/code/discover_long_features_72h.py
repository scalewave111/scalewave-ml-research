"""User-run 72h excursion search from NEW multiday features only."""
import argparse
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import improve_excursion_features_72h as prior
from fresh_features_72h import build
import fresh_features_72h as feature_module

b = prior.base
NEURAL = b.HERE/'experiments/excursion_neural_72h_v1/20261113T123304519186Z'
REFERENCE = b.HERE/'experiments/excursion_pipeline_72h_v1/20261113T111815030467Z/price_followup/20261113T113500606206Z'


def load(args):
    contract = json.loads((REFERENCE/'CONTRACT.json').read_text())
    done = json.loads((REFERENCE/'COMPLETE.json').read_text())
    b.check(contract['horizon_minutes'] == 4320 and done['status'] == 'COMPLETE', 'Bad reference')
    b.check(b.sha(REFERENCE/'predictions.csv') == done['predictions_sha256'], 'Reference hash changed')
    args.data_dir = Path(contract['arguments']['data_dir'])
    args.y_dir = Path(contract['arguments']['y_dir'])
    original, y, dev, files = b.load(args)
    original.index = original.index.as_unit('ns')
    y.index, dev = y.index.as_unit('ns'), dev.as_unit('ns')
    returns = Path(contract['arguments']['returns'])
    for path in files+[returns]:
        b.check(b.sha(path) == contract['source_hashes'][str(path)], 'Input changed: '+str(path))
    r = prior.canonical.load_returns(returns)
    x, dictionary = build(r, original.index, 72)
    x.attrs['excluded_labels'] = original.attrs['excluded_labels']
    cut = original.index[len(original)//2]
    keys = original.index[original.index <= cut][-10:]
    pd.testing.assert_frame_equal(build(r, keys, 72)[0], build(r[r.index <= cut], keys,72)[0])
    changed = r.copy(); changed.loc[changed.index >= cut] = 1e6
    pd.testing.assert_frame_equal(build(r, keys,72)[0], build(changed,keys,72)[0])
    old = pd.read_csv(REFERENCE/'predictions.csv')
    old['decision_time'] = pd.to_datetime(old.decision_time,utc=True).dt.as_unit('ns')
    old['issue'] = pd.to_datetime(old.issue,utc=True)
    b.check(len(old)==2*len(dev) and not old.duplicated(['decision_time','target']).any(), 'Reference coverage')
    for target in b.TARGETS:
        part = old[old.target.eq(target)].set_index('decision_time').sort_index()
        b.check(part.index.equals(dev), 'Reference timestamps')
        np.testing.assert_allclose(part.actual,y.loc[dev,target],rtol=0,atol=1e-10)
        for i, issue in enumerate(b.ISSUES):
            end = b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            b.check(part.loc[(part.index>=issue)&(part.index<end),'issue'].eq(issue).all(), 'Reference issuance')
    ndone = json.loads((NEURAL/'COMPLETE.json').read_text())
    b.check(ndone['status']=='COMPLETE' and b.sha(NEURAL/'predictions.csv')==ndone['predictions_sha256'], 'Neural reference hash')
    neural = pd.read_csv(NEURAL/'predictions.csv')
    neural['decision_time'] = pd.to_datetime(neural.decision_time,utc=True).dt.as_unit('ns')
    neural['issue'] = pd.to_datetime(neural.issue,utc=True)
    b.check(len(neural)==len(old) and not neural.duplicated(['decision_time','target']).any(), 'Neural coverage')
    keys = ['decision_time','target','issue']
    old = old.merge(neural[keys+['actual','selected']].rename(columns={'actual':'neural_actual','selected':'previous_neural'}),on=keys,validate='one_to_one')
    b.check(len(old)==2*len(dev), 'Neural key alignment')
    np.testing.assert_allclose(old.actual,old.neural_actual,rtol=0,atol=1e-10)
    return x,y,dev,old,dictionary,files+[returns,REFERENCE/'CONTRACT.json',REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv',NEURAL/'COMPLETE.json',NEURAL/'predictions.csv']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/long_features_72h_v1')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--shortlist',type=int,default=20)
    p.add_argument('--max-features',type=int,default=6)
    p.add_argument('--screen-rows',type=int,default=4096)
    p.add_argument('--min-gain',type=float,default=.001)
    args = p.parse_args()
    b.check(args.threads>0 and args.shortlist>=args.max_features>0 and args.screen_rows>=1000 and 0<=args.min_gain<1,'Invalid options')
    x,y,dev,old,dictionary,files = load(args)
    folds = {issue:b.windows(x.index,issue) for issue in b.ISSUES}
    print(f'Checked {len(x)} rows, {len(dev)} evaluation origins, {len(x.columns)} NEW features; causal checks passed.',flush=True)
    if args.check_inputs:
        return
    out = args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    dictionary.to_csv(out/'feature_dictionary.csv',index=False)
    x.notna().mean().rename('coverage_all_rows_diagnostic_only').to_csv(out/'feature_coverage.csv')
    x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv',index=False)
    sources = [Path(__file__),Path(feature_module.__file__),Path(feature_module.shared.__file__),Path(b.__file__),Path(prior.__file__),Path(prior.canonical.__file__)]
    guidance = [b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=4320,targets=b.TARGETS,
        candidate_features=list(x),reference=REFERENCE,neural_reference=NEURAL,lookbacks_hours=[24,36,72,144,168,336,720,1440],
        method='Empty-start feature search; separate UP/DOWN and quarter. Training-only quality, screen, redundancy, Ridge/MAE boosting forward/removal selection versus median. Old forecasts comparison only.',
        availability='Hourly bin labelled t contains 5m return endpoints [t-1h,t); only completed bins used, no future fills. Half-hour origins reuse latest complete hour.',
        missingness='Every hour requires 12 finite returns; every rolling feature requires all hours in its window; missing remains NaN. No invalid target imputation.',
        limitations=['Historically exposed evaluation; adaptive inner search, not fresh confirmation','Longer inputs are hypotheses, not proven predictors','Hourly ranges describe hourly sampled canonical log-price paths, not exchange highs/lows','Bounded univariate screen may miss interaction-only predictors','Overlapping 72h targets are dependent','Old model forecasts are hashed and aligned, not refitted or independently replayed here'],
        input_hashes={str(v):b.sha(v) for v in files},source_hashes={str(v):b.sha(v) for v in sources},
        guidance_hashes={str(v):b.sha(v) for v in guidance},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__)))
    for source in sources:
        (out/source.name).write_bytes(source.read_bytes())
    frames, selections, ledger = [],[],[]
    with threadpool_limits(limits=args.threads):
        for i,issue in enumerate(b.ISSUES):
            end = b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te = x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            tr = np.flatnonzero(x.index+pd.Timedelta(hours=72)<issue)
            for target in b.TARGETS:
                print(f'{issue:%Y-%m} {target}: new-feature selection',flush=True)
                folder = out/f'{issue:%Y-%m}_{target}'; folder.mkdir()
                truth = y[target].to_numpy(float)
                choice = b.search(x,truth,folds[issue],args,folder)
                cols = choice['features']; a = x[cols].to_numpy(float)
                model = b.fit(a[tr],truth[tr],choice['model'])
                prediction = b.predict(model,a[te])
                joblib.dump(dict(model=model,features=cols,target=target,issue=issue),folder/'model.joblib')
                saved = joblib.load(folder/'model.joblib')
                np.testing.assert_allclose(prediction,b.predict(saved['model'],x.iloc[te][saved['features']].to_numpy(float)),rtol=0,atol=1e-10)
                b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
                ref = old[old.target.eq(target)].set_index('decision_time').loc[x.index[te]]
                frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=target,actual=truth[te],
                    new_features=prediction,previous=ref.selected.to_numpy(),previous_neural=ref.previous_neural.to_numpy(),training_median=np.median(truth[tr]),training_mean=np.mean(truth[tr]))))
                selections.append(dict(issue=str(issue),target=target,**choice))
                for label,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                    ledger.append(dict(issue=issue,target=target,window=label,training_rows=len(train),evaluation_rows=len(val),
                        training_label_end_max=x.index[train].max()+pd.Timedelta(hours=72),evaluation_start=x.index[val].min(),
                        evaluation_label_end_max=x.index[val].max()+pd.Timedelta(hours=72)))
    pred = pd.concat(frames,ignore_index=True)
    b.check(len(pred)==2*len(dev) and not pred.duplicated(['decision_time','target']).any(),'Output coverage')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    scores=[]
    for target,g in pred.groupby('target'):
        for period,f in [('ALL',g)]+list(g.groupby('period')):
            for model in ['new_features','previous','previous_neural','training_median','training_mean']:
                scores.append(dict(target=target,period=period,model=model,**b.metric(f.actual.to_numpy(),f[model].to_numpy())))
    metrics=pd.DataFrame(scores); metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    b.dump(out/'selections.json',selections)
    chosen=pd.DataFrame([dict(issue=s['issue'],target=s['target'],feature=c) for s in selections for c in s['features']],columns=['issue','target','feature'])
    chosen.merge(dictionary,on='feature',how='left').to_csv(out/'selected_features.csv',index=False)
    summary='72h NEW multiday-feature search; log bps\n'+metrics[metrics.period.eq('ALL')].to_string(index=False)+'\nHistorical evaluation; no independent confirmation.\n'
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary); print('Saved:',out)


if __name__=='__main__':
    main()
