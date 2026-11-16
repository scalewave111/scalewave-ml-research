"""User-run 72h excursion search from NEW multiday features only."""
import argparse
import compact_trees_72h as compact
import discover_long_features_72h as source
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


LATEST=b.HERE/'experiments/long_features_72h_v1/20261114T074315743848Z'


def load(args):
    x,y,dev,old,dictionary,files=source.load(args)
    done=json.loads((LATEST/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(LATEST/'predictions.csv')==done['predictions_sha256'],'Fresh reference hash')
    latest=pd.read_csv(LATEST/'predictions.csv')
    latest.decision_time=pd.to_datetime(latest.decision_time,utc=True).dt.as_unit('ns')
    latest.issue=pd.to_datetime(latest.issue,utc=True)
    keys=['decision_time','target','issue']
    merged=old.merge(latest[keys+['actual','new_features']].rename(columns={'actual':'fresh_actual','new_features':'previous_fresh'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(merged['_merge'].eq('both').all(),'Fresh reference keys')
    np.testing.assert_allclose(merged.actual,merged.fresh_actual,atol=1e-10,rtol=0)
    return x,y,dev,merged,dictionary,files+[LATEST/'COMPLETE.json',LATEST/'predictions.csv']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/compact_trees_72h_v1')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--shortlist',type=int,default=12)
    p.add_argument('--max-features',type=int,default=3)
    p.add_argument('--screen-rows',type=int,default=4096)
    p.add_argument('--min-gain',type=float,default=.001)
    p.add_argument('--max-window-loss',type=float,default=.02)
    args = p.parse_args()
    b.check(args.threads>0 and args.shortlist>=args.max_features>0 and args.max_features<=3 and 0<=args.max_window_loss<=.1 and args.screen_rows>=1000 and 0<=args.min_gain<1,'Invalid options')
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
    sources = [Path(__file__),Path(compact.__file__),Path(source.__file__),Path(feature_module.__file__),Path(feature_module.shared.__file__),Path(b.__file__),Path(prior.__file__),Path(prior.canonical.__file__)]
    guidance = [b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=4320,targets=b.TARGETS,
        candidate_features=list(x),reference=REFERENCE,neural_reference=NEURAL,fresh_reference=LATEST,tree_configs=compact.CONFIGS,blend_weights=compact.WEIGHTS,lookbacks_hours=[24,36,72,144,168,336,720,1440],
        method='Empty-start feature search; separate UP/DOWN and quarter. Training-only quality, screen, redundancy, compact MAE boosting depth1-3, up to3 fresh features, median blend .25/.5/.75/1, forward selection and per-window loss guard versus median. Old forecasts comparison only.',
        availability='Hourly bin labelled t contains 5m return endpoints [t-1h,t); only completed bins used, no future fills. Half-hour origins reuse latest complete hour.',
        missingness='Every hour requires 12 finite returns; every rolling feature requires all hours in its window; missing remains NaN. No invalid target imputation.',
        limitations=['Historically exposed evaluation; adaptive inner search, not fresh confirmation','Longer inputs are hypotheses, not proven predictors','Hourly ranges describe hourly sampled canonical log-price paths, not exchange highs/lows','Bounded univariate screen may miss interaction-only predictors','Overlapping 72h targets are dependent','Old model forecasts are hashed and aligned, not refitted or independently replayed here'],
        input_hashes={str(v):b.sha(v) for v in files},source_hashes={str(v):b.sha(v) for v in sources},
        guidance_hashes={str(v):b.sha(v) for v in guidance},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__)))
    for source_path in sources:
        (out/source_path.name).write_bytes(source_path.read_bytes())
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
                choice = compact.search(x,truth,folds[issue],args,folder)
                cols = choice['features']; a = x[cols].to_numpy(float)
                model = compact.fit(a[tr],truth[tr],choice['model'])
                prediction = compact.predict(model,a[te])
                joblib.dump(dict(model=model,features=cols,target=target,issue=issue),folder/'model.joblib')
                saved = joblib.load(folder/'model.joblib')
                np.testing.assert_allclose(prediction,compact.predict(saved['model'],x.iloc[te][saved['features']].to_numpy(float)),rtol=0,atol=1e-10)
                b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
                ref = old[old.target.eq(target)].set_index('decision_time').loc[x.index[te]]
                frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=target,actual=truth[te],
                    new_features=prediction,previous_fresh=ref.previous_fresh.to_numpy(),previous=ref.selected.to_numpy(),previous_neural=ref.previous_neural.to_numpy(),training_median=np.median(truth[tr]),training_mean=np.mean(truth[tr]))))
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
            for model in ['new_features','previous_fresh','previous','previous_neural','training_median','training_mean']:
                scores.append(dict(target=target,period=period,model=model,**b.metric(f.actual.to_numpy(),f[model].to_numpy())))
    metrics=pd.DataFrame(scores); metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    b.dump(out/'selections.json',selections)
    chosen=pd.DataFrame([dict(issue=s['issue'],target=s['target'],feature=c) for s in selections for c in s['features']],columns=['issue','target','feature'])
    chosen.merge(dictionary,on='feature',how='left').to_csv(out/'selected_features.csv',index=False)
    summary='72h compact tree and median search; log bps\n'+metrics[metrics.period.eq('ALL')].to_string(index=False)+'\nHistorical evaluation; no independent confirmation.\n'
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary); print('Saved:',out)


if __name__=='__main__':
    main()
