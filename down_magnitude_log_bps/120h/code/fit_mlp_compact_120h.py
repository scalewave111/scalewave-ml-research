"""User-run small 120h MLP experiment with original-unit MAE and median blends."""
import argparse
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import improve_excursion_features_120h as prior
from long_horizon_features import build
import long_horizon_features as feature_module
import mlp_search_120h as neural

b = prior.base
REFERENCE = b.HERE/'experiments/excursion_pipeline_120h_v1/20261113T114805583307Z/price_followup/20261113T120541578982Z'
LONG_REFERENCE = b.HERE/'experiments/long_features_120h_v1/20261113T123515172594Z'


def load(args):
    contract = json.loads((REFERENCE/'CONTRACT.json').read_text())
    done = json.loads((REFERENCE/'COMPLETE.json').read_text())
    b.check(contract['horizon_minutes'] == 7200 and done['status'] == 'COMPLETE', 'Bad reference')
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
    x, dictionary = build(r, original.index, 120)
    x.attrs['excluded_labels'] = original.attrs['excluded_labels']
    cut = original.index[len(original)//2]
    keys = original.index[original.index <= cut][-10:]
    pd.testing.assert_frame_equal(build(r, keys, 120)[0], build(r[r.index <= cut], keys,120)[0])
    changed = r.copy(); changed.loc[changed.index >= cut] = 1e6
    pd.testing.assert_frame_equal(build(r, keys,120)[0], build(changed,keys,120)[0])
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
    long_done=json.loads((LONG_REFERENCE/'COMPLETE.json').read_text())
    b.check(long_done['status']=='COMPLETE' and b.sha(LONG_REFERENCE/'predictions.csv')==long_done['predictions_sha256'],'Long-feature reference changed')
    long=pd.read_csv(LONG_REFERENCE/'predictions.csv')
    long['decision_time']=pd.to_datetime(long.decision_time,utc=True).dt.as_unit('ns')
    long['issue']=pd.to_datetime(long.issue,utc=True)
    old=old.merge(long[['decision_time','issue','target','actual','new_features']],on=['decision_time','issue','target'],how='outer',validate='one_to_one',suffixes=('','_long'),indicator=True)
    b.check(old['_merge'].eq('both').all(),'Long-reference keys differ')
    np.testing.assert_allclose(old.actual,old.actual_long,rtol=0,atol=1e-10)
    old=old.rename(columns={'new_features':'long_reference'})
    return x,y,dev,old,dictionary,files+[returns,REFERENCE/'CONTRACT.json',REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv',LONG_REFERENCE/'COMPLETE.json',LONG_REFERENCE/'predictions.csv']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/mlp_compact_120h_v1')
    p.add_argument('--threads',type=int,default=2)
    args = p.parse_args()
    b.check(args.threads>0,'Invalid threads')
    neural.torch.set_num_threads(args.threads)
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
    sources = [Path(__file__),Path(feature_module.__file__),Path(b.__file__),Path(prior.__file__),Path(prior.canonical.__file__),Path(neural.__file__)]
    guidance = [b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),horizon_minutes=7200,targets=b.TARGETS,
        candidate_features=neural.PACKS,configs=neural.CONFIGS,seeds=neural.SEEDS,blend_weights=neural.WEIGHTS,target_modes=['raw','volatility'],reference=REFERENCE,lookbacks_hours=[24,60,120,168,240,336,720,1440],
        method='Four fixed four-feature groups; 2MLP architectures x raw/volatility-scaled outputs; 2seed average; earlier3window MAE gain>=.1percent,2wins,no window>2percent worse than median; else median. MAE evaluated and optimized in original logbps. Old forecasts comparison only.',
        availability='Hourly bin labelled t contains 5m return endpoints [t-1h,t); only completed bins used, no future fills. Half-hour origins reuse latest complete hour.',
        missingness='Every hour requires 12 finite returns; every rolling feature requires all hours in its window; missing remains NaN. No invalid target imputation.',
        limitations=['Historically exposed evaluation; adaptive inner search, not fresh confirmation','Longer inputs are hypotheses, not proven predictors','Hourly ranges describe hourly sampled canonical log-price paths, not exchange highs/lows','Fixed small feature groups may miss useful combinations','Overlapping 120h targets are dependent','Old model forecasts are hashed and aligned, not refitted or independently replayed here'],
        input_hashes={str(v):b.sha(v) for v in files},source_hashes={str(v):b.sha(v) for v in sources},
        guidance_hashes={str(v):b.sha(v) for v in guidance},
        normalization='Causal 120h volatility=five-day sqrt-hour scale; train-only P10 floor>=1 and median fallback. Positive Softplus prediction multiplied by scale; original-unit MAE loss, divided only by training target median for numeric stability.',
        optimizer='CPU AdamW lr.001 weight_decay.1; batch256; gradient norm clip1; fixed40/60epochs; deterministic seed48/148 ensemble; no early stopping',
        preprocessing='Train-only median imputation,missing indicators,standardization; clip standardized inputs to[-8,8]',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__,torch=neural.torch.__version__)))
    for source in sources:
        (out/source.name).write_bytes(source.read_bytes())
    frames, selections, ledger = [],[],[]
    with threadpool_limits(limits=args.threads):
        for i,issue in enumerate(b.ISSUES):
            end = b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te = x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            tr = np.flatnonzero(x.index+pd.Timedelta(hours=120)<issue)
            for target in b.TARGETS:
                print(f'{issue:%Y-%m} {target}: new-feature selection',flush=True)
                folder = out/f'{issue:%Y-%m}_{target}'; folder.mkdir()
                truth = y[target].to_numpy(float)
                choice = neural.search(x,truth,folds[issue],args,folder)
                cols = choice['features']; a = x[cols].to_numpy(float)
                model = neural.fit(x.iloc[tr],truth[tr],choice)
                prediction = neural.predict(model,x.iloc[te])
                joblib.dump(dict(model=model,features=cols,target=target,issue=issue),folder/'model.joblib')
                saved = joblib.load(folder/'model.joblib')
                np.testing.assert_allclose(prediction,neural.predict(saved['model'],x.iloc[te]),rtol=0,atol=1e-10)
                b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
                ref = old[old.target.eq(target)].set_index('decision_time').loc[x.index[te]]
                frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=target,actual=truth[te],
                    mlp_selected=prediction,previous=ref.selected.to_numpy(),long_reference=ref.long_reference.to_numpy(),training_median=np.median(truth[tr]),training_mean=np.mean(truth[tr]))))
                selections.append(dict(issue=str(issue),target=target,**choice))
                for label,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                    ledger.append(dict(issue=issue,target=target,window=label,training_rows=len(train),evaluation_rows=len(val),
                        training_label_end_max=x.index[train].max()+pd.Timedelta(hours=120),evaluation_start=x.index[val].min(),
                        evaluation_label_end_max=x.index[val].max()+pd.Timedelta(hours=120)))
    pred = pd.concat(frames,ignore_index=True)
    b.check(len(pred)==2*len(dev) and not pred.duplicated(['decision_time','target']).any(),'Output coverage')
    pred['period'] = pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    scores=[]
    for target,g in pred.groupby('target'):
        for period,f in [('ALL',g)]+list(g.groupby('period')):
            for model in ['mlp_selected','previous','long_reference','training_median','training_mean']:
                scores.append(dict(target=target,period=period,model=model,**b.metric(f.actual.to_numpy(),f[model].to_numpy())))
    metrics=pd.DataFrame(scores); metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    b.dump(out/'selections.json',selections)
    chosen=pd.DataFrame([dict(issue=s['issue'],target=s['target'],feature=c) for s in selections for c in s['features']],columns=['issue','target','feature'])
    chosen.merge(dictionary,on='feature',how='left').to_csv(out/'selected_features.csv',index=False)
    summary='120h small MLP with median blends; log bps\n'+metrics[metrics.period.eq('ALL')].to_string(index=False)+'\nHistorical evaluation; no independent confirmation.\n'
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary); print('Saved:',out)


if __name__=='__main__':
    main()

