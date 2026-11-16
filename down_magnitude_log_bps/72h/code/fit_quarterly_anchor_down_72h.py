"""Quarterly-anchor DOWN72h residual experiment; user-run training only."""
import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_guarded_down_72h as source
import quarterly_anchor_down_72h as q

b=q.b
TARGET=source.TARGET
REFERENCE=b.HERE/'experiments/guarded_down_72h_v1/20261114T112121591551Z'


def load(args):
    x,y,dev,old,dictionary,_,_,files=source.load(args)
    done=json.loads((REFERENCE/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(REFERENCE/'predictions.csv')==done['predictions_sha256'],'Guarded reference hash')
    control=pd.read_csv(REFERENCE/'predictions.csv')
    control.decision_time=pd.to_datetime(control.decision_time,utc=True).dt.as_unit('ns')
    control.issue=pd.to_datetime(control.issue,utc=True)
    keys=['decision_time','target','issue']
    old=old.drop(columns=['_merge']).merge(control[keys+['actual','selected']].rename(columns={'actual':'guard_actual','selected':'previous_guarded'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==len(dev),'Guarded reference keys')
    np.testing.assert_allclose(old.actual,old.guard_actual,atol=1e-10,rtol=0)
    labels=pd.Series(y,index=x.index);base,ledger=q.anchors(labels)
    te=x.index.get_indexer(dev)
    b.check(all(np.isfinite(v[te]).all() for v in base.values()),'Missing evaluation anchor')
    cut=dev[len(dev)//2];changed=labels.copy();changed.loc[changed.index+pd.Timedelta(hours=72)>=cut]=1e8
    again,_=q.anchors(changed)
    for name in base:
        np.testing.assert_allclose(base[name][x.index<=cut],again[name][x.index<=cut],rtol=0,atol=1e-10,equal_nan=True)
    return x,y,dev,old,dictionary,base,ledger,files+[REFERENCE/'COMPLETE.json',REFERENCE/'predictions.csv']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--bootstrap-draws',type=int,default=5000)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/quarterly_anchor_down_72h_v1')
    args=parser.parse_args();b.check(args.threads>0 and args.bootstrap_draws>=100,'Invalid options')
    x,y,dev,old,dictionary,base,anchor_ledger,files=load(args)
    joint=np.where(np.isfinite(base['quarterly'])&np.isfinite(base['rolling']),base['quarterly'],np.nan)
    folds={issue:q.previous.windows(x.index,issue,joint) for issue in b.ISSUES}
    for issue in b.ISSUES:
        te=dev[(dev>=issue)&(dev<(issue+pd.DateOffset(months=3) if issue!=b.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=30)))]
        mature=y[x.index+pd.Timedelta(hours=72)<issue]
        np.testing.assert_allclose(base['quarterly'][x.index.get_indexer(te)],np.median(mature),atol=1e-10,rtol=0)
    print(f'Checked {len(x)} rows / {len(dev)} DOWN origins; causal quarterly anchors and four purged windows.',flush=True)
    if args.check_inputs:
        for issue in b.ISSUES:
            print(f'{issue:%Y-%m} nonredundant stable features by inner window: {[len(q.features(x,y,tr)[0]) for tr,_ in folds[issue]]}',flush=True)
        return
    out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    dictionary.to_csv(out/'feature_dictionary.csv',index=False);anchor_ledger.to_csv(out/'historical_anchor_ledger.csv',index=False)
    x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv',index=False)
    names=['fit_quarterly_anchor_down_72h.py','quarterly_anchor_down_72h.py','fit_guarded_down_72h.py','guarded_down_72h.py',
        'fit_transformed_models_72h.py','fit_compact_trees_72h.py','discover_long_features_72h.py','fresh_features_72h.py',
        'long_horizon_features.py','discover_excursion_features_72h.py','improve_excursion_features_72h.py',
        'compare_rolling_medians_72h.py','transformed_models_72h.py']
    sources=[b.HERE/n for n in names]+[Path(source.source.prior.canonical.__file__)]
    guidance=[b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
        Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),target=TARGET,horizon_minutes=4320,weights=q.WEIGHTS,
        model='Fixed MAE HGB residual:80iterations,depth2,4leaves,minleaf256,rate.05,L2=10,seed48',
        selection='Primary quarterly-anchor only;four earlier windows;>=.1%pooledgain,>=3/4ties/wins,no window>2%worse;baselinefallback',
        residual_target='Historical actual minus the median available at that historical quarter start;rolling counterpart as matched diagnostic',
        history='Both branches use identical training rows with both historical anchors available;at least48maturelabels for anchor;all valid mature labels still form quarterly benchmark',
        features='Same training-only3block sign/coverage stability;top6 after removing absoluteSpearman>=.90 duplicates with>=1000trainingpairs',
        fallback='Missing any selected input forces zero adjustment;no eligible features also zero;no deadzone or dynamic shift tuning',
        comparison='Six fixed anchor/weight variants saved diagnostically;do not select on their outer scores',
        bootstrap='5000default paired calendar-block draws at7/14/28days,stratified by modelissue with original stratum weights;conditional on fitted policy,not search-adjusted',
        limitations=['Historically exposed adaptive research;no guaranteed gain or independent confirmation',
        'Intervals do not account for repeated model selection;bootstrap positive fraction is not probability of true superiority',
        'Calendar blocking approximates dependence;few long blocks per quarter',
        'Redundancy and fixed model differ from prior guarded run;matched new branches isolate anchors within this experiment only',
        'UP untouched;old prediction controls checked,not old model replayed'],
        input_hashes={str(p):b.sha(p) for p in files},source_hashes={str(p):b.sha(p) for p in sources},
        guidance_hashes={str(p):b.sha(p) for p in guidance},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__)))
    for path in sources:(out/path.name).write_bytes(path.read_bytes())
    frames=[];selections=[];fit_ledger=[]
    with threadpool_limits(limits=args.threads):
        for i,issue in enumerate(b.ISSUES):
            end=b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te=x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            mature=np.flatnonzero(x.index+pd.Timedelta(hours=72)<issue)
            tr=mature[np.isfinite(joint[mature])]
            folder=out/f'{issue:%Y-%m}';folder.mkdir()
            print(f'{issue:%Y-%m} DOWN anchor comparison',flush=True)
            choice=q.search(x,y,base,folds[issue],folder)
            variants={};models={}
            for anchor in ['quarterly','rolling']:
                model=q.fit(x,y,base[anchor],tr);models[anchor]=model
                for w in q.WEIGHTS:variants[f'{anchor}_w{int(w*100):02d}']=q.predict(model,x,base[anchor],te,w)[0]
            b.check(models['quarterly']['features']==models['rolling']['features'],'Anchor feature sets differ')
            if choice['weight']==0:prediction=base['quarterly'][te];effective=np.zeros(len(te))
            else:prediction,effective=q.predict(models['quarterly'],x,base['quarterly'],te,choice['weight'])
            bundle=dict(models=models,choice=choice,issue=issue,target=TARGET)
            joblib.dump(bundle,folder/'model.joblib');saved=joblib.load(folder/'model.joblib')
            for anchor in models:
                for w in q.WEIGHTS:
                    replay=q.predict(saved['models'][anchor],x,base[anchor],te,w)[0]
                    np.testing.assert_allclose(variants[f'{anchor}_w{int(w*100):02d}'],replay,rtol=0,atol=1e-10)
            b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
            refs=old.set_index('decision_time').loc[x.index[te]]
            np.testing.assert_allclose(base['quarterly'][te],refs.training_median,atol=1e-8,rtol=0)
            frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=TARGET,actual=y[te],selected=prediction,
                effective_weight=effective,training_median=base['quarterly'][te],rolling_median=base['rolling'][te],training_mean=np.mean(y[mature]),
                previous_guarded=refs.previous_guarded.to_numpy(),previous_transformed=refs.previous_transformed.to_numpy(),
                previous_neural=refs.previous_neural.to_numpy(),**variants)))
            selections.append(dict(issue=str(issue),**choice,features=models['quarterly']['features'],
                final_unavailable=models['quarterly'].get('unavailable',False)))
            for name,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                fit_ledger.append(dict(issue=issue,window=name,training_rows=len(train),evaluation_rows=len(val),
                    training_label_end_max=x.index[train].max()+pd.Timedelta(hours=72),evaluation_start=x.index[val].min()))
    pred=pd.concat(frames,ignore_index=True)
    b.check(len(pred)==len(dev) and not pred.decision_time.duplicated().any(),'Prediction coverage')
    model_names=['selected','training_median','rolling_median','training_mean','previous_guarded','previous_transformed','previous_neural']+list(variants)
    b.check(np.isfinite(pred[model_names]).all().all() and pred[model_names].ge(0).all().all(),'Invalid predictions')
    pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    metrics=[dict(target=TARGET,period=period,model=name,**b.metric(f.actual.to_numpy(),f[name].to_numpy()))
        for period,f in [('ALL',pred)]+list(pred.groupby('period')) for name in model_names]
    scores=pd.DataFrame(metrics);scores.to_csv(out/'metrics.csv',index=False)
    bootstrap=[q.bootstrap(pred,name,days,args.bootstrap_draws) for name in ['selected']+list(variants) for days in [7,14,28]]
    b.dump(out/'bootstrap.json',bootstrap)
    pd.DataFrame([{k:v for k,v in row.items() if k!='blocks'} for row in bootstrap]).to_csv(out/'bootstrap.csv',index=False)
    pd.DataFrame(fit_ledger).to_csv(out/'fit_windows.csv',index=False);b.dump(out/'selections.json',selections)
    summary='Quarterly-anchor DOWN72h; logbps; fixed variants are diagnostics, not outer-selected.\n'+scores[scores.period.eq('ALL')].to_string(index=False)
    summary+='\nSelected-policy paired MAE gain intervals (positive=better than quarterly median):\n'+pd.DataFrame([v for v in bootstrap if v['model']=='selected'])[['block_days','mae_gain','lower95','upper95']].to_string(index=False)
    summary+='\nConditional bootstrap; not adjusted for adaptive search. Historical exposed evaluation.\n'
    (out/'SUMMARY.txt').write_text(summary)
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',target=TARGET,development_origins=len(dev),replay_passed=True,
        predictions_sha256=b.sha(out/'predictions.csv'),bootstrap_sha256=b.sha(out/'bootstrap.json')))
    print(summary);print('Saved:',out)


if __name__=='__main__':main()
