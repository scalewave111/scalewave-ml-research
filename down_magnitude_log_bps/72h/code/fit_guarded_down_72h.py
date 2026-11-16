"""User-run DOWN72h residual, gating, sizing and stability experiment."""
import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_transformed_models_72h as source
import guarded_down_72h as g

b=g.b
TARGET='down_magnitude_log_bps'
LATEST=b.HERE/'experiments/transformed_models_72h_v1/20261114T085824921737Z'


def load(args):
    x,truth,dev,old,dictionary,files=source.load(args)
    old=old[old.target.eq(TARGET)].drop(columns=['_merge'])
    done=json.loads((LATEST/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(LATEST/'predictions.csv')==done['predictions_sha256'],'Transformed reference hash')
    latest=pd.read_csv(LATEST/'predictions.csv');latest=latest[latest.target.eq(TARGET)]
    latest.decision_time=pd.to_datetime(latest.decision_time,utc=True).dt.as_unit('ns')
    latest.issue=pd.to_datetime(latest.issue,utc=True)
    keys=['decision_time','target','issue']
    old=old.merge(latest[keys+['actual','new_features']].rename(columns={'actual':'latest_actual','new_features':'previous_transformed'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==len(dev),'Latest reference alignment')
    np.testing.assert_allclose(old.actual,old.latest_actual,atol=1e-10,rtol=0)
    labels=truth[TARGET]
    baseline,counts=g.baselines(labels)
    b.check(np.isfinite(baseline[x.index.get_indexer(dev)]).all(),'Missing evaluation baseline')
    cut=dev[len(dev)//2]
    altered=labels.copy();altered.loc[altered.index+pd.Timedelta(hours=72)>=cut]=1e8
    check,_=g.baselines(altered)
    np.testing.assert_allclose(baseline[x.index<=cut],check[x.index<=cut],rtol=0,atol=1e-10,equal_nan=True)
    return x,labels.to_numpy(float),dev,old,dictionary,baseline,counts,files+[LATEST/'COMPLETE.json',LATEST/'predictions.csv']


def forecast(bundle,x,baseline,rows):
    choice=bundle['choice'];family=choice['family']
    if family=='quarterly_median':return np.full(len(rows),bundle['quarterly_median']),np.zeros(len(rows))
    if family=='rolling_median':return baseline[rows],np.zeros(len(rows))
    model=bundle['model']
    if model.get('unavailable'):return baseline[rows],np.zeros(len(rows))
    raw=g.raw_predict(model,x,baseline,rows);quality=g.dynamic_weight(model,x,rows)
    return g.adjust(raw,baseline[rows],quality,model['dead_scale'],choice['weight'],choice['deadzone'],choice['dynamic'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/guarded_down_72h_v1')
    args=parser.parse_args();b.check(args.threads>0,'Invalid threads')
    x,y,dev,old,dictionary,baseline,counts,files=load(args)
    folds={issue:g.windows(x.index,issue,baseline) for issue in b.ISSUES}
    print(f'Checked {len(x)} rows / {len(dev)} DOWN origins; four-window purges and causal rolling baseline.',flush=True)
    if args.check_inputs:
        for issue in b.ISSUES:
            admitted=[len(g.stable_features(x,y,tr)[0]) for tr,_ in folds[issue]]
            tr=np.flatnonzero((x.index+pd.Timedelta(hours=72)<issue)&np.isfinite(baseline))
            print(f'{issue:%Y-%m}: stable features in four inner fits={admitted}; outer={len(g.stable_features(x,y,tr)[0])}',flush=True)
        return
    out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    dictionary.to_csv(out/'feature_dictionary.csv',index=False)
    x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv',index=False)
    sources=[Path(__file__),Path(g.__file__),Path(source.__file__),Path(source.previous_runner.__file__),
        Path(source.source.__file__),Path(source.feature_module.__file__),Path(source.feature_module.shared.__file__),
        Path(b.__file__),Path(source.prior.__file__),Path(source.prior.canonical.__file__),
        b.HERE/'compare_rolling_medians_72h.py',b.HERE/'transformed_models_72h.py']
    guidance=[b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
        Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),target=TARGET,horizon_minutes=4320,
        families=g.FAMILIES,weights=g.WEIGHTS,deadzone_IQR_fractions=g.DEADZONES,
        baseline='180day median of labels with maturity in [t-180d,t); min48; online expanding fallback; quarterly median remains primary control',
        selection='Four earlier three-month windows; >=.1% pooled MAE gain, >=3/4 ties or wins and no window>2% worse against BOTH rolling and quarterly median. Otherwise baseline fallback.',
        stability='Training-only three trailing3month blocks: >=300values,>=80%coverage each,coverage spread<=10percentagepoints; same Spearman sign with abs>.01;top6 by minimum absolute correlation.',
        calibration='Quantile mapping uses frozen purged core model and held-out last90days of mature training; no in-sample calibration, no refit after calibration; ranks clipped .05/.95.',
        dynamic_gate='Zero weight for any selected input missing,7daymissing spike>training+10points,or insufficient48row history; otherwise weight*(1-.75*fraction outside trainingP1/P99). Proxy for shift, not calibrated confidence.',
        limitations=['Exposed adaptive historical evaluation; no guaranteed gain','Deadzone magnitude is not calibrated confidence',
        'Hard sign filter may discard useful nonlinear features','Marginal quantile mapping can worsen MAE',
        'Online baseline may use earlier evaluation labels only after72h maturity; differs from frozen quarterly baseline',
        'Prior prediction controls aligned/hashed, not independently replayed','Only DOWN fitted; UP results untouched'],
        input_hashes={str(p):b.sha(p) for p in files},source_hashes={str(p):b.sha(p) for p in sources},
        guidance_hashes={str(p):b.sha(p) for p in guidance},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__)))
    for path in sources:(out/path.name).write_bytes(path.read_bytes())
    frames=[];selections=[];ledger=[]
    with threadpool_limits(limits=args.threads):
        for i,issue in enumerate(b.ISSUES):
            end=b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te=x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            mature=np.flatnonzero(x.index+pd.Timedelta(hours=72)<issue)
            tr=mature[np.isfinite(baseline[mature])]
            folder=out/f'{issue:%Y-%m}';folder.mkdir()
            print(f'{issue:%Y-%m} DOWN selection',flush=True)
            choice=g.search(x,y,baseline,folds[issue],folder)
            model=g.fit(x,y,baseline,tr,choice['family']) if choice['family'] in g.FAMILIES else None
            bundle=dict(choice=choice,model=model,quarterly_median=float(np.median(y[mature])),issue=issue,target=TARGET)
            prediction,weight=forecast(bundle,x,baseline,te)
            joblib.dump(bundle,folder/'model.joblib')
            replay=forecast(joblib.load(folder/'model.joblib'),x,baseline,te)
            np.testing.assert_allclose(prediction,replay[0],rtol=0,atol=1e-10)
            b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
            refs=old.set_index('decision_time').loc[x.index[te]]
            np.testing.assert_allclose(refs.training_median,bundle['quarterly_median'],rtol=0,atol=1e-8)
            frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=TARGET,actual=y[te],selected=prediction,
                effective_weight=weight,rolling_median=baseline[te],rolling_count=counts[te],
                training_median=bundle['quarterly_median'],training_mean=np.mean(y[mature]),
                previous_neural=refs.previous_neural.to_numpy(),previous_fresh=refs.previous_fresh.to_numpy(),
                previous_compact=refs.previous_compact.to_numpy(),previous_transformed=refs.previous_transformed.to_numpy())))
            selections.append(dict(issue=str(issue),**choice,features=model['features'] if model else [],
                final_unavailable=model.get('unavailable',False) if model else False))
            for name,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                ledger.append(dict(issue=issue,window=name,train_rows=len(train),validation_rows=len(val),
                    train_label_end_max=x.index[train].max()+pd.Timedelta(hours=72),validation_start=x.index[val].min(),
                    validation_label_end_max=x.index[val].max()+pd.Timedelta(hours=72)))
    pred=pd.concat(frames,ignore_index=True)
    b.check(len(pred)==len(dev) and not pred.decision_time.duplicated().any(),'Output keys')
    b.check(np.isfinite(pred.selected).all() and pred.selected.ge(0).all(),'Invalid forecast')
    pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    metrics=[]
    for period,f in [('ALL',pred)]+list(pred.groupby('period')):
        for name in ['selected','rolling_median','training_median','training_mean','previous_neural','previous_fresh','previous_compact','previous_transformed']:
            metrics.append(dict(target=TARGET,period=period,model=name,**b.metric(f.actual.to_numpy(),f[name].to_numpy())))
    scores=pd.DataFrame(metrics);scores.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False);b.dump(out/'selections.json',selections)
    summary='Guarded DOWN72h; logbps. Historical exposed evaluation.\n'+scores[scores.period.eq('ALL')].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary)
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',development_origins=len(dev),target=TARGET,replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary);print('Saved:',out)


if __name__=='__main__':main()
