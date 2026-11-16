"""Fast DOWN72h focused market-feature trees; candidate and guarded policy both saved."""
import argparse
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_quarterly_anchor_down_72h as source
import market_state_features_72h as feature_module
import focused_market_trees_72h as models
import fit_market_state_down_72h as previous_runner

b=models.b
TARGET=source.TARGET
REFERENCE=b.HERE/'experiments/quarterly_anchor_down_72h_v1/20261114T113416293003Z'


def windows(index,issue):
    result=[]
    for offset in [12,9,6,3]:
        start=issue-pd.DateOffset(months=offset);end=start+pd.DateOffset(months=3)
        tr=np.flatnonzero(index+pd.Timedelta(hours=72)<start)
        va=np.flatnonzero((index>=start)&(index+pd.Timedelta(hours=72)<end))
        b.check(len(tr)>=1000 and len(va)>=100,'Insufficient history')
        result.append((tr,va))
    return result


FAST_REFERENCE=b.HERE/'experiments/market_state_down_72h_fast_v1/20261114T120044967681Z'


def load(args):
    x,y,dev,old,dictionary,files=previous_runner.load(args)
    done=json.loads((FAST_REFERENCE/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(FAST_REFERENCE/'predictions.csv')==done['predictions_sha256'],'Fast reference hash')
    control=pd.read_csv(FAST_REFERENCE/'predictions.csv')
    control.decision_time=pd.to_datetime(control.decision_time,utc=True).dt.as_unit('ns')
    control.issue=pd.to_datetime(control.issue,utc=True)
    keys=['decision_time','target','issue']
    old=old.drop(columns=['_merge']).merge(control[keys+['actual','selected']].rename(columns={'actual':'fast_actual','selected':'previous_fast'}),on=keys,how='outer',validate='one_to_one',indicator=True)
    b.check(old['_merge'].eq('both').all() and len(old)==len(dev),'Fast reference keys')
    np.testing.assert_allclose(old.actual,old.fast_actual,rtol=0,atol=1e-10)
    for pack in models.PACKS:models.pack_columns(x,pack)
    return x,y,dev,old,dictionary,files+[FAST_REFERENCE/'COMPLETE.json',FAST_REFERENCE/'predictions.csv']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output-dir',type=Path,default=b.HERE/'experiments/focused_market_down_72h_v1')
    args=parser.parse_args();b.check(args.threads>0,'Invalid threads')
    
    x,y,dev,old,dictionary,files=load(args)
    folds={issue:windows(x.index,issue) for issue in b.ISSUES}
    print(f'Checked {len(x)} inputs/{len(dev)} DOWN origins/{len(x.columns)} candidates; causal market-state audit passed.',flush=True)
    if args.check_inputs:
        for issue in b.ISSUES:
            tr=folds[issue][0][0]
            training=x.iloc[tr]
            counts={pack:sum(np.isfinite(training[c]).mean()>=.8 and np.isfinite(training[c]).sum()>=1000 and training[c].nunique()>1 for c in models.pack_columns(x,pack)) for pack in models.PACKS}
            print(f'{issue:%Y-%m} earliest inner-training admitted columns: {counts}',flush=True)
        return
    out=args.output_dir/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True,exist_ok=False)
    dictionary.to_csv(out/'feature_dictionary.csv',index=False);x.attrs['excluded_labels'].to_csv(out/'excluded_labels.csv',index=False)
    names=['fit_focused_market_down_72h.py','focused_market_trees_72h.py','fit_market_state_down_72h.py','market_state_features_72h.py','market_state_trees_72h.py','fit_quarterly_anchor_down_72h.py',
        'quarterly_anchor_down_72h.py','fit_guarded_down_72h.py','guarded_down_72h.py','fit_transformed_models_72h.py',
        'fit_compact_trees_72h.py','discover_long_features_72h.py','fresh_features_72h.py','long_horizon_features.py',
        'discover_excursion_features_72h.py','improve_excursion_features_72h.py','compare_rolling_medians_72h.py','transformed_models_72h.py']
    paths=[b.HERE/n for n in names]+[Path(source.source.source.prior.canonical.__file__)]
    guidance=[b.HERE.parent/'PROJECT_CONTEXT.md',Path('C:/Users/Siddh/Desktop/ML/ml_framework/AGENTS.md'),
        Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/WORKFLOW.md'),Path('C:/Users/Siddh/Desktop/ML/ml_framework/docs/methods/DATA_AND_METRICS.md')]
    b.dump(out/'CONTRACT.json',dict(arguments=vars(args),target=TARGET,horizon_minutes=4320,sources=feature_module.SOURCES,
        candidate_features=list(x),configs=models.CONFIG,packs={p:models.pack_columns(x,p) for p in models.PACKS},weights=models.WEIGHTS,
        fast=True,estimator_training_row_cap=6000,
        sampling='Deterministic evenly-spaced historical training subset for estimator only;quality/preprocessing/median use full mature training;validation/evaluation never subsampled',
        method='Focused38/30/62feature packs; best earlier candidate also saved separately regardless of acceptance; selected retains unchanged guard; train-only80%coverage/1000count/variation admission per fit;no univariate target screen,no forced old selected subset;four earlier purged quarters select MAE with.1%gain/3wins-or-ties/no window>2%worse;medianfallback',
        availability='Masked source half-hour snapshots strictly before origin;no future/backfill;rolling80%expectedcoverage;original full X inventory independent of target validity',
        fallback='If less than80%selected predictors finite at forecast,trainingmedian;otherwise .5/1 candidate blend',
        limitations=['Exposed adaptive historical research;no guaranteed win','Cross-products are hypotheses,not causal facts',
        'Snapshot aggregates are not full5minute histories;missing source values remain missing',
        'Inherits source-X masking and availability limitations;new source availability provenance not independently reconstructed',
        'Historical liquidation activity rate is not current liquidation size;flow activity is not signed flow',
        'Feature admission per training fit;quarterly selected feature inventory may vary',
        'Prior prediction controls hashed/aligned,not prior model replayed;UP untouched'],
        input_hashes={str(p):b.sha(p) for p in files},source_hashes={str(p):b.sha(p) for p in paths},guidance_hashes={str(p):b.sha(p) for p in guidance},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=b.sklearn.__version__)))
    for p in paths:(out/p.name).write_bytes(p.read_bytes())
    frames=[];selections=[];ledger=[]
    with threadpool_limits(limits=args.threads):
        for i,issue in enumerate(b.ISSUES):
            end=b.ISSUES[i+1] if i+1<len(b.ISSUES) else dev.max()+pd.Timedelta(minutes=30)
            te=x.index.get_indexer(dev[(dev>=issue)&(dev<end)])
            tr=np.flatnonzero(x.index+pd.Timedelta(hours=72)<issue)
            folder=out/f'{issue:%Y-%m}';folder.mkdir()
            print(f'{issue:%Y-%m} market-state search',flush=True)
            plan=models.search(x,y,folds[issue],folder)
            choice=plan['selected'];candidate_choice=plan['candidate']
            candidate_model=models.fit(x.iloc[tr],y[tr],candidate_choice)
            candidate=models.predict(candidate_model,x.iloc[te],candidate_choice.get('weight',1.))
            model=(models.fit(x.iloc[tr],y[tr],choice) if choice!=candidate_choice else candidate_model)
            prediction=models.predict(model,x.iloc[te],choice.get('weight',1.))
            joblib.dump(dict(model=model,choice=choice,candidate_model=candidate_model,candidate_choice=candidate_choice,issue=issue,target=TARGET),folder/'model.joblib')
            saved=joblib.load(folder/'model.joblib')
            np.testing.assert_allclose(prediction,models.predict(saved['model'],x.iloc[te],saved['choice'].get('weight',1.)),rtol=0,atol=1e-10)
            np.testing.assert_allclose(candidate,models.predict(saved['candidate_model'],x.iloc[te],saved['candidate_choice'].get('weight',1.)),rtol=0,atol=1e-10)
            b.dump(folder/'MODEL_HASH.json',dict(sha256=b.sha(folder/'model.joblib')))
            refs=old.set_index('decision_time').loc[x.index[te]]
            median=np.median(y[tr]);np.testing.assert_allclose(refs.training_median,median,atol=1e-8,rtol=0)
            frames.append(pd.DataFrame(dict(decision_time=x.index[te],issue=issue,target=TARGET,actual=y[te],selected=prediction,candidate=candidate,previous_fast=refs.previous_fast.to_numpy(),
                training_median=median,training_mean=np.mean(y[tr]),previous_neural=refs.previous_neural.to_numpy(),
                previous_transformed=refs.previous_transformed.to_numpy(),previous_guarded=refs.previous_guarded.to_numpy(),previous_anchor=refs.previous_anchor.to_numpy())))
            selections.append(dict(issue=str(issue),selected={**choice,'features':model['features']},candidate={**candidate_choice,'features':candidate_model['features']}))
            b.dump(folder/'final_quality.json',model.get('quality',[]))
            for name,(train,val) in [('outer',(tr,te))]+[(f'inner_{j+1}',pair) for j,pair in enumerate(folds[issue])]:
                ledger.append(dict(issue=issue,window=name,training_rows=len(train),evaluation_rows=len(val),
                    training_label_end_max=x.index[train].max()+pd.Timedelta(hours=72),evaluation_start=x.index[val].min()))
    pred=pd.concat(frames,ignore_index=True)
    b.check(len(pred)==len(dev) and not pred.decision_time.duplicated().any() and np.isfinite(pred.selected).all(),'Invalid output')
    pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
    pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
    pred.to_csv(out/'predictions.csv',index=False)
    scores=pd.DataFrame([dict(target=TARGET,period=period,model=name,**b.metric(f.actual.to_numpy(),f[name].to_numpy()))
        for period,f in [('ALL',pred)]+list(pred.groupby('period')) for name in ['selected','candidate','previous_fast','training_median','training_mean','previous_neural','previous_transformed','previous_guarded','previous_anchor']])
    scores.to_csv(out/'metrics.csv',index=False);b.dump(out/'selections.json',selections);pd.DataFrame(ledger).to_csv(out/'fit_windows.csv',index=False)
    summary='Focused market-state DOWN72h;errors in logbps; candidate is diagnostic, selected retains acceptance gate.\n'+scores[scores.period.eq('ALL')].to_string(index=False)+'\nHistorically exposed research.\n'
    (out/'SUMMARY.txt').write_text(summary)
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',target=TARGET,development_origins=len(dev),replay_passed=True,predictions_sha256=b.sha(out/'predictions.csv')))
    print(summary);print('Saved:',out)


if __name__=='__main__':main()
