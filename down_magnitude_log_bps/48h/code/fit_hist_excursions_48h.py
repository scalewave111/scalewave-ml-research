"""User-run histogram boosting for48h path excursions; no fitting on import."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import fit_excursions_48h as prior
import improve_excursion_features_4h as price

b=prior.b
REFERENCE=b.EXP/'excursion_models_48h_v1/20261113T093614902840Z'
SETTINGS=[dict(id='small',max_depth=2,max_leaf_nodes=4,min_samples_leaf=128,
               max_iter=80,learning_rate=.05,l2_regularization=10.),
          dict(id='medium',max_depth=3,max_leaf_nodes=7,min_samples_leaf=64,
               max_iter=120,learning_rate=.05,l2_regularization=5.)]


def fit(x,target,keys,spec):
    params={k:v for k,v in spec['settings'].items() if k!='id'}
    model=HistGradientBoostingRegressor(loss=spec['loss'],early_stopping=False,
        random_state=48,max_bins=127,**params)
    # Native NaN handling and quantile binning are learned only from training rows.
    model.fit(x.loc[keys,spec['columns']].to_numpy(float),target.loc[keys].to_numpy(float))
    return dict(model=model,columns=spec['columns'],spec=spec,training_rows=len(keys))


def predict(a,x):
    result=np.maximum(0,a['model'].predict(x[a['columns']].to_numpy(float)))
    b.check(np.isfinite(result).all(),'Nonfinite HGB prediction')
    return result


def losses(truth,pred):
    error=np.asarray(pred)-np.asarray(truth)
    return dict(mae=float(np.abs(error).mean()),mse=float(np.mean(error**2)))


def choose(scores,metric,baseline):
    best=min(range(len(scores)),key=lambda i:(scores[i][metric],i))
    return best,scores[best][metric]<baseline-1e-9


def load(root,reference):
    raw,engineered,y,dev,population,files,audit=prior.load(root)
    r=price.canonical.load_returns(prior.e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet')
    extras,groups=price.price_features(r,raw.index)
    price.causal_check(r,raw.index)
    x=raw.join(engineered).join(extras).replace([np.inf,-np.inf],np.nan)
    b.check(x.columns.is_unique and x.index.equals(y.index),'Feature alignment differs')
    packs={'raw13':list(raw),'raw13_price51':list(raw)+list(extras),
           'engineered235':list(engineered),'engineered235_price51':list(engineered)+list(extras)}
    done=json.loads((reference/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and done['replay_passed'],'Prior run incomplete')
    b.check(b.sha(reference/'predictions.csv')==done['prediction_sha256'],'Reference prediction hash mismatch')
    old=pd.read_csv(reference/'predictions.csv')
    old['decision_time']=pd.to_datetime(old.decision_time,utc=True).dt.as_unit('ns')
    old['issue']=pd.to_datetime(old.issue,utc=True)
    b.check(set(old.target)==set(prior.TARGETS) and not old.duplicated(['decision_time','target']).any(),'Reference targets/keys differ')
    for t,g in old.groupby('target'):
        g=g.set_index('decision_time').sort_index()
        b.check(g.index.equals(dev.sort_values()),'Reference origins differ')
        np.testing.assert_allclose(g.actual_log_bps,y.loc[g.index,t],atol=1e-7,rtol=0)
        for issue in b.ISSUES:
            end=issue+pd.DateOffset(months=3) if issue!=b.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
            b.check(g.index[g.issue.eq(issue)].equals(dev[(dev>=issue)&(dev<end)]),'Reference issuance differs')
    return x,y,dev,population,files,audit,packs,groups,old


def self_test():
    rng=np.random.default_rng(48)
    idx=pd.date_range('2020-01-01',periods=1600,freq='h',tz='UTC')
    x=pd.DataFrame({'signal':rng.normal(size=len(idx)),'noise':rng.normal(size=len(idx))},index=idx)
    x.loc[idx[::17],'noise']=np.nan
    target=pd.Series(100+30*x.signal**2+5*rng.normal(size=len(idx)),index=idx)
    cut=idx[1250]; keys=idx[idx+pd.Timedelta(hours=48)<cut];test=idx[idx>=cut]
    assert (keys+pd.Timedelta(hours=48)<test.min()).all()
    with threadpool_limits(limits=1):
        for loss in ['squared_error','absolute_error']:
            spec=dict(loss=loss,columns=list(x),settings=SETTINGS[0])
            a=fit(x,target,keys,spec);pred=predict(a,x.loc[test])
            base=float(target.loc[keys].mean() if loss=='squared_error' else target.loc[keys].median())
            metric='mse' if loss=='squared_error' else 'mae'
            assert losses(target.loc[test],pred)[metric]<losses(target.loc[test],np.full(len(test),base))[metric]
            changed=x.copy();changed.loc[test]=1e9
            changed_y=target.copy();changed_y.loc[test]=1e9
            again=fit(changed,changed_y,keys,spec)
            np.testing.assert_allclose(pred,predict(again,x.loc[test]),rtol=0,atol=1e-10)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.joblib';joblib.dump(a,path)
                np.testing.assert_allclose(pred,predict(joblib.load(path),x.loc[test]))
    assert choose([dict(mae=2,mse=9),dict(mae=3,mse=4)],'mae',1)==(0,False)
    assert choose([dict(mae=2,mse=9),dict(mae=3,mse=4)],'mse',10)==(1,True)
    ridx=pd.date_range('2020-01-01',periods=3000,freq='5min',tz='UTC')
    r=pd.Series(rng.normal(size=len(ridx)),index=ridx)
    price.causal_check(r,pd.DatetimeIndex([ridx[2000],ridx[2400],ridx[2500]]))
    origin=ridx[2500]
    f,_=price.price_features(r,pd.DatetimeIndex([origin]))
    recent=r[(r.index>=origin-pd.Timedelta(hours=4))&(r.index<origin)]
    path=np.r_[0.,recent.cumsum().to_numpy()]
    np.testing.assert_allclose(f.iloc[0]['NEWPRICE__net__4h'],recent.sum())
    np.testing.assert_allclose(f.iloc[0]['NEWPRICE__past_range__4h'],path.max()-path.min())
    np.testing.assert_allclose(f.iloc[0]['NEWPRICE__distance_from_high__4h'],path.max()-path[-1])
    np.testing.assert_allclose(f.iloc[0]['NEWPRICE__distance_from_low__4h'],path[-1]-path.min())
    missing=r.copy();missing.loc[recent.index[-5]]=np.nan
    ff,_=price.price_features(missing,pd.DatetimeIndex([origin]))
    assert pd.isna(ff.iloc[0]['NEWPRICE__net__4h'])
    print('PASS: both HGB losses detect nonlinear signal;48hpurge;future/preprocessing isolation;selection/fallback;price formulas/causality/missingness;saved replay. Synthetic only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root',type=Path,default=prior.source.CANONICAL)
    p.add_argument('--reference-run',type=Path,default=REFERENCE)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--check-data-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args();b.check(args.threads>0,'Positive threads required')
    if args.self_test:self_test();return
    x,y,dev,population,files,audit,packs,groups,old=load(args.canonical_root,args.reference_run)
    if args.check_data_only:
        print(json.dumps(dict(**audit,packs={k:len(v) for k,v in packs.items()}),indent=2))
        print('PASS: matched reference labels/origins,causal price inputs;no fitting.');return
    modules=[prior,prior.e2,prior.e2.old,prior.e2.old.data,b,prior.e2.price,prior.e2.price.canonical,
             prior.package,prior.source,price]
    files += [Path(__file__)]+[Path(m.__file__) for m in modules]+[
        prior.e2.old.data.PREPARED,prior.e2.old.data.DATA/'X_train.parquet',prior.e2.old.data.DATA/'X_development.parquet',
        prior.e2.old.data.REF/'signed_return_predictions.csv',prior.e2.old.data.REF/'COMPLETE.json',
        prior.e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet',args.reference_run/'predictions.csv',
        args.reference_run/'COMPLETE.json',args.reference_run/'CONTRACT.json']
    specs=[dict(id=f'{pack}_{loss}_{setting["id"]}',pack=pack,columns=cols,loss=loss,settings=setting)
        for pack,cols in packs.items() for loss in ['squared_error','absolute_error'] for setting in SETTINGS]
    identity=dict(hashes={str(f):b.sha(f) for f in files},specs=specs,threads=args.threads,
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']})
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'hist_excursion_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not(out/'COMPLETE.json').exists(),'Already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature']==signature,'Resume signature differs')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,identity=identity,input_audit=audit,price_groups=groups,
            model='sklearn HistGradientBoostingRegressor;fixed80/120iterations,no random earlystop;native missing values;training-only binning;seed48,max_bins127;nonnegative forecasts.',
            selection='3earlier2monthwindows offset9/6/3months,strict48hpurge. L2 candidates selected by pooledMSE;L1 by pooledMAE. BestHGB each loss;gated selectors retain mean/median if corresponding earlier baseline is better. Eachpack/loss also reported separately.',
            prices='51past-only features at1/4/12/24/48/168h plus3RVratios. 5mclose paths,not intrabar extrema. Some new columns duplicate or correlate with existing E2 measures. No future inputs.',
            scope='Both48hexcursion targets on same18385evalorigins;96bindingcrossing exclusions preserved;logbps. No direction filter or probability combination.',
            limitations='Inheritedfeaturefamilies and repeatedly exposedhistory;pairedpack comparisons are exploratory. SeparateMAE/RMSEpolicies,not posthoc winner selection. No runtime or improvement guarantee.'))
        population.to_csv(out/'POPULATION_AUDIT.csv',index_label='decision_time')
        for f in files:
            if f.suffix=='.py':(out/f.name).write_bytes(f.read_bytes())
    cache=out/'cache';cache.mkdir(exist_ok=True)
    def cached(name,compute):
        path=cache/(name+'.joblib');receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):return joblib.load(path)
        a=compute();joblib.dump(a,path,compress=3);b.dump(receipt,dict(sha256=b.sha(path)));return a
    frames=[];ledger=[];selections=[]
    print('Output:',out,flush=True)
    with threadpool_limits(limits=args.threads):
        for t in prior.TARGETS:
            target=y[t];side='UP' if t.startswith('up_') else 'DOWN'
            previous=old[old.target.eq(t)].set_index('decision_time').sort_index()
            def model_at(spec,cut):
                keys=y.index[y.label_end<cut]
                return cached(f'{side}_{spec["id"]}_{cut:%Y%m%d}',lambda:fit(x,target,keys,spec))
            for issue in b.ISSUES:
                windows=[]
                for offset in [9,6,3]:
                    cut=issue-pd.DateOffset(months=offset)
                    va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&(y.label_end<issue)]
                    tr=y.index[y.label_end<cut]
                    b.check(len(tr)>1000 and len(va)>100 and y.loc[tr,'label_end'].max()<va.min(),'Invalid purge/window')
                    windows.append((cut,tr,va))
                truths=np.concatenate([target.loc[va] for _,_,va in windows])
                mean=np.concatenate([np.full(len(va),target.loc[tr].mean()) for _,tr,va in windows])
                median=np.concatenate([np.full(len(va),target.loc[tr].median()) for _,tr,va in windows])
                controls={'squared_error':losses(truths,mean)['mse'],'absolute_error':losses(truths,median)['mae']}
                scores=[]
                for spec in specs:
                    pred=np.concatenate([predict(model_at(spec,cut),x.loc[va]) for cut,_,va in windows])
                    score=losses(truths,pred);scores.append(score)
                    ledger.append(dict(target=t,issue=issue,candidate=spec['id'],**score))
                    pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv',index=False)
                    print(side,issue.date(),spec['id'],flush=True)
                picks={}
                for loss,metric in [('squared_error','mse'),('absolute_error','mae')]:
                    eligible=[i for i,s in enumerate(specs) if s['loss']==loss]
                    best=min(eligible,key=lambda i:(scores[i][metric],specs[i]['id']))
                    label='rmse' if loss=='squared_error' else 'mae'
                    picks['hgb_'+label]=best
                    picks['selected_'+label]=best if scores[best][metric]<controls[loss]-1e-9 else None
                    for pack in packs:
                        ids=[i for i in eligible if specs[i]['pack']==pack]
                        picks[f'{pack}_{label}']=min(ids,key=lambda i:(scores[i][metric],specs[i]['id']))
                te=previous.index[previous.issue.eq(issue)]
                frame=pd.DataFrame({'target':t,'issue':issue,'actual_log_bps':target.loc[te]},index=te)
                for name in ['mean','median','fixed_ridge','ridge','tree','e2','selected']:
                    frame['prior_'+name]=previous.loc[te,name]
                outer=y.index[y.label_end<issue]
                for label,stat in [('rmse','mean'),('mae','median')]:
                    np.testing.assert_allclose(frame['prior_'+stat],getattr(target.loc[outer],stat)(),atol=1e-7,rtol=0)
                for name,i in picks.items():
                    if i is None:
                        frame[name]=frame['prior_mean' if name.endswith('rmse') else 'prior_median']
                        selections.append(dict(target=t,issue=issue,policy=name,candidate='constant',artifact=None))
                        continue
                    a=model_at(specs[i],issue)
                    path=out/f'{side}_{issue:%Y%m}_{name}.joblib';joblib.dump(a,path,compress=3)
                    pred=predict(a,x.loc[te]);np.testing.assert_allclose(pred,predict(joblib.load(path),x.loc[te]),atol=1e-7,rtol=0)
                    frame[name]=pred
                    selections.append(dict(target=t,issue=issue,policy=name,candidate=specs[i],validation=scores[i],
                        training_rows=len(outer),max_training_label_end=y.loc[outer,'label_end'].max(),
                        artifact=path.name,sha256=b.sha(path),replay_passed=True))
                b.dump(out/'SELECTIONS.json',selections)
                frame.to_csv(out/f'{side}_{issue:%Y%m}_predictions.csv',index_label='decision_time');frames.append(frame)
    result=pd.concat(frames).sort_index();result.to_csv(out/'predictions.csv',index_label='decision_time')
    models=[c for c in result if c not in ['target','issue','actual_log_bps']]
    rows=[]
    for t,g in result.groupby('target'):
        b.check(g.index.is_unique and g.index.sort_values().equals(dev.sort_values()),'Final coverage differs')
        periods=[('ALL',g)]+[(f'Cycle {q}',g[(g.index.year==2024)&(g.index.quarter==q)]) for q in range(1,5)]+[('Jan2025',g[g.index.year==2025])]
        for period,h in periods:
            if h.empty:continue
            for model in models:
                m=b.metric(h.actual_log_bps,h[model]);ref=b.metric(h.actual_log_bps,h.prior_selected)
                rows.append(dict(target=t,period=period,model=model,rows=len(h),mae_log_bps=m['mae_bps'],rmse_log_bps=m['rmse_bps'],
                    rank_ic=m['rank_ic'],bias_log_bps=m['bias_bps'],mae_gain_vs_prior_selected=ref['mae_bps']-m['mae_bps'],rmse_gain_vs_prior_selected=ref['rmse_bps']-m['rmse_bps']))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    metrics[metrics.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby(['target','model'])[['mae_log_bps','rmse_log_bps']].mean().to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    focus=['prior_mean','prior_median','prior_fixed_ridge','prior_ridge','prior_tree','prior_e2','prior_selected','hgb_rmse','hgb_mae','selected_rmse','selected_mae']
    summary='48H HISTOGRAM BOOSTING EXCURSIONS - LOG BPS\nSeparate prespecified RMSE/MAE policies; exposed historical evaluation.\n'+metrics[metrics.model.isin(focus)].to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows_per_target=len(dev),prediction_rows=len(result),replay_passed=True,
        prediction_sha256=b.sha(out/'predictions.csv'),signature=signature))
    print(summary+'\nSaved to '+str(out),flush=True)


if __name__=='__main__':main()
