r"""Four controlled user-run LightGBM repair experiments; no fitting on import.
python .\Code\lightgbm_repair_experiments_48h.py
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tune_lightgbm_residual_48h as old
import discover_mean_magnitude_features_48h as price
import tree_interaction_signed_search_48h as direction_source
b=old.b
HERE=Path(__file__).resolve().parent
WEIGHTS=[0.,.1,.25,.5,1.]
PRIOR=b.EXP/'lightgbm_residual_48h_v1/20261113T050448624486Z'


def long_features(r,index):
    values={};level=r.fillna(0).cumsum()
    for h in [24,48,168,720]:
        kw=dict(window=f'{h}h',closed='left',min_periods=h*12)
        roll=r.rolling(**kw);valid=roll.count().eq(h*12)
        values[f'LONG_rv_{h}h']=np.sqrt(r.pow(2).rolling(**kw).sum())
        values[f'LONG_up_rv_{h}h']=np.sqrt(r.clip(lower=0).pow(2).rolling(**kw).sum())
        values[f'LONG_down_rv_{h}h']=np.sqrt(r.clip(upper=0).pow(2).rolling(**kw).sum())
        values[f'LONG_abs_net_{h}h']=roll.sum().abs()
        values[f'LONG_abs_sum_{h}h']=r.abs().rolling(**kw).sum()
        values[f'LONG_range_logbps_{h}h']=(level.rolling(**kw).max()-level.rolling(**kw).min()).where(valid)
    return pd.DataFrame(values).reindex(index)


def chronological_oof(x,y,target,side):
    """Monthly forward Ridge forecasts, always from strictly matured older labels."""
    side_mask=y.signed_return_log_bps>0 if side=='UP' else y.signed_return_log_bps<0
    months=x.index.tz_convert('UTC').tz_localize(None).to_period('M')
    residual=pd.Series(np.nan,index=x.index,dtype=float);audit=[]
    for month in months.unique():
        cut=month.start_time.tz_localize('UTC')
        tr=y.index[(y.label_end<cut)&side_mask]
        va=x.index[(months==month)&side_mask]
        if len(tr)<1024 or not len(va): continue
        ridge=old.fit_ridge(x,target,tr,side)
        residual.loc[va]=target.loc[va]-old.ridge_predict(ridge,x.loc[va])
        audit.append(dict(side=side,cutoff=cut,rows=len(va),training_rows=len(tr),
            max_training_label_end=y.loc[tr,'label_end'].max()))
        b.check(y.loc[tr,'label_end'].max()<va.min(),'OOF leakage')
    return residual,audit


def recipes():
    return [dict(id=f'{pack}_{days}_{cfg["id"]}',pack=pack,days=days,params=cfg)
        for pack,days in [('current',None),('long',None),('long',180),('long',365)]
        for cfg in old.schedule()]


def fit_candidate(features,residual,y,cut,cfg,threads,lgb):
    keys=y.index[(y.label_end<cut)&residual.notna()]
    if cfg['days'] is not None: keys=keys[keys>=cut-pd.Timedelta(days=cfg['days'])]
    b.check(len(keys)>=500,f'Insufficient OOF residuals: {cut}, {cfg["id"]}, {len(keys)}')
    tr,ev=old.split_early(keys,y)
    cols=[c for c in features if cfg['pack']=='long' or not c.startswith('LONG_')]
    pp=old.prep(features,tr,cols);yy=residual.loc[tr].to_numpy()
    center,scale=float(yy.mean()),max(float(yy.std()),1.)
    params={k:v for k,v in cfg['params'].items() if k!='id'}
    params.update(objective='regression_l2',n_estimators=1000,min_child_samples=30,
        subsample=.75,subsample_freq=1,colsample_bytree=.75,reg_alpha=2.,reg_lambda=5.,
        random_state=48,n_jobs=threads,verbosity=-1,deterministic=True,force_col_wise=True)
    model=lgb.LGBMRegressor(**params)
    vx=b.transform(features.loc[ev],pp);vy=(residual.loc[ev].to_numpy()-center)/scale
    evaluation=dict(eval_X=vx,eval_y=vy) if 'eval_X' in inspect.signature(model.fit).parameters else dict(eval_set=[(vx,vy)])
    model.fit(b.transform(features.loc[tr],pp),(yy-center)/scale,eval_metric='rmse',
        callbacks=[lgb.early_stopping(50,verbose=False)],**evaluation)
    iterations=max(1,model.best_iteration_)
    pp=old.prep(features,keys,cols);yy=residual.loc[keys].to_numpy()
    center,scale=float(yy.mean()),max(float(yy.std()),1.)
    params['n_estimators']=iterations
    model=lgb.LGBMRegressor(**params).fit(b.transform(features.loc[keys],pp),(yy-center)/scale)
    return dict(model=model,prep=pp,center=center,scale=scale,config=cfg,best_iteration=iterations,
        rows=len(keys),max_training_label_end=y.loc[keys,'label_end'].max(),
        early_train_max_label_end=y.loc[tr,'label_end'].max(),early_validation_start=ev.min())


def correction(a,features):
    result=a['model'].predict(b.transform(features,a['prep']))*a['scale']+a['center']
    b.check(np.isfinite(result).all(),'Nonfinite correction')
    return result


def joint_select(up,down,prob,truth,folds):
    """Matrices include Ridge at column0; accept pooled gain and >=2/3 fold wins."""
    pooled=direction_source.pair_mse(up,down,prob,truth)
    wins=np.zeros(pooled.shape,dtype=int)
    for k in np.unique(folds):
        mask=folds==k
        scores=direction_source.pair_mse(up[mask],down[mask],prob[mask],truth[mask])
        wins+=scores<scores[0,0]-1e-9
    eligible=(pooled<pooled[0,0]-1e-9)&(wins>=2)
    if not eligible.any(): return (0,0),dict(baseline_mse=float(pooled[0,0]),selected_mse=float(pooled[0,0]),fold_wins=0)
    choice=np.unravel_index(np.argmin(np.where(eligible,pooled,np.inf)),pooled.shape)
    return tuple(map(int,choice)),dict(baseline_mse=float(pooled[0,0]),selected_mse=float(pooled[choice]),fold_wins=int(wins[choice]))


def load_inputs():
    x,current,y,signed,ref=old.load()
    returnpath=price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=price.canonical.load_returns(returnpath);audit=price.canonical.validate(r,y)
    extra=long_features(r,x.index)
    cut=x.index[len(x)//2];points=x.index[x.index<=cut][-20:]
    pd.testing.assert_frame_equal(long_features(r,points),long_features(r.loc[:cut],points))
    changed=r.copy();changed.loc[changed.index>=cut]=99999
    pd.testing.assert_frame_equal(long_features(r,pd.DatetimeIndex([cut])),long_features(changed,pd.DatetimeIndex([cut])))
    probabilities,pa=direction_source.load_probabilities()
    common=y.index.intersection(probabilities.index)
    b.check(np.array_equal(probabilities.loc[common,'y'],(signed.loc[common]>0).astype(int)),'Direction truth mismatch')
    np.testing.assert_allclose(probabilities.loc[ref.index,'p_up'],ref.p_up,rtol=0,atol=1e-7)
    for issue in b.ISSUES:
        for offset in [9,6,3]:
            c=issue-pd.DateOffset(months=offset)
            keys=y.index[(y.index>=c)&(y.index<c+pd.DateOffset(months=2))]
            b.check(keys.isin(probabilities.index).all() and y.loc[keys,'label_end'].max()<issue,'Validation coverage/purge failed')
    return x,current.join(extra),y,signed,ref,probabilities,dict(price=audit,direction=pa)


def self_test():
    r=pd.Series(np.arange(10000,dtype=float)%13,index=pd.date_range('2020-01-01',periods=10000,freq='5min',tz='UTC'))
    cut=r.index[9000];modified=r.copy();modified.loc[cut:]=-9999
    pd.testing.assert_frame_equal(long_features(r,pd.DatetimeIndex([cut])),long_features(modified,pd.DatetimeIndex([cut])))
    window=r[(r.index>=cut-pd.Timedelta(hours=24))&(r.index<cut)]
    np.testing.assert_allclose(long_features(r,pd.DatetimeIndex([cut])).iloc[0]['LONG_rv_24h'],np.sqrt(np.sum(window**2)))
    rng=np.random.default_rng(48);up=rng.uniform(10,100,(90,3));down=rng.uniform(10,100,(90,4));p=rng.uniform(.2,.8,90)
    truth=p*up[:,1]-(1-p)*down[:,2];folds=np.repeat([0,1,2],30)
    assert joint_select(up,down,p,truth,folds)[0]==(1,2)
    assert joint_select(up,down,p,p*up[:,0]-(1-p)*down[:,0],folds)[0]==(0,0)
    idx=pd.date_range('2020-01-01',periods=1800,freq='D',tz='UTC')
    x=pd.DataFrame(rng.normal(size=(len(idx),8)),columns=b.COLS,index=idx)
    target=pd.Series(300+10*x.iloc[:,0],index=idx)
    y=pd.DataFrame(dict(label_end=idx+pd.Timedelta(hours=48),signed_return_log_bps=target),index=idx)
    residual,audit=chronological_oof(x,y,target,'UP');assert residual.notna().sum()>500
    for row in audit: assert row['max_training_label_end']<row['cutoff']
    changed=target.copy();changed.iloc[1500:]=1e6
    again,_=chronological_oof(x.iloc[:1500],y.iloc[:1500],changed.iloc[:1500],'UP')
    pd.testing.assert_series_equal(residual.iloc[:1500],again)
    cfg=recipes()[0];model=fit_candidate(x,residual,y,idx[-1]+pd.Timedelta(days=3),cfg,1,old.lightgbm())
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'a.joblib';joblib.dump(model,path)
        np.testing.assert_allclose(correction(model,x.iloc[-30:]),correction(joblib.load(path),x.iloc[-30:]))
    print('Passed causal price windows, monthly OOF label maturity, prefix invariance, joint selection/fallback, early stopping and model replay.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--check-data-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();b.check(args.threads>0,'Positive threads required')
    if args.self_test:self_test();return
    x,features,y,signed,ref,probabilities,input_audit=load_inputs();target=signed.abs()
    if args.check_data_only:
        print(f'PASS: {len(x)} input rows, {len(features.columns)} features, {len(ref)} evaluation origins; canonical labels, causal features and earlier probabilities checked. No fitting.');return
    lgb=old.lightgbm();configs=recipes()
    files=[Path(__file__),Path(old.__file__),Path(old.data.__file__),Path(b.__file__),Path(price.__file__),Path(price.canonical.__file__),Path(direction_source.__file__),Path(direction_source.old.__file__),
        old.data.PREPARED,old.data.DATA/'X_train.parquet',old.data.DATA/'X_development.parquet',old.data.REF/'signed_return_predictions.csv',
        price.WEEK1/'Dataset/canonical_returns_5m.parquet',direction_source.DIR/'COMPLETE.json',
        direction_source.DIR/'validation_predictions.csv',direction_source.DIR/'development_predictions.csv',PRIOR/'predictions.csv',PRIOR/'COMPLETE.json']
    identity=dict(hashes={str(p):b.sha(p) for p in files},configs=configs,weights=WEIGHTS,threads=args.threads,lightgbm_version=lgb.__version__)
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'lightgbm_repair_experiments_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not (out/'COMPLETE.json').exists(),'Run already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature']==signature,'Resume code/input/config signature mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,identity=identity,audit=input_audit,
            experiments=['E1 current features + monthly OOF residuals, expanding','E2 add24h/48h/7d/30d price inputs','E3 select residual history180d/365d/expanding','E4 joint signed MSE + weights + >=2/3 earlier window wins'],
            ridge='Always expanding fixed UP3/DOWN7 alpha=.01*n; history selection changes correction history only, preserving weight0 Ridge benchmark.',
            oof='Monthly Ridge trained on label_end<month start; side residuals y-raw_Ridge; unavailable warmup excluded; each residual used only after its48h label matures.',
            early_stop='Latest20percent residual origins; purge48h labels, patience50,max1000; all mature residuals refit at selected count.',
            scale='Affine residual normalization; fixed L1=2/L2=5 on normalized scale. No target clipping. Final magnitude=max(0,Ridge+w*correction).',
            selection='E1-3 side-specific pooled conditional MSE over three earlier windows; E4 jointly selects histories,configs,weights on identical signed rows; pooled gain and2/3 wins; otherwise exact Ridge.',
            limitation='Exposed historical evaluation; exploratory stability gate not statistical proof; no guarantee; baseline compared on same rows.'))
        for p in files:
            if p.suffix=='.py':(out/p.name).write_bytes(p.read_bytes())
    cache=out/'cache';cache.mkdir(exist_ok=True)
    def cached(name,compute):
        path=cache/(name+'.joblib');receipt=cache/(name+'.json')
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):return joblib.load(path)
        value=compute();joblib.dump(value,path,compress=3);b.dump(receipt,dict(sha256=b.sha(path)));return value
    residuals={};oof_audit=[]
    for side in ['UP','DOWN']:
        residuals[side],a=cached('OOF_'+side,lambda side=side:chronological_oof(x,y,target,side));oof_audit.extend(a)
    b.dump(out/'OOF_AUDIT.json',oof_audit)
    selections=[];ledger=[];frames=[]
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            windows=[]
            for offset in [9,6,3]:
                cut=issue-pd.DateOffset(months=offset)
                va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))]
                windows.append((cut,va))
            vk=windows[0][1].append([v[1] for v in windows[1:]])
            foldids=np.concatenate([np.repeat(i,len(v)) for i,(_,v) in enumerate(windows)])
            te=ref.index[ref.issue.eq(issue)]
            frame=ref.loc[te,['p_up','p_down','actual_signed_return_bps']].copy();frame['issue']=issue
            matrices={};metas={};bases={};outer_ridge={};choices={}
            for side in ['UP','DOWN']:
                mask=signed>0 if side=='UP' else signed<0
                def ridge_at(cut):
                    return cached(f'Ridge_{side}_{cut:%Y%m%d}',lambda:old.fit_ridge(x,target,b.train_keys(y,mask,cut,None),side))
                rawbase=np.concatenate([old.ridge_predict(ridge_at(cut),x.loc[va]) for cut,va in windows])
                outer_ridge[side]=ridge_at(issue);bases[side]=rawbase
                matrix=[];scores=[]
                for cfg in configs:
                    name=f'{side}_{cfg["id"]}_{issue:%Y%m%d}_validation'
                    def make_validation(cfg=cfg):
                        pieces=[]
                        for cut,va in windows:
                            a=cached(f'Model_{side}_{cfg["id"]}_{cut:%Y%m%d}',lambda cut=cut:fit_candidate(features,residuals[side],y,cut,cfg,args.threads,lgb))
                            pieces.append(correction(a,features.loc[va]))
                        return np.concatenate(pieces)
                    delta=cached(name,make_validation);matrix.append(delta)
                    take=mask.loc[vk].to_numpy();m=b.metric(target.loc[vk[take]],np.maximum(0,rawbase[take]+delta[take]))
                    scores.append(m['mse_bps2']);ledger.append(dict(issue=issue,side=side,candidate=cfg['id'],**m))
                    print(issue.date(),side,cfg['id'],flush=True)
                    pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv',index=False)
                correction_matrix=np.column_stack(matrix)
                for stage in ['E1','E2','E3']:
                    eligible=[i for i,c in enumerate(configs) if (stage=='E1' and c['pack']=='current') or (stage=='E2' and c['pack']=='long' and c['days'] is None) or (stage=='E3' and c['pack']=='long')]
                    i=min(eligible,key=lambda i:(scores[i],configs[i]['id']))
                    choices[(side,stage)]=(i,1.)
                values=[np.maximum(0,rawbase)];meta=[(None,0.)]
                for i,c in enumerate(configs):
                    if c['pack']!='long':continue
                    for weight in WEIGHTS[1:]:
                        values.append(np.maximum(0,rawbase+weight*correction_matrix[:,i]));meta.append((i,weight))
                matrices[side]=np.column_stack(values);metas[side]=meta
            pair,gate=joint_select(matrices['UP'],matrices['DOWN'],probabilities.loc[vk,'p_up'].to_numpy(),signed.loc[vk].to_numpy(),foldids)
            for side,chosen in zip(['UP','DOWN'],pair):choices[(side,'E4')]=metas[side][chosen]
            for side in ['UP','DOWN']:
                rp=old.ridge_predict(outer_ridge[side],x.loc[te]);frame[f'{side}_ridge']=np.maximum(0,rp)
                for stage in ['E1','E2','E3','E4']:
                    i,weight=choices[(side,stage)]
                    a=None
                    if i is not None:
                        cfg=configs[i]
                        a=cached(f'Model_{side}_{cfg["id"]}_{issue:%Y%m%d}',lambda:fit_candidate(features,residuals[side],y,issue,cfg,args.threads,lgb))
                        pred=np.maximum(0,rp+weight*correction(a,features.loc[te]))
                    else:pred=np.maximum(0,rp)
                    bundle=dict(ridge=outer_ridge[side],correction=a,weight=weight,issue=issue,side=side,stage=stage)
                    path=out/f'{issue:%Y-%m}_{side}_{stage}.joblib';joblib.dump(bundle,path,compress=3)
                    replay=joblib.load(path);rr=old.ridge_predict(replay['ridge'],x.loc[te])
                    if replay['correction'] is not None:rr+=replay['weight']*correction(replay['correction'],features.loc[te])
                    np.testing.assert_allclose(pred,np.maximum(0,rr),rtol=0,atol=1e-7)
                    frame[f'{side}_{stage}']=pred
                    selections.append(dict(issue=issue,side=side,stage=stage,candidate=configs[i] if i is not None else 'ridge',weight=weight,
                        gate=gate if stage=='E4' else None,artifact=path.name,sha256=b.sha(path),best_iteration=a['best_iteration'] if a else 0,
                        residual_max_label_end=a['max_training_label_end'] if a else None,replay_passed=True))
            for stage in ['ridge','E1','E2','E3','E4']:
                frame[f'{stage}_signed_bps']=frame.p_up*frame[f'UP_{stage}']-frame.p_down*frame[f'DOWN_{stage}']
            np.testing.assert_allclose(frame.ridge_signed_bps,ref.loc[te,'signed_return_forecast_bps'],rtol=0,atol=1e-6)
            frames.append(frame);frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
            b.dump(out/'SELECTIONS_AND_FIT_AUDIT.json',selections)
    result=pd.concat(frames).sort_index();b.check(result.index.equals(ref.index),'Final timestamps mismatch')
    prior_done=json.loads((PRIOR/'COMPLETE.json').read_text());b.check(b.sha(PRIOR/'predictions.csv')==prior_done['prediction_sha256'],'Prior predictions changed')
    previous=pd.read_csv(PRIOR/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True)
    b.check(previous.index.equals(result.index),'Prior timestamps differ')
    np.testing.assert_allclose(previous.actual_signed_return_bps,result.actual_signed_return_bps,atol=1e-7,rtol=0)
    np.testing.assert_allclose(previous.p_up,result.p_up,atol=1e-7,rtol=0)
    for mode in ['direct','residual']:result[f'old_{mode}_signed_bps']=previous[f'{mode}_signed_bps']
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    rows=[];tails=[];sides=[];comparison=[]
    groups=[('ALL',result)]+[(f'Cycle {q}',result[(result.index.year==2024)&(result.index.quarter==q)]) for q in range(1,5)]+[('Jan2025',result[result.index.year==2025])]
    for period,g in groups:
        if g.empty:continue
        actual=g.actual_signed_return_bps;baseline=b.metric(actual,g.ridge_signed_bps)
        # Tail threshold is available at each issuance, not selected on its outcomes.
        thresholds=pd.Series(index=g.index,dtype=float)
        for issue,part in g.groupby('issue'):thresholds.loc[part.index]=target.loc[y.label_end<issue].quantile(.9)
        tail=actual.abs()>=thresholds
        for stage in ['ridge','old_direct','old_residual','E1','E2','E3','E4']:
            pred=g[f'{stage}_signed_bps'];m=b.metric(actual,pred)
            rows.append(dict(period=period,model=stage,**m,rmse_gain_vs_ridge=baseline['rmse_bps']-m['rmse_bps'],mae_gain_vs_ridge=baseline['mae_bps']-m['mae_bps'],direction_accuracy_pct=float(100*(old.data.direction(pred)==old.data.direction(actual)).mean())))
            excess=(pred-actual)**2-(g.ridge_signed_bps-actual)**2
            tails.append(dict(period=period,model=stage,tail_rows=int(tail.sum()),excess_mse=float(excess.mean()),tail_excess_contribution=float(excess[tail].sum()/len(g)),nontail_excess_contribution=float(excess[~tail].sum()/len(g))))
            if stage in ['ridge','E1','E2','E3','E4']:
                for side in ['UP','DOWN']:
                    mask=actual>0 if side=='UP' else actual<0
                    sides.append(dict(period=period,side=side,model=stage,**b.metric(actual[mask].abs(),g.loc[mask,f'{side}_{stage}'])))
    for stage in ['ridge','old_direct','old_residual','E1','E2','E3','E4']:
        g=result[['actual_signed_return_bps']].copy();g['model']=stage;g['predicted_return_bps']=result[f'{stage}_signed_bps']
        g['actual_return_pct']=g.actual_signed_return_bps/100;g['predicted_return_pct']=g.predicted_return_bps/100
        g['actual_direction']=old.data.direction(g.actual_signed_return_bps);g['predicted_direction']=old.data.direction(g.predicted_return_bps)
        g['error_bps']=g.predicted_return_bps-g.actual_signed_return_bps;comparison.append(g)
    pd.concat(comparison).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    table=pd.DataFrame(rows);table.to_csv(out/'cycle_signed_metrics.csv',index=False)
    pd.DataFrame(sides).to_csv(out/'cycle_magnitude_metrics.csv',index=False);pd.DataFrame(tails).to_csv(out/'tail_error_contributions.csv',index=False)
    mean=table[table.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby('model')[['rmse_bps','mae_bps','rmse_gain_vs_ridge','mae_gain_vs_ridge']].mean()
    mean.to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    summary='EXPOSED HISTORICAL EVALUATION; no guaranteed gain. E4 prespecified validation gate, no outer promotion.\n'+table[['period','model','rmse_bps','mae_bps','rank_ic','rmse_gain_vs_ridge','mae_gain_vs_ridge']].to_string(index=False)+'\n\nCycle2-4 unweighted mean:\n'+mean.to_string()
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=b.sha(out/'predictions.csv'),replay_passed=True,signature=signature))
    print(summary+'\nSaved to '+str(out))


if __name__=='__main__':main()
