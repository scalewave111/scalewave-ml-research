r"""User-run constrained LightGBM and Ridge-residual mean experiment.
python .\Code\tune_lightgbm_residual_48h.py
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import fit_report_hybrid_means_48h as data
b = data.b
HERE = Path(__file__).resolve().parent
RIDGE_COLS = {'UP':[b.COLS[i] for i in [0,1,3]], 'DOWN':[b.COLS[i] for i in [0,1,3,4,5,6,7]]}


def lightgbm():
    sys.path.insert(0,str(HERE/'tree_model_deps'))
    import lightgbm as lgb
    b.check(hasattr(lgb,'LGBMRegressor'),'LightGBM installation unreadable')
    return lgb


def engineer(raw):
    # Reindex to elapsed half-hour periods so gaps never shorten a rolling window.
    grid = pd.date_range(raw.index.min(),raw.index.max(),freq='30min',tz='UTC')
    b.check(raw.index.isin(grid).all(),'Inputs must lie on half-hour grid')
    z = raw.reindex(grid).shift(1)
    result = {f'lag1__{c}':z[c] for c in raw.columns}
    for c in raw.columns:
        means = {}
        for w in [7,14,30]:
            roll = z[c].rolling(w,min_periods=w)
            mean,sd = roll.mean(),roll.std(ddof=0)
            means[w] = mean
            result[f'std{w}__{c}'] = sd
            result[f'range{w}__{c}'] = roll.max()-roll.min()
            result[f'iqr{w}__{c}'] = roll.quantile(.75)-roll.quantile(.25)
            result[f'zscore{w}__{c}'] = (z[c]-mean)/sd.where(sd>1e-12)
        for short in [7,14]:
            result[f'ma_ratio{short}_30__{c}'] = means[short]/means[30].where(means[30].abs()>1e-8)
    result = pd.DataFrame(result,index=grid).reindex(raw.index)
    # Calendar values are known at issuance; no outcome-derived regime labels.
    for day in range(7): result[f'weekday_{day}'] = (raw.index.dayofweek==day).astype(float)
    for week in range(1,6): result[f'week_of_month_{week}'] = (((raw.index.day-1)//7+1)==week).astype(float)
    for quarter in range(1,5): result[f'quarter_{quarter}'] = (raw.index.quarter==quarter).astype(float)
    return result.replace([np.inf,-np.inf],np.nan)


def prep(x,keys,cols):
    a=x.loc[keys,cols]; median=a.median().fillna(0); filled=a.fillna(median)
    return dict(columns=list(cols),median=median,center=filled.mean(),
        scale=filled.std(ddof=0).where(lambda s:s>1e-8,1.).fillna(1.))


def fit_ridge(x,target,keys,side):
    p=prep(x,keys,RIDGE_COLS[side])
    model=Ridge(alpha=.01*len(keys)).fit(b.transform(x.loc[keys],p),target.loc[keys])
    return dict(prep=p,model=model)


def ridge_predict(a,x):
    return a['model'].predict(b.transform(x,a['prep']))


def predict(a,x,features):
    rp=ridge_predict(a['ridge'],x)
    if a['mode']=='ridge': return np.maximum(0,rp)
    pred=a['model'].predict(b.transform(features,a['prep']))*a['scale']+a['center']
    if a['mode']=='residual': pred=rp+pred
    b.check(np.isfinite(pred).all(),'Nonfinite prediction')
    return np.maximum(0,pred)


def schedule():
    return [dict(id=f'd{d}_l{leaves}_r{rate}',max_depth=d,num_leaves=leaves,learning_rate=rate)
            for d,leaves in [(3,4),(3,6),(4,4),(4,8),(4,12)] for rate in [.01,.02,.03]]


def split_early(keys,y):
    # Last20% of eligible side-specific origins; purge48h before that slice.
    start=keys[int(.8*len(keys))]
    train=keys[y.loc[keys,'label_end']<start]
    valid=keys[keys>=start]
    b.check(len(train)>=100 and len(valid)>=50,'Insufficient early-stop history')
    b.check(y.loc[train,'label_end'].max()<valid.min(),'Early-stop leakage')
    return train,valid


def fit(x,features,y,target,keys,side,mode,cfg,threads):
    lgb=lightgbm(); inner,ev=split_early(keys,y)
    ridge=fit_ridge(x,target,inner,side)
    pp=prep(features,inner,features.columns)
    yy=target.loc[inner].to_numpy(float)
    vv=target.loc[ev].to_numpy(float)
    if mode=='residual':
        yy=yy-ridge_predict(ridge,x.loc[inner]); vv=vv-ridge_predict(ridge,x.loc[ev])
    center,scale=float(yy.mean()),max(float(yy.std()),1.)
    params={k:v for k,v in cfg.items() if k!='id'}
    params.update(objective='regression_l2',n_estimators=1000,min_child_samples=30,
        subsample=.75,subsample_freq=1,colsample_bytree=.75,reg_alpha=2.,reg_lambda=5.,
        random_state=48,n_jobs=threads,verbosity=-1,deterministic=True,force_col_wise=True)
    model=lgb.LGBMRegressor(**params)
    model.fit(b.transform(features.loc[inner],pp),(yy-center)/scale,
        eval_set=[(b.transform(features.loc[ev],pp),(vv-center)/scale)],eval_metric='rmse',
        callbacks=[lgb.early_stopping(50,verbose=False)])
    iterations=max(1,model.best_iteration_)
    # Refit on all matured training data with the independently chosen tree count.
    ridge=fit_ridge(x,target,keys,side); pp=prep(features,keys,features.columns)
    yy=target.loc[keys].to_numpy(float)
    if mode=='residual': yy=yy-ridge_predict(ridge,x.loc[keys])
    center,scale=float(yy.mean()),max(float(yy.std()),1.)
    params['n_estimators']=iterations
    model=lgb.LGBMRegressor(**params).fit(b.transform(features.loc[keys],pp),(yy-center)/scale)
    return dict(mode=mode,ridge=ridge,model=model,prep=pp,center=center,scale=scale,
        config=cfg,best_iteration=iterations,training_rows=len(keys),
        early_train_max_label_end=y.loc[inner,'label_end'].max(),early_valid_start=ev.min(),
        early_valid_rows=len(ev),max_training_label_end=y.loc[keys,'label_end'].max())


def load():
    seven,y,signed,ref=data.load_data()
    original,_,_,_=joblib.load(data.PREPARED)
    x=original.join(seven[[c for c in seven if c not in original]]).replace([np.inf,-np.inf],np.nan)
    features=engineer(x)
    cut=x.index[len(x)//2]
    pd.testing.assert_frame_equal(features.loc[:cut],engineer(x.loc[:cut]))
    changed=x.copy(); changed.loc[changed.index>=cut]=999999
    pd.testing.assert_series_equal(features.loc[cut],engineer(changed).loc[cut])
    return x,features,y,signed,ref


def self_test():
    idx=pd.date_range('2020-01-01',periods=1500,freq='30min',tz='UTC')
    rng=np.random.default_rng(48)
    x=pd.DataFrame(rng.normal(size=(len(idx),8)),index=idx,columns=b.COLS)
    f=engineer(x); cut=idx[1000]; changed=x.copy(); changed.loc[cut:]=1e6
    pd.testing.assert_frame_equal(f.loc[:cut],engineer(changed).loc[:cut])
    np.testing.assert_allclose(f.iloc[40]['std7__'+b.COLS[0]],x.iloc[33:40,0].std(ddof=0))
    y=pd.DataFrame({'label_end':idx+pd.Timedelta(hours=48)},index=idx)
    t=pd.Series(300+30*x.iloc[:,0]+10*x.iloc[:,1]**2,index=idx)
    for mode in ['direct','residual']:
        a=fit(x,f,y,t,idx[:1200],'UP',mode,schedule()[0],1)
        pred=predict(a,x.iloc[1200:],f.iloc[1200:])
        assert 1<=a['best_iteration']<=1000 and np.isfinite(pred).all()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.joblib';joblib.dump(a,p)
            np.testing.assert_allclose(pred,predict(joblib.load(p),x.iloc[1200:],f.iloc[1200:]))
    print('Synthetic causal windows, purge, direct/residual early stopping and replay passed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--check-data-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();b.check(args.threads>0,'Positive threads required')
    if args.self_test: self_test();return
    x,features,y,signed,ref=load();target=signed.abs()
    if args.check_data_only:
        print(f'Data/causality checks passed: {len(x)} rows, {len(features.columns)} tree inputs, {len(ref)} evaluation origins. No training.');return
    lgb=lightgbm()
    out=b.EXP/'lightgbm_residual_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    b.dump(out/'CONTRACT.json',dict(candidates=schedule(),raw_features=list(x),tree_features=list(features),
        target='Separate UP/DOWN conditional mean ordinary-bps magnitudes; fixed reference direction probabilities.',
        periods='One period=30minutes; shift1 before rolling7/14/30. Calendar known at origin. Regular grid preserves gaps.',
        constraints='Depth2 excluded because4<=leaves<2**depth impossible. Depth3/4, leaves4/6/8/12 valid by depth; min_child30; rate.01/.02/.03; subsample/colsample.75, freq1, L1=2,L2=5.',
        early_stop='Last20percent of eligible side training origins; purge preceding labels; patience50 max1000; refit full matured history at best iteration.',
        residual='Ridge alpha=.01*n on existing UP3/DOWN7 features, followed by LGBM on in-sample residuals. Early-stop residuals use only inner-fitted Ridge; final residuals use full-training Ridge.',
        selection='Three earlier windows, per-side pooled magnitude MSE. Tune direct and residual separately. Selected policy picks ridge/direct/residual using only earlier validation. No outer fallback switching.',
        scale='Affine target/residual normalization; penalties apply on normalized scale. Train-only preprocessing and nonnegative final magnitudes.',
        cycles='2024Q1,Q2(M4-6),Q3(M7-9),Q4(M10-12),Jan2025 tail. Pooled metrics and unweighted cycle2-4 mean metrics separately.',
        limitation='Exposed history; inherited raw feature availability; no guarantee of beating Ridge.',
        lightgbm_version=lgb.__version__,threads=args.threads,
        hashes={str(p):b.sha(p) for p in [Path(__file__),Path(b.__file__),Path(data.__file__),data.PREPARED,data.REF/'signed_return_predictions.csv',data.DATA/'X_train.parquet',data.DATA/'X_development.parquet']}))
    for p in [Path(__file__),Path(b.__file__),Path(data.__file__)]: (out/p.name).write_bytes(p.read_bytes())
    frames=[];ledger=[];selections=[];audits=[]
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            te=ref.index[ref.issue.eq(issue)]
            frame=ref.loc[te,['p_up','p_down','actual_signed_return_bps']].copy()
            frame['issue']=issue;frame['ridge_signed_bps']=ref.loc[te,'signed_return_forecast_bps']
            for side in ['UP','DOWN']:
                mask=signed>0 if side=='UP' else signed<0
                windows=[];ridge_sse=0.;count=0
                for offset in [9,6,3]:
                    cut=issue-pd.DateOffset(months=offset);tr=b.train_keys(y,mask,cut,None)
                    va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&mask]
                    b.check(len(va)>0 and y.loc[va,'label_end'].max()<issue,'Validation maturity failed')
                    rp=np.maximum(0,ridge_predict(fit_ridge(x,target,tr,side),x.loc[va]))
                    ridge_sse+=float(np.sum((rp-target.loc[va])**2));count+=len(va)
                    windows.append((tr,va))
                best={}
                for mode in ['direct','residual']:
                    scores=[]
                    for cfg in schedule():
                        print(f'{issue.date()} {side} {mode} {cfg["id"]}',flush=True)
                        loss=0.
                        for window,(tr,va) in enumerate(windows):
                            a=fit(x,features,y,target,tr,side,mode,cfg,args.threads)
                            m=b.metric(target.loc[va],predict(a,x.loc[va],features.loc[va]));loss+=m['mse_bps2']*len(va)
                            ledger.append(dict(issue=issue,side=side,mode=mode,candidate=cfg['id'],window=window,
                                best_iteration=a['best_iteration'],early_train_max_label_end=a['early_train_max_label_end'],early_valid_start=a['early_valid_start'],**m))
                        scores.append((loss/count,cfg))
                        pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv',index=False)
                    best[mode]=min(scores,key=lambda item:(item[0],item[1]['id']))
                policy=min(['ridge','direct','residual'],key=lambda mode:(ridge_sse/count if mode=='ridge' else best[mode][0],['ridge','direct','residual'].index(mode)))
                selections.append(dict(issue=issue,side=side,selected=policy,ridge_validation_mse=ridge_sse/count,best=best))
                b.dump(out/'SELECTIONS.json',selections)
                tr=b.train_keys(y,mask,issue,None)
                frame[f'{side}_ridge']=np.maximum(0,ridge_predict(fit_ridge(x,target,tr,side),x.loc[te]))
                for mode in ['direct','residual']:
                    a=fit(x,features,y,target,tr,side,mode,best[mode][1],args.threads)
                    a.update(issue=issue,side=side)
                    path=out/f'{issue:%Y-%m}_{side}_{mode}.joblib';joblib.dump(a,path,compress=3)
                    pred=predict(a,x.loc[te],features.loc[te])
                    np.testing.assert_allclose(pred,predict(joblib.load(path),x.loc[te],features.loc[te]),atol=1e-7,rtol=0)
                    frame[f'{side}_{mode}']=pred
                    audits.append(dict(issue=issue,side=side,mode=mode,artifact=path.name,sha256=b.sha(path),
                        best_iteration=a['best_iteration'],config=a['config'],training_rows=a['training_rows'],
                        max_training_label_end=a['max_training_label_end'],
                        early_train_max_label_end=a['early_train_max_label_end'],early_valid_start=a['early_valid_start'],replay_passed=True))
                    b.dump(out/'FIT_AUDIT.json',audits)
                frame[f'{side}_selected']=frame[f'{side}_{policy}']
            for mode in ['direct','residual','selected','ridge']:
                p=frame.p_up*frame[f'UP_{mode}']-frame.p_down*frame[f'DOWN_{mode}']
                if mode=='ridge': np.testing.assert_allclose(p,frame.ridge_signed_bps,atol=1e-6,rtol=0)
                else: frame[f'{mode}_signed_bps']=p
            frames.append(frame);frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
    result=pd.concat(frames).sort_index();b.check(result.index.equals(ref.index),'Final population differs')
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    metrics=[];side_metrics=[];comparisons=[]
    groups=[('ALL',result)]
    for cycle in range(1,5):
        groups.append((f'Cycle {cycle}',result[(result.index.year==2024)&(result.index.quarter==cycle)]))
    groups.append(('Jan2025 tail',result[result.index.year==2025]))
    for period,g in groups:
        if g.empty: continue
        baseline=b.metric(g.actual_signed_return_bps,g.ridge_signed_bps)
        for mode in ['ridge','direct','residual','selected']:
            pred=g[f'{mode}_signed_bps']; m=b.metric(g.actual_signed_return_bps,pred)
            metrics.append(dict(period=period,model=mode,**m,rmse_gain_vs_ridge=baseline['rmse_bps']-m['rmse_bps'],
                mae_gain_vs_ridge=baseline['mae_bps']-m['mae_bps'],direction_accuracy_pct=100*float((data.direction(pred)==data.direction(g.actual_signed_return_bps)).mean())))
            for side in ['UP','DOWN']:
                s=g[g.actual_signed_return_bps>0] if side=='UP' else g[g.actual_signed_return_bps<0]
                side_metrics.append(dict(period=period,side=side,model=mode,**b.metric(s.actual_signed_return_bps.abs(),s[f'{side}_{mode}'])))
    table=pd.DataFrame(metrics);table.to_csv(out/'cycle_signed_metrics.csv',index=False)
    pd.DataFrame(side_metrics).to_csv(out/'cycle_magnitude_metrics.csv',index=False)
    mean=table[table.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby('model')[['rmse_bps','mae_bps','rmse_gain_vs_ridge','mae_gain_vs_ridge']].mean()
    mean.to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    for mode in ['ridge','direct','residual','selected']:
        g=result[['actual_signed_return_bps']].copy();g['model']=mode
        g['predicted_return_bps']=result[f'{mode}_signed_bps'];g['predicted_return_pct']=g.predicted_return_bps/100;g['actual_return_pct']=g.actual_signed_return_bps/100
        g['predicted_direction']=data.direction(g.predicted_return_bps);g['actual_direction']=data.direction(g.actual_signed_return_bps)
        g['error_bps']=g.predicted_return_bps-g.actual_signed_return_bps;comparisons.append(g)
    pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    summary=table[['period','model','rmse_bps','mae_bps','rank_ic','rmse_gain_vs_ridge','mae_gain_vs_ridge']].to_string(index=False)+'\n\nUnweighted cycle2-4 mean:\n'+mean.to_string()
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),replay_passed=True,prediction_sha256=b.sha(out/'predictions.csv')))
    print(summary+'\nSaved to '+str(out),flush=True)


if __name__=='__main__': main()
