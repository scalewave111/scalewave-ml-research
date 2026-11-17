r"""Independent changes to unchanged E2: horizons, capacity, seed ensemble.
python .\Code\tune_e2_direct_controls_48h.py
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
import lightgbm_repair_experiments_48h as repair
old,b=repair.old,repair.b
HERE=Path(__file__).resolve().parent
BASE=b.EXP/'lightgbm_repair_experiments_48h_v1/20261113T053731575845Z'
SEEDS=[48,148,248]


def added_features(r,index):
    values={};level=r.fillna(0).cumsum()
    for h in [4,12,72]:
        kw=dict(window=f'{h}h',closed='left',min_periods=h*12)
        roll=r.rolling(**kw);valid=roll.count().eq(h*12)
        values[f'ADD_rv_{h}h']=np.sqrt(r.pow(2).rolling(**kw).sum())
        values[f'ADD_up_rv_{h}h']=np.sqrt(r.clip(lower=0).pow(2).rolling(**kw).sum())
        values[f'ADD_down_rv_{h}h']=np.sqrt(r.clip(upper=0).pow(2).rolling(**kw).sum())
        values[f'ADD_abs_net_{h}h']=roll.sum().abs()
        values[f'ADD_abs_sum_{h}h']=r.abs().rolling(**kw).sum()
        values[f'ADD_range_logbps_{h}h']=(level.rolling(**kw).max()-level.rolling(**kw).min()).where(valid)
    return pd.DataFrame(values).reindex(index)


def ablations(columns):
    predicates={
        'drop_calendar':lambda c:c.startswith(('weekday_','week_of_month_','quarter_')),
        'drop_short_rolling':lambda c:c.startswith(('std','range','iqr','zscore','ma_ratio')),
        'drop_long_total_vol':lambda c:c.startswith('LONG_rv_'),
        'drop_long_directional_vol':lambda c:c.startswith(('LONG_up_rv_','LONG_down_rv_')),
        'drop_long_absolute_returns':lambda c:c.startswith(('LONG_abs_net_','LONG_abs_sum_')),
        'drop_long_range':lambda c:c.startswith('LONG_range_')}
    for h in [24,48,168,720]:predicates[f'drop_long_{h}h']=lambda c,h=h:c.startswith('LONG_') and c.endswith(f'_{h}h')
    result={'unchanged':list(columns)}
    for name,predicate in predicates.items():
        result[name]=[c for c in columns if not predicate(c)]
        b.check(0<len(result[name])<len(columns),f'Empty/no-op mask: {name}')
    return result


def direct_specs(columns,all_columns,base_params):
    specs=[dict(name='control_unchanged',columns=list(columns),params=dict(base_params),seeds=[48])]
    for h in [4,12,72]:
        extra=[c for c in all_columns if c.startswith('ADD_') and c.endswith(f'_{h}h')]
        b.check(len(extra)==6,f'Expected six added features for{h}h')
        specs.append(dict(name=f'H_add_{h}h',columns=list(columns)+extra,params=dict(base_params),seeds=[48]))
    for depth,leaves in [(3,4),(3,6),(4,8),(4,12)]:
        for leaf in [20,30,50]:
            specs.append(dict(name=f'C_d{depth}_l{leaves}_child{leaf}',columns=list(columns),
                params=dict(base_params,max_depth=depth,num_leaves=leaves,min_child_samples=leaf),seeds=[48]))
    specs.append(dict(name='S_three_seeds',columns=list(columns),params=dict(base_params),seeds=SEEDS))
    for spec in specs:
        b.check(set(columns).issubset(spec['columns']),'Original E2 feature removed')
        if not spec['name'].startswith('H_add_'): b.check(spec['columns']==list(columns),'Non-horizon branch changed inputs')
        if not spec['name'].startswith('C_d'): b.check(spec['params']==base_params,'Non-capacity branch changed settings')
    return specs


def fit(features,residual,y,cut,columns,params,seed,threads,lgb):
    keys=y.index[(y.label_end<cut)&residual.notna()]
    b.check(len(keys)>=500,'Insufficient OOF history')
    tr,ev=old.split_early(keys,y);pp=old.prep(features,tr,columns)
    yy=residual.loc[tr].to_numpy();center,scale=float(yy.mean()),max(float(yy.std()),1.)
    cfg=dict(objective='regression_l2',n_estimators=1000,subsample=.75,subsample_freq=1,
        colsample_bytree=.75,reg_alpha=2.,reg_lambda=5.,random_state=seed,n_jobs=threads,
        verbosity=-1,deterministic=True,force_col_wise=True,**params)
    model=lgb.LGBMRegressor(**cfg)
    vx=b.transform(features.loc[ev],pp);vy=(residual.loc[ev].to_numpy()-center)/scale
    evaluation=dict(eval_X=vx,eval_y=vy) if 'eval_X' in inspect.signature(model.fit).parameters else dict(eval_set=[(vx,vy)])
    model.fit(b.transform(features.loc[tr],pp),(yy-center)/scale,eval_metric='rmse',
        callbacks=[lgb.early_stopping(50,verbose=False)],**evaluation)
    count=max(1,model.best_iteration_);cfg['n_estimators']=count
    pp=old.prep(features,keys,columns);yy=residual.loc[keys].to_numpy()
    center,scale=float(yy.mean()),max(float(yy.std()),1.)
    model=lgb.LGBMRegressor(**cfg).fit(b.transform(features.loc[keys],pp),(yy-center)/scale)
    return dict(model=model,prep=pp,center=center,scale=scale,params=params,seed=seed,
        best_iteration=count,training_rows=len(keys),max_training_label_end=y.loc[keys,'label_end'].max(),
        early_train_max_label_end=y.loc[tr,'label_end'].max(),early_validation_start=ev.min())


def predict(bundle,x,features):
    raw=old.ridge_predict(bundle['ridge'],x)
    return np.mean([np.maximum(0,raw+repair.correction(a,features)) for a in bundle['models']],axis=0)


def choose(entries):
    # Deterministic simplicity tie break; no access to outer outcomes.
    return min(entries,key=lambda e:(e['validation_mse'],len(e['columns']),len(e['seeds']),e['name']))


def load():
    x,features,y,signed,ref,_,audit=repair.load_inputs()
    path=repair.price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=repair.price.canonical.load_returns(path);extra=added_features(r,x.index)
    cut=x.index[len(x)//2];points=x.index[x.index<=cut][-20:]
    pd.testing.assert_frame_equal(added_features(r,points),added_features(r.loc[:cut],points))
    changed=r.copy();changed.loc[changed.index>=cut]=99999
    pd.testing.assert_frame_equal(added_features(r,pd.DatetimeIndex([cut])),added_features(changed,pd.DatetimeIndex([cut])))
    done=json.loads((BASE/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(BASE/'predictions.csv')==done['prediction_sha256'],'E2 reference integrity failed')
    reference=pd.read_csv(BASE/'predictions.csv',index_col='decision_time');reference.index=pd.to_datetime(reference.index,utc=True)
    reference['issue']=pd.to_datetime(reference.issue,utc=True)
    b.check(reference.index.equals(ref.index),'E2 origins differ')
    for c in ['p_up','p_down','actual_signed_return_bps']:
        np.testing.assert_allclose(reference[c],ref[c],rtol=0,atol=1e-7)
    entries=[a for a in json.loads((BASE/'SELECTIONS_AND_FIT_AUDIT.json').read_text()) if a['stage']=='E2']
    b.check(len(entries)==8,'Missing E2 model configurations')
    for a in entries:b.check(b.sha(BASE/a['artifact'])==a['sha256'],'E2 model hash differs')
    return x,features.join(extra),y,signed,reference,entries,audit


def self_test():
    rng=np.random.default_rng(48);idx=pd.date_range('2020-01-01',periods=13000,freq='5min',tz='UTC')
    r=pd.Series(rng.normal(size=len(idx)),index=idx);cut=idx[12000]
    altered=r.copy();altered.loc[cut:]=10000
    pd.testing.assert_frame_equal(added_features(r,pd.DatetimeIndex([cut])),added_features(altered,pd.DatetimeIndex([cut])))
    np.testing.assert_allclose(added_features(r,pd.DatetimeIndex([cut])).iloc[0]['ADD_abs_sum_4h'],r.iloc[11952:12000].abs().sum())
    names=['lag1__x','std7__x','weekday_1']+[f'LONG_{s}_{h}h' for s in ['rv','up_rv','down_rv','abs_net','abs_sum','range_logbps'] for h in [24,48,168,720]]
    masks=ablations(names);assert len(masks)==11 and len(masks['drop_long_24h'])==len(names)-6
    idx=pd.date_range('2020-01-01',periods=1500,freq='30min',tz='UTC')
    f=pd.DataFrame(rng.normal(size=(len(idx),8)),index=idx,columns=b.COLS)
    y=pd.DataFrame({'label_end':idx+pd.Timedelta(hours=48)},index=idx)
    target=pd.Series(300+10*f.iloc[:,0],index=idx);residual=pd.Series(rng.normal(size=len(idx)),index=idx)
    params=dict(max_depth=3,num_leaves=4,learning_rate=.02,min_child_samples=30)
    specs=direct_specs(list(f),list(f)+list(added_features(r,pd.DatetimeIndex([cut])).columns),params)
    assert len(specs)==17 and all(set(f).issubset(v['columns']) for v in specs)
    assert next(v for v in specs if v['name']=='S_three_seeds')['params']==params
    a=fit(f,residual,y,idx[-1]+pd.Timedelta(days=3),list(f),params,48,1,old.lightgbm())
    ridge=old.fit_ridge(f,target,idx[:1200],'UP');bundle=dict(ridge=ridge,models=[a,a,a])
    single=predict(dict(ridge=ridge,models=[a]),f.iloc[-20:],f.iloc[-20:])
    np.testing.assert_allclose(single,predict(bundle,f.iloc[-20:],f.iloc[-20:]))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.joblib';joblib.dump(bundle,path)
        np.testing.assert_allclose(single,predict(joblib.load(path),f.iloc[-20:],f.iloc[-20:]))
    print('Passed independent branch invariants, retained inputs/settings, causal horizons, purge, seed averaging and replay.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--resume',type=Path)
    p.add_argument('--check-data-only',action='store_true');p.add_argument('--self-test',action='store_true')
    args=p.parse_args();b.check(args.threads>0,'Positive threads required')
    if args.self_test:self_test();return
    x,features,y,signed,reference,base_entries,input_audit=load();target=signed.abs()
    baseline_columns=[c for c in features if not c.startswith('ADD_')]
    b.check(len(baseline_columns)==235,'E2 baseline must retain235 columns')
    if args.check_data_only:
        print(f'PASS: {len(x)} rows, E2={len(baseline_columns)} features, expanded={len(features.columns)}, {len(reference)} origins, 3 independent experiment branches; all235 E2 columns retained. No fitting.');return
    lgb=old.lightgbm()
    paths=[Path(__file__),Path(repair.__file__),Path(old.__file__),Path(old.data.__file__),Path(b.__file__),Path(repair.price.__file__),Path(repair.price.canonical.__file__),Path(repair.direction_source.__file__),
        old.data.PREPARED,old.data.DATA/'X_train.parquet',old.data.DATA/'X_development.parquet',repair.price.WEEK1/'Dataset/canonical_returns_5m.parquet',
        BASE/'COMPLETE.json',BASE/'predictions.csv',BASE/'SELECTIONS_AND_FIT_AUDIT.json']+[BASE/a['artifact'] for a in base_entries]
    identity=dict(hashes={str(q):b.sha(q) for q in paths},threads=args.threads,lgb_version=lgb.__version__,seeds=SEEDS)
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'e2_direct_controls_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not (out/'COMPLETE.json').exists(),'Already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature']==signature,'Resume signature mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,identity=identity,input_audit=input_audit,baseline_columns=baseline_columns,
            plan='Independent branches from unchanged E2: H add4h/12h/72h separately; C12capacity settings; S average seeds48/148/248. All235inputs always retained. No branch consumes another branch selection.',
            selection='Separate branch selectors use per-side pooled conditional-magnitude MSE on three earlier windows, each includes unchanged E2. Branches independent, no outer selection or global winner promotion. Validation still historically exposed.',
            fixed='Expanding UP3/DOWN7 Ridge, expanding monthly chronological OOF residuals, fixed direction; full-strength correction. All core LightGBM settings unchanged except declared capacity and seeds.',
            baseline_params='Frozen E2 side/issue parameters from earlier validation. Horizons and seed averaging keep every E2 setting; only capacity branch changes depth/leaves/min_child. Learning rate fixed at E2 value.',
            tail='Per-issuance matured-training absolute magnitude P90; tail and nontail excess MSE versus E2.',
            limitation='Repeatedly exposed historical evaluation; model selection rules prespecified, no guarantee or automatic outer winner promotion.'))
        for q in paths:
            if q.suffix=='.py':(out/q.name).write_bytes(q.read_bytes())
    print('Output:',out,flush=True)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    def cached(key,fn):
        name=hashlib.sha256(key.encode()).hexdigest();path=cache/(name+'.joblib');receipt=cache/(name+'.json')
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):return joblib.load(path)
        value=fn();joblib.dump(value,path,compress=3);b.dump(receipt,dict(key=key,sha256=b.sha(path)));return value
    residuals={};oa=[]
    for side in ['UP','DOWN']:
        residuals[side],a=cached('OOF_'+side,lambda side=side:repair.chronological_oof(x,y,target,side));oa.extend(a)
    b.dump(out/'OOF_AUDIT.json',oa)
    frames=[];validations=[];selections=[];fit_audit=[]
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            te=reference.index[reference.issue.eq(issue)]
            frame=reference.loc[te,['p_up','p_down','actual_signed_return_bps']].copy();frame['issue']=issue
            for side in ['UP','DOWN']:
                entry=next(a for a in base_entries if pd.Timestamp(a['issue'])==issue and a['side']==side)
                original=joblib.load(BASE/entry['artifact']);ridge=original['ridge']
                baseline=np.maximum(0,old.ridge_predict(ridge,x.loc[te])+repair.correction(original['correction'],features.loc[te]))
                np.testing.assert_allclose(baseline,reference.loc[te,f'{side}_E2'],rtol=0,atol=1e-7)
                frame[f'{side}_E2']=baseline;frame[f'{side}_ridge']=np.maximum(0,old.ridge_predict(ridge,x.loc[te]))
                base_params={k:v for k,v in entry['candidate']['params'].items() if k!='id'};base_params['min_child_samples']=30
                mask=signed>0 if side=='UP' else signed<0
                windows=[]
                for offset in [9,6,3]:
                    cut=issue-pd.DateOffset(months=offset);va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&mask]
                    b.check(len(va)>0 and y.loc[va,'label_end'].max()<issue,'Validation label maturity failed')
                    rk=b.train_keys(y,mask,cut,None)
                    rr=cached(f'Ridge_{side}_{cut}',lambda rk=rk:old.fit_ridge(x,target,rk,side))
                    windows.append((cut,va,old.ridge_predict(rr,x.loc[va])))
                def get_model(cut,cols,params,seed):
                    key=json.dumps(dict(side=side,cut=str(cut),columns=cols,params=params,seed=seed),sort_keys=True)
                    return cached('model_'+key,lambda:fit(features,residuals[side],y,cut,cols,params,seed,args.threads,lgb))
                evaluated=[]
                def evaluate(name,cols,params,seeds):
                    loss=0.;count=0
                    for n,(cut,va,rp) in enumerate(windows):
                        preds=[];trees=[]
                        for seed in seeds:
                            a=get_model(cut,cols,params,seed);trees.append(a['best_iteration'])
                            preds.append(np.maximum(0,rp+repair.correction(a,features.loc[va])))
                        m=b.metric(target.loc[va],np.mean(preds,axis=0));loss+=m['mse_bps2']*len(va);count+=len(va)
                        validations.append(dict(issue=issue,side=side,experiment=name,window=n,iterations=trees,**m))
                    result=dict(name=name,columns=cols,params=params,seeds=seeds,validation_mse=loss/count)
                    evaluated.append(result);pd.DataFrame(validations).to_csv(out/'validation_metrics.csv',index=False)
                    print(issue.date(),side,name,'validation RMSE',round(np.sqrt(loss/count),3),flush=True)
                    return result
                specs=direct_specs(baseline_columns,list(features),base_params)
                results={spec['name']:evaluate(spec['name'],spec['columns'],spec['params'],spec['seeds']) for spec in specs}
                control=results['control_unchanged']
                horizon=[control]+[v for k,v in results.items() if k.startswith('H_add_')]
                capacity=[control]+[v for k,v in results.items() if k.startswith('C_d')]
                seeds=[control,results['S_three_seeds']]
                policies={'H_selected':choose(horizon),'C_selected':choose(capacity),'S_selected':choose(seeds)}
                for name,choice in policies.items():selections.append(dict(issue=issue,side=side,stage=name,choice=choice))
                # Evaluate every declared variant as well as the four earlier-selected policies.
                for experiment in evaluated+[dict(v,name=k) for k,v in policies.items()]:
                    models=[get_model(issue,experiment['columns'],experiment['params'],seed) for seed in experiment['seeds']]
                    bundle=dict(ridge=ridge,models=models,issue=issue,side=side,experiment=experiment)
                    path=out/f'{issue:%Y-%m}_{side}_{experiment["name"]}.joblib';joblib.dump(bundle,path,compress=3)
                    pred=predict(bundle,x.loc[te],features.loc[te])
                    np.testing.assert_allclose(pred,predict(joblib.load(path),x.loc[te],features.loc[te]),rtol=0,atol=1e-7)
                    frame[f'{side}_{experiment["name"]}']=pred
                    fit_audit.append(dict(issue=issue,side=side,experiment=experiment['name'],artifact=path.name,sha256=b.sha(path),
                        iterations=[m['best_iteration'] for m in models],max_training_label_end=[m['max_training_label_end'] for m in models],replay_passed=True))
                # Rebuilt unchanged E2 must match the saved baseline, before any conclusions.
                np.testing.assert_allclose(frame[f'{side}_control_unchanged'],frame[f'{side}_E2'],rtol=0,atol=1e-6)
                b.dump(out/'SELECTIONS.json',selections);b.dump(out/'FIT_AUDIT.json',fit_audit)
            names=[col[3:] for col in frame if col.startswith('UP_')]
            for name in names:frame[f'{name}_signed_bps']=frame.p_up*frame[f'UP_{name}']-frame.p_down*frame[f'DOWN_{name}']
            np.testing.assert_allclose(frame.ridge_signed_bps,reference.loc[te,'ridge_signed_bps'],rtol=0,atol=1e-6)
            frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
    result=pd.concat(frames).sort_index();b.check(result.index.equals(reference.index),'Final population changed')
    result.to_csv(out/'predictions.csv',index_label='decision_time')
    metrics=[];sides=[];tails=[];comparisons=[]
    groups=[('ALL',result)]+[(f'Cycle {q}',result[(result.index.year==2024)&(result.index.quarter==q)]) for q in range(1,5)]+[('Jan2025',result[result.index.year==2025])]
    for period,g in groups:
        if g.empty:continue
        truth=g.actual_signed_return_bps;em=b.metric(truth,g.E2_signed_bps);rm=b.metric(truth,g.ridge_signed_bps)
        threshold=pd.Series(index=g.index,dtype=float)
        for issue,part in g.groupby('issue'):threshold.loc[part.index]=target[y.label_end<issue].quantile(.9)
        tail=truth.abs()>=threshold
        for name in names:
            pred=g[f'{name}_signed_bps'];m=b.metric(truth,pred)
            metrics.append(dict(period=period,model=name,**m,rmse_gain_vs_E2=em['rmse_bps']-m['rmse_bps'],mae_gain_vs_E2=em['mae_bps']-m['mae_bps'],rmse_gain_vs_ridge=rm['rmse_bps']-m['rmse_bps'],mae_gain_vs_ridge=rm['mae_bps']-m['mae_bps'],direction_accuracy_pct=float(100*(old.data.direction(pred)==old.data.direction(truth)).mean())))
            excess=(pred-truth)**2-(g.E2_signed_bps-truth)**2
            tails.append(dict(period=period,model=name,tail_rows=int(tail.sum()),tail_excess_mse_vs_E2=float(excess[tail].sum()/len(g)),nontail_excess_mse_vs_E2=float(excess[~tail].sum()/len(g))))
            for side in ['UP','DOWN']:
                take=truth>0 if side=='UP' else truth<0
                sides.append(dict(period=period,model=name,side=side,**b.metric(truth[take].abs(),g.loc[take,f'{side}_{name}'])))
    for name in names:
        g=result[['actual_signed_return_bps']].copy();g['model']=name;g['predicted_return_bps']=result[f'{name}_signed_bps']
        g['actual_return_pct']=g.actual_signed_return_bps/100;g['predicted_return_pct']=g.predicted_return_bps/100
        g['actual_direction']=old.data.direction(g.actual_signed_return_bps);g['predicted_direction']=old.data.direction(g.predicted_return_bps)
        g['error_bps']=g.predicted_return_bps-g.actual_signed_return_bps;comparisons.append(g)
    pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    table=pd.DataFrame(metrics);table.to_csv(out/'cycle_signed_metrics.csv',index=False)
    pd.DataFrame(sides).to_csv(out/'cycle_magnitude_metrics.csv',index=False);pd.DataFrame(tails).to_csv(out/'tail_error_contributions.csv',index=False)
    mean=table[table.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby('model')[['rmse_bps','mae_bps','rmse_gain_vs_E2','mae_gain_vs_E2']].mean();mean.to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    columns=['period','model','rmse_bps','mae_bps','rank_ic','rmse_gain_vs_E2','mae_gain_vs_E2']
    summary='DIRECT E2 CONTROLS: all235 original inputs retained; independent branches; historical exposed evaluation; no outer promotion.\nALL VARIANTS:\n'+table[table.period.eq('ALL')][columns].to_string(index=False)
    summary+='\n\nEARLIER-SELECTED POLICIES BY CYCLE:\n'+table[table.model.isin(['ridge','E2','H_selected','C_selected','S_selected'])][columns].to_string(index=False)
    summary+='\n\nUnweighted Cycle2-4 means:\n'+mean.to_string()
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),prediction_sha256=b.sha(out/'predictions.csv'),replay_passed=True,signature=signature))
    print(summary+'\nSaved to '+str(out))


if __name__=='__main__':main()
