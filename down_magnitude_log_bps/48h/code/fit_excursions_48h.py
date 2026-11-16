"""User-run 48h canonical UP/DOWN excursion Ridge, LightGBM and E2-style models."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import lightgbm_repair_experiments_48h as e2
import fit_package_median_hybrid_signed_48h as package
import discover_excursion_features_4h as source

b = e2.b
TARGETS = source.TARGETS
TRANSITION = pd.Timestamp('2024-10-29T00:01:00Z')


def eligible_labels(raw, index):
    raw = raw.reindex(index)
    valid = (raw.label_status.eq('VALID') & raw.canonical_symbol.eq('BTCUSDT') &
             pd.to_numeric(raw.horizon_minutes).eq(2880) &
             np.isfinite(raw[TARGETS]).all(axis=1) & raw[TARGETS].ge(0).all(axis=1))
    crossing = (index < TRANSITION) & (index+pd.Timedelta(hours=48) >= TRANSITION)
    reason = pd.Series('included', index=index)
    reason.loc[~valid] = 'missing_or_invalid_label'
    reason.loc[valid & crossing] = 'source_binding_crossing'
    return valid & ~crossing, reason


def load(root):
    # Reuse already checked market values and causal engineering; no probabilities enter X.
    x, short, previous_y, _, ref = e2.old.load()
    returns = e2.price.canonical.load_returns(e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet')
    e2.price.canonical.validate(returns, previous_y)
    features = short.join(e2.long_features(returns, x.index))
    cut = x.index[len(x)//2]
    pd.testing.assert_frame_equal(e2.long_features(returns, pd.DatetimeIndex([cut])),
        e2.long_features(returns.loc[:cut], pd.DatetimeIndex([cut])))
    changed = returns.copy(); changed.loc[changed.index >= cut] = 99999.
    pd.testing.assert_frame_equal(e2.long_features(returns, pd.DatetimeIndex([cut])),
        e2.long_features(changed, pd.DatetimeIndex([cut])))
    files = sorted((root/'y/horizon_minutes=2880/symbol=BTCUSDT').rglob('*.parquet'))
    b.check(bool(files), 'No canonical48h Y partitions')
    receipts = {r['path']: r['sha256'] for r in json.loads((root/'OUTPUT_FILES.json').read_text())}
    for path in files:
        rel = path.relative_to(root).as_posix()
        b.check(rel in receipts and b.sha(path) == receipts[rel], f'Canonical Y checksum mismatch: {rel}')
    with duckdb.connect() as con:
        con.register('origins', pd.DataFrame({'decision_time': x.index}))
        raw = con.execute('''SELECT y.decision_time,y.canonical_symbol,y.horizon_minutes,
            y.label_status,y.up_magnitude_log_bps,y.down_magnitude_log_bps
            FROM read_parquet(?,hive_partitioning=false) y
            JOIN origins k ON k.decision_time=y.decision_time''', [[str(p) for p in files]]).fetchdf()
    raw['decision_time'] = pd.to_datetime(raw.decision_time, utc=True).dt.as_unit('ns')
    b.check(raw.decision_time.is_unique, 'Duplicate Y keys')
    raw = raw.set_index('decision_time')
    keep, reason = eligible_labels(raw, x.index)
    population = pd.DataFrame({'status': reason, 'original_evaluation': x.index.isin(ref.index)})
    keys = x.index[keep]
    y = raw.loc[keys, TARGETS].astype(float)
    y['label_end'] = y.index+pd.Timedelta(hours=48)
    dev = ref.index.intersection(keys)
    audit = dict(original_rows=len(x), valid_rows=len(keys), original_evaluation_rows=len(ref),
        evaluation_rows=len(dev), exclusions=population.status.value_counts().to_dict(),
        label_policy='VALID,finite,nonnegative; exclude origin<2024-10-29T00:01Z<=label_end; no Y imputation.',
        reader='Checksum-verified DuckDB projection of packaged Y, matching canonical validity/binding policy; source dataset unchanged.',
        access='Existing exposed2022-2025development history only, original playground origin grid, not full canonical minute population.')
    return x.loc[keys], features.loc[keys], y, dev, population, files+[root/'OUTPUT_FILES.json'], audit


def ridge_fit(x, target, keys, cols, alpha):
    pp = e2.old.prep(x, keys, cols)
    model = Ridge(alpha=alpha*len(keys)).fit(b.transform(x.loc[keys], pp), target.loc[keys])
    return dict(prep=pp, model=model)


def raw_ridge(a, x):
    return e2.old.ridge_predict(a, x)


def predict(a, x, features):
    if a['kind'] == 'constant':
        return np.full(len(x), a['value'])
    if a['kind'] == 'ridge':
        p = raw_ridge(a['base'], x)
    elif a['kind'] == 'tree':
        p = e2.correction(a['tree'], features)
    else:
        p = raw_ridge(a['base'], x)+e2.correction(a['tree'], features)
    return np.maximum(0, p)


def monthly_oof(x, y, target, cols):
    residual = pd.Series(np.nan, index=x.index, dtype=float)
    months = x.index.tz_convert('UTC').tz_localize(None).to_period('M')
    audit = []
    for month in months.unique():
        cut = month.start_time.tz_localize('UTC')
        tr = y.index[y.label_end < cut]
        va = x.index[months == month]
        if len(tr) < 1024:
            continue
        a = ridge_fit(x, target, tr, cols, .01)
        residual.loc[va] = target.loc[va]-raw_ridge(a, x.loc[va])
        audit.append(dict(cutoff=cut, rows=len(va), training_rows=len(tr), max_label_end=y.loc[tr,'label_end'].max()))
    return residual, audit


def self_test():
    rng = np.random.default_rng(48)
    idx = pd.date_range('2020-01-01', periods=1700, freq='D', tz='UTC')
    x = pd.DataFrame(rng.normal(size=(len(idx),3)), index=idx, columns=['a','b','c'])
    x.iloc[::21,0] = np.nan
    target = pd.Series(100+5*x.b, index=idx)
    y = pd.DataFrame({'label_end':idx+pd.Timedelta(hours=48)}, index=idx)
    residual, audit = monthly_oof(x,y,target,list(x))
    assert audit and all(a['max_label_end'] < a['cutoff'] for a in audit)
    prefix,_ = monthly_oof(x.iloc[:1450],y.iloc[:1450],target.iloc[:1450],list(x))
    pd.testing.assert_series_equal(residual.iloc[:1450],prefix)
    a = dict(kind='ridge',base=ridge_fit(x,target,idx[:1200],list(x),.01))
    changed=x.copy(); changed.iloc[1200:]=1e9
    other=ridge_fit(changed,target,idx[:1200],list(x),.01)
    np.testing.assert_allclose(raw_ridge(a['base'],x),raw_ridge(other,x))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'a.joblib'; joblib.dump(a,path)
        np.testing.assert_allclose(predict(a,x,x),predict(joblib.load(path),x,x))
    ix=pd.DatetimeIndex([TRANSITION-pd.Timedelta(hours=48),TRANSITION,TRANSITION+pd.Timedelta(days=1)])
    raw=pd.DataFrame({'label_status':'VALID','canonical_symbol':'BTCUSDT','horizon_minutes':2880,
                      TARGETS[0]:[0.,1.,np.nan],TARGETS[1]:[2.,3.,4.]},index=ix)
    keep,_=eligible_labels(raw,ix)
    assert keep.tolist()==[False,True,False]
    cfg=dict(id='synthetic',pack='long',days=None,params=e2.old.schedule()[0])
    for kind in ['tree','e2']:
        tree=e2.fit_candidate(x,residual if kind=='e2' else target,y,idx[-1]+pd.Timedelta(days=3),cfg,1,e2.old.lightgbm())
        bundle=dict(kind=kind,base=a['base'],tree=tree)
        values=predict(bundle,x.iloc[-20:],x.iloc[-20:])
        assert np.isfinite(values).all() and (values>=0).all()
        assert tree['early_train_max_label_end']<tree['early_validation_start']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tree.joblib';joblib.dump(bundle,path)
            np.testing.assert_allclose(values,predict(joblib.load(path),x.iloc[-20:],x.iloc[-20:]))
    print('PASS: OOF maturity/prefix, training-only preprocessing, source-crossing/invalid-label exclusion, Ridge/direct-tree/E2 synthetic fits, early-stop purge and saved replay.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root',type=Path,default=source.CANONICAL)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--check-data-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(); b.check(args.threads>0,'Positive threads required')
    if args.self_test: self_test(); return
    x,features,y,dev,population,files,audit=load(args.canonical_root)
    if args.check_data_only:
        print(json.dumps(audit,indent=2)); print(f'PASS: {len(features.columns)} E2features; both excursion targets on same rows. No fitting.'); return
    lgb=e2.old.lightgbm()
    _,seven,_=package.package_spec()
    packs={'original8':b.COLS,'package7':seven,'E2_235':list(features)}
    alphas=[.001,.01,.1,1.]
    trees=[dict(id=c['id'],pack='long',days=None,params=c) for c in e2.old.schedule()
           if (c['max_depth'],c['num_leaves'],c['learning_rate']) in [(3,4,.01),(3,4,.03),(4,8,.02)]]
    modules=[e2,e2.old,e2.old.data,b,e2.price,e2.price.canonical,package,source]
    files += [Path(__file__)]+[Path(m.__file__) for m in modules]+[e2.old.data.PREPARED,
        e2.old.data.DATA/'X_train.parquet',e2.old.data.DATA/'X_development.parquet',
        e2.old.data.REF/'signed_return_predictions.csv',e2.old.data.REF/'COMPLETE.json',
        e2.price.WEEK1/'Dataset/canonical_returns_5m.parquet',package.PACKAGE/'features.csv']
    identity=dict(hashes={str(f):b.sha(f) for f in files},packs=packs,alphas=alphas,trees=trees,threads=args.threads,
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']},lightgbm=lgb.__version__)
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'excursion_models_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not(out/'COMPLETE.json').exists(),'Run already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature']==signature,'Resume signature mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,identity=identity,input_audit=audit,
            targets=TARGETS,units='log bps; no ordinary-bps conversion',
            model='Both targets fit on all valid rows, no endpoint sign filter/probability combination.',
            selection='Three earlier2month windows offset9/6/3months; pooled target MSE. Family best Ridge/directLGBM/E2; selected additionally compares training mean baseline. Median reported as MAE baseline, not MSE-selected.',
            e2='Fixed legacy UP3/DOWN7 Ridge alpha=.01*n on ALL valid rows; monthly OOF residuals; same235causalfeatures;3LightGBM L2configs;expanding history;20percent purged earlystop,1000cap/50patience,refit. Adapted target/population and smaller search, not previous E2 numerical reproduction.',
            limitation='Exposed historical experiment, inherited masked F4Z availability and original playground grid; no significance/fresh confirmation, no guarantee. New source-crossing exclusions mean endpoint scores are not comparable.'))
        population.to_csv(out/'POPULATION_AUDIT.csv',index_label='decision_time')
        for f in files:
            if f.suffix=='.py': (out/f.name).write_bytes(f.read_bytes())
    cache=out/'cache';cache.mkdir(exist_ok=True)
    def cached(name,compute):
        path=cache/(name+'.joblib'); receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):return joblib.load(path)
        a=compute();joblib.dump(a,path,compress=3);b.dump(receipt,dict(sha256=b.sha(path)));return a
    frames=[];ledger=[];selections=[]
    print('Output:',out,flush=True)
    with threadpool_limits(limits=args.threads):
        for target_name,side in zip(TARGETS,['UP','DOWN']):
            target=y[target_name];cols=e2.old.RIDGE_COLS[side]
            # OOF forecasts need only origins before the last outer fit; no evaluation labels feed earlier fits.
            history=y.index[y.index<b.ISSUES[-1]]
            rr,oa=cached('OOF_'+side,lambda:monthly_oof(x.loc[history],y.loc[history],target.loc[history],cols))
            residual=rr.reindex(y.index);b.dump(out/f'OOF_{side}_AUDIT.json',oa)
            specs=[dict(kind='ridge',id=f'{pack}_{alpha}',pack=pack,alpha=alpha) for pack in packs for alpha in alphas]
            specs += [dict(kind=kind,id=kind+'_'+c['id'],config=c) for kind in ['tree','e2'] for c in trees]
            specs += [dict(kind='fixed_ridge',id='fixed_ridge'),dict(kind='mean',id='mean')]
            def fitted(spec,cut):
                keys=y.index[y.label_end<cut]
                def compute():
                    kind=spec['kind']
                    if kind=='mean':return dict(kind='constant',value=float(target.loc[keys].mean()))
                    if kind=='ridge':
                        z=features if spec['pack']=='E2_235' else x
                        return dict(kind='ridge',base=ridge_fit(z,target,keys,packs[spec['pack']],spec['alpha']),design=spec['pack'])
                    base=dict(kind='ridge',base=ridge_fit(x,target,keys,cols,.01)) if kind!='tree' else None
                    if kind=='fixed_ridge':return base
                    tree=e2.fit_candidate(features,residual if kind=='e2' else target,y,cut,spec['config'],args.threads,lgb)
                    return dict(kind=kind,tree=tree,base=base['base'] if base else None)
                return cached(f'{side}_{spec["id"]}_{cut:%Y%m%d}',compute)
            def infer(a,keys):
                z=features if a.get('design')=='E2_235' else x
                return predict(a,z.loc[keys],features.loc[keys])
            for issue in b.ISSUES:
                windows=[]
                for offset in [9,6,3]:
                    cut=issue-pd.DateOffset(months=offset)
                    va=y.index[(y.index>=cut)&(y.index<cut+pd.DateOffset(months=2))&(y.label_end<issue)]
                    b.check(len(va)>100 and y.loc[y.label_end<cut,'label_end'].max()<va.min(),'Invalid inner purge')
                    windows.append((cut,va))
                scores=[]
                for spec in specs:
                    error=np.concatenate([infer(fitted(spec,cut),va)-target.loc[va].to_numpy() for cut,va in windows])
                    mse=float(np.mean(error**2));scores.append(mse)
                    ledger.append(dict(issue=issue,target=target_name,candidate=spec['id'],mse_logbps2=mse))
                pd.DataFrame(ledger).to_csv(out/'validation_metrics.csv',index=False)
                picks={kind:min([i for i,s in enumerate(specs) if s['kind']==kind],key=lambda i:(scores[i],specs[i]['id'])) for kind in ['ridge','tree','e2','fixed_ridge','mean']}
                picks['selected']=min(range(len(specs)),key=lambda i:(scores[i],specs[i]['id']))
                end=issue+pd.DateOffset(months=3) if issue!=b.ISSUES[-1] else dev.max()+pd.Timedelta(minutes=1)
                te=dev[(dev>=issue)&(dev<end)]
                frame=pd.DataFrame({'actual_log_bps':target.loc[te],'target':target_name,'issue':issue},index=te)
                for name,i in picks.items():
                    a=fitted(specs[i],issue);path=out/f'{side}_{issue:%Y%m}_{name}.joblib';joblib.dump(a,path,compress=3)
                    frame[name]=infer(a,te)
                    np.testing.assert_allclose(frame[name],infer(joblib.load(path),te),atol=1e-7,rtol=0)
                    selections.append(dict(issue=issue,target=target_name,model=name,candidate=specs[i],validation_mse=scores[i],training_rows=int((y.label_end<issue).sum()),max_training_label_end=y.loc[y.label_end<issue,'label_end'].max(),artifact=path.name,sha256=b.sha(path),replay_passed=True))
                frame['median']=float(target.loc[y.label_end<issue].median())
                frames.append(frame);b.dump(out/'SELECTIONS.json',selections)
                frame.to_csv(out/f'{side}_{issue:%Y%m}_predictions.csv',index_label='decision_time')
                print(target_name,issue.date(),'selected',specs[picks['selected']]['id'],flush=True)
    result=pd.concat(frames).sort_index();result.to_csv(out/'predictions.csv',index_label='decision_time')
    rows=[]
    for t,g in result.groupby('target'):
        b.check(g.index.is_unique and g.index.sort_values().equals(dev.sort_values()),'Final population differs')
        groups=[('ALL',g)]+[(f'Cycle {q}',g[(g.index.year==2024)&(g.index.quarter==q)]) for q in range(1,5)]+[('Jan2025',g[g.index.year==2025])]
        for period,h in groups:
            if h.empty:continue
            for model in ['mean','median','fixed_ridge','ridge','tree','e2','selected']:
                m=b.metric(h.actual_log_bps,h[model])
                rows.append(dict(target=t,period=period,model=model,rows=len(h),mae_log_bps=m['mae_bps'],rmse_log_bps=m['rmse_bps'],bias_log_bps=m['bias_bps'],rank_ic=m['rank_ic']))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    metrics[metrics.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby(['target','model'])[['mae_log_bps','rmse_log_bps']].mean().to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    summary='48H PATH EXCURSIONS - LOG BPS, NOT ENDPOINT RETURNS\nBoth targets use same valid rows; binding-crossing paths excluded. Historical exposed evaluation.\n'+metrics.to_string(index=False)
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows_per_target=len(dev),prediction_rows=len(result),replay_passed=True,prediction_sha256=b.sha(out/'predictions.csv'),signature=signature))
    print(summary+'\nSaved to '+str(out),flush=True)


if __name__=='__main__':main()
