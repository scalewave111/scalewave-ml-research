r"""Separate UP/DOWN package-style median/RF hybrids; user-run training.
python .\Code\fit_package_median_hybrid_signed_48h.py
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import fit_report_hybrid_means_48h as data
b=data.b
HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'ScaleWave_Magnitude_Model'
E2=b.EXP/'lightgbm_repair_experiments_48h_v1/20261113T053731575845Z'


def package_spec():
    cfg=json.loads((PACKAGE/'config.json').read_text())
    features=pd.read_csv(PACKAGE/'features.csv').sort_values('order').exact_column.tolist()
    b.check(features==cfg['anchor_features']+list(cfg['additions'].values()),'Package feature order mismatch')
    b.check(len(features)==7 and cfg['seeds']==[48,17,91],'Unexpected package features/seeds')
    cards=[json.loads((PACKAGE/f'models/fold_{i}_selected.json').read_text())['selection'] for i in range(1,5)]
    return cfg,features,cards


def prepare(frame,columns):
    metadata={}
    for col in columns:
        v=frame[col].replace([np.inf,-np.inf],np.nan)
        b.check(v.notna().any(),f'All-missing training column: {col}')
        median=float(v.median());filled=v.fillna(median);scale=float(filled.std(ddof=0))
        metadata[col]=dict(median=median,center=float(filled.mean()),scale=scale if scale>0 else 1.)
    return metadata


def transform(x,metadata):
    columns=[]
    for col,a in metadata.items():
        v=x[col].replace([np.inf,-np.inf],np.nan)
        columns.extend([((v.fillna(a['median'])-a['center'])/a['scale']).to_numpy(),v.isna().to_numpy(float)])
    z=np.column_stack(columns);b.check(np.isfinite(z).all(),'Nonfinite feature design');return z


def fit(x,target,keys,features,card,seeds,threads):
    pp=prepare(x.loc[keys],features[:2]);fp=prepare(x.loc[keys],features)
    yy=target.loc[keys].to_numpy(float)
    pair=QuantileRegressor(quantile=.5,alpha=0.,solver='highs').fit(transform(x.loc[keys],pp),yy)
    # sklearn prohibits an integer max_samples larger than the side-specific data.
    draws=min(int(card['max_samples']),len(keys))
    params=dict(n_estimators=128,criterion='absolute_error',max_depth=int(card['depth']),
        min_samples_leaf=int(card['leaf']),max_features=1.,bootstrap=True,max_samples=draws,n_jobs=threads)
    forests=[];z=transform(x.loc[keys],fp)
    for seed in seeds:
        print(f'  Forest seed={seed}, rows={len(keys)}, bootstrap={draws}',flush=True)
        forests.append(RandomForestRegressor(random_state=seed,**params).fit(z,yy))
    return dict(pair=pair,pair_preprocessing=pp,forests=forests,forest_preprocessing=fp,
        weight=float(card['weight']),seeds=seeds,settings=card,training_rows=len(keys),
        nominal_bootstrap=int(card['max_samples']),effective_bootstrap=draws,
        training_mean=float(yy.mean()),training_median=float(np.median(yy)))


def components(a,x):
    pair=np.maximum(0,a['pair'].predict(transform(x,a['pair_preprocessing'])))
    z=transform(x,a['forest_preprocessing'])
    # Preserve package's clipping and per-tree averaging order.
    forest=np.mean([np.maximum(0,np.mean([t.predict(z) for t in m.estimators_],axis=0)) for m in a['forests']],axis=0)
    hybrid=(1-a['weight'])*pair+a['weight']*forest
    b.check(np.isfinite(hybrid).all(),'Nonfinite forecasts')
    return dict(pair=pair,forest=forest,hybrid=hybrid)


def load():
    cfg,features,cards=package_spec()
    x,y,signed,ref=data.load_data();x=x[features]
    done=json.loads((E2/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(E2/'predictions.csv')==done['prediction_sha256'],'E2 integrity failed')
    previous=pd.read_csv(E2/'predictions.csv',index_col='decision_time');previous.index=pd.to_datetime(previous.index,utc=True)
    b.check(previous.index.equals(ref.index),'E2 population differs')
    for col in ['p_up','p_down','actual_signed_return_bps']:
        np.testing.assert_allclose(previous[col],ref[col],rtol=0,atol=1e-7)
    b.check(len(cfg['validation_windows'])==4,'Expected four package windows')
    for issue,(start,end),card in zip(b.ISSUES,cfg['validation_windows'],cards):
        b.check(pd.Timestamp(start)==issue,'Package cutoff differs')
        expected=x.index[(x.index>=pd.Timestamp(start))&(x.index<pd.Timestamp(end))]
        b.check(expected.equals(ref.index[ref.issue.eq(issue)]),'Package evaluation window differs')
    return x,y,signed,ref,previous,cfg,features,cards


def self_test():
    cfg,features,cards=package_spec();rng=np.random.default_rng(48)
    x=pd.DataFrame(rng.normal(size=(320,7)),columns=features);x.iloc[::19,0]=np.nan
    target=pd.Series(100+5*x.iloc[:,1]+rng.exponential(10,len(x)))
    a=fit(x,target,x.index[:256],features,cards[0],cfg['seeds'],1)
    result=components(a,x.iloc[256:]);assert a['effective_bootstrap']==256
    assert a['pair'].quantile==.5 and a['pair'].alpha==0
    assert all(m.criterion=='absolute_error' and m.n_estimators==128 for m in a['forests'])
    np.testing.assert_allclose(result['hybrid'],.5*result['pair']+.5*result['forest'])
    assert all(np.isfinite(v).all() and (v>=0).all() for v in result.values())
    changed=x.copy();changed.iloc[256:]=1e6
    assert prepare(x.iloc[:256],features)==prepare(changed.iloc[:256],features)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'a.joblib';joblib.dump(a,path)
        np.testing.assert_allclose(result['hybrid'],components(joblib.load(path),x.iloc[256:])['hybrid'],atol=1e-10,rtol=0)
    a['weight']=0.;np.testing.assert_allclose(components(a,x)['hybrid'],components(a,x)['pair'])
    print('PASS: package feature order, median/RF objectives, seeds, bootstrap cap, preprocessing, blend, zero-weight and saved replay.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',type=Path);p.add_argument('--check-data-only',action='store_true');p.add_argument('--self-test',action='store_true')
    args=p.parse_args();b.check(args.threads>0,'Positive threads required')
    if args.self_test:self_test();return
    x,y,signed,ref,previous,cfg,features,cards=load();target=signed.abs()
    if args.check_data_only:
        print(f'PASS: {len(x)} rows, seven exact package features, {len(ref)} aligned E2/Ridge origins; no training.')
        for issue,card in zip(b.ISSUES,cards):
            for side,mask in [('UP',signed>0),('DOWN',signed<0)]:
                keys=b.train_keys(y,mask,issue,None)
                print(issue.date(),side,'rows',len(keys),'nominal/effective bootstrap',card['max_samples'],min(card['max_samples'],len(keys)))
        return
    sources=[Path(__file__),Path(data.__file__),Path(b.__file__),PACKAGE/'config.json',PACKAGE/'features.csv']+[PACKAGE/f'models/fold_{i}_selected.json' for i in range(1,5)]
    inputs=[data.PREPARED,data.DATA/'X_train.parquet',data.DATA/'X_development.parquet',data.REF/'signed_return_predictions.csv',E2/'predictions.csv',E2/'COMPLETE.json']
    identity=dict(hashes={str(q):b.sha(q) for q in sources+inputs},threads=args.threads,
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib']})
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'package_median_hybrid_signed_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        b.check(not (out/'COMPLETE.json').exists(),'Run already complete')
        b.check(json.loads((out/'CONTRACT.json').read_text())['signature']==signature,'Resume signature mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,identity=identity,features=features,seeds=cfg['seeds'],quarterly_settings=cards,
            model='Two-input QuantileRegressor(q=.5,alpha=0,highs) + mean of three128tree absolute_error forests using seven inputs; fixed package weights.',
            adaptation='Separate UP/DOWN training on positive ordinary-bps endpoint magnitudes. Same package architecture/settings except bootstrap count capped by side training rows when necessary. Not exact original overall-log-magnitude reproduction.',
            preprocessing='Training median imputation,mean/std scaling,interleaved missing flags per feature; no8sigma clipping,no target normalization; clip pair and each forest prediction at0 before ensemble/blend.',
            evaluation='Same18481origins,fixed v45-style probabilities; signed=p_up*UP_median_hybrid-p_down*DOWN_median_hybrid. This is a median-oriented weighted forecast, not a conditional-mean expected return.',
            selection='No new tuning or outer-driven settings. Fixed Jan/Apr/Jul/Oct weights.5/.5/.25/0 and card hyperparameters were historically selected in original package; same weights used for both new sides.',
            limitation='Previously exposed historical evaluation, inherited F4Z availability, no guarantee of beating E2 or Ridge. Target is endpoint magnitude, not path excursion.'))
        for q in sources:
            if q.suffix=='.py':(out/q.name).write_bytes(q.read_bytes())
    print('Output:',out,flush=True);frames=[];audit=[]
    with threadpool_limits(limits=args.threads):
        for issue,card in zip(b.ISSUES,cards):
            te=ref.index[ref.issue.eq(issue)]
            f=ref.loc[te,['p_up','p_down','actual_signed_return_bps']].copy();f['issue']=issue
            f['symbol']='BTCUSDT';f['horizon_hours']=48;f['forecast_end']=te+pd.Timedelta(hours=48)
            f['E2_signed_bps']=previous.loc[te,'E2_signed_bps'];f['ridge_signed_bps']=ref.loc[te,'signed_return_forecast_bps'];f['zero_signed_bps']=0.
            for side,mask in [('UP',signed>0),('DOWN',signed<0)]:
                keys=b.train_keys(y,mask,issue,None);path=out/f'{issue:%Y-%m}_{side}.joblib';receipt=path.with_suffix('.json')
                if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):a=joblib.load(path)
                else:
                    print(f'Fitting {issue.date()} {side}',flush=True)
                    a=fit(x,target,keys,features,card,cfg['seeds'],args.threads)
                    a.update(issue=issue,side=side,max_training_label_end=y.loc[keys,'label_end'].max())
                    joblib.dump(a,path,compress=3);b.dump(receipt,dict(sha256=b.sha(path)))
                predictions=components(a,x.loc[te]);replay=components(joblib.load(path),x.loc[te])
                for name,values in predictions.items():
                    np.testing.assert_allclose(values,replay[name],atol=1e-10,rtol=0);f[f'{side}_{name}']=values
                f[f'{side}_side_medians']=a['training_median'];f[f'{side}_side_means']=a['training_mean']
                for name in ['E2','ridge']:f[f'{side}_{name}']=previous.loc[te,f'{side}_{name}']
                audit.append(dict(issue=issue,side=side,artifact=path.name,sha256=b.sha(path),training_rows=len(keys),
                    max_training_label_end=a['max_training_label_end'],nominal_bootstrap=a['nominal_bootstrap'],effective_bootstrap=a['effective_bootstrap'],weight=a['weight'],replay_passed=True))
                b.dump(out/'FIT_AUDIT.json',audit)
            for name in ['pair','forest','hybrid','side_medians','side_means']:
                f[f'{name}_signed_bps']=f.p_up*f[f'UP_{name}']-f.p_down*f[f'DOWN_{name}']
            f.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(f)
    result=pd.concat(frames).sort_index();b.check(result.index.equals(ref.index),'Final evaluation keys differ')
    names=['hybrid','pair','forest','side_medians','side_means','E2','ridge','zero']
    metrics=[];sides=[];comparisons=[]
    groups=[('ALL',result)]+[(f'Cycle {i}',result[(result.index.year==2024)&(result.index.quarter==i)]) for i in range(1,5)]+[('Jan2025',result[result.index.year==2025])]
    for period,g in groups:
        if g.empty:continue
        actual=g.actual_signed_return_bps;em=b.metric(actual,g.E2_signed_bps);rm=b.metric(actual,g.ridge_signed_bps)
        for name in names:
            pred=g[f'{name}_signed_bps'];m=b.metric(actual,pred)
            metrics.append(dict(period=period,model=name,**m,rmse_gain_vs_E2=em['rmse_bps']-m['rmse_bps'],mae_gain_vs_E2=em['mae_bps']-m['mae_bps'],rmse_gain_vs_ridge=rm['rmse_bps']-m['rmse_bps'],mae_gain_vs_ridge=rm['mae_bps']-m['mae_bps'],direction_accuracy_pct=float(100*(data.direction(pred)==data.direction(actual)).mean())))
            if name!='zero':
                for side in ['UP','DOWN']:
                    mask=actual>0 if side=='UP' else actual<0
                    sides.append(dict(period=period,model=name,side=side,**b.metric(actual[mask].abs(),g.loc[mask,f'{side}_{name}'])))
    for name in names:
        f=result[['actual_signed_return_bps']].copy();f['model']=name;f['predicted_return_bps']=result[f'{name}_signed_bps']
        f['actual_return_pct']=f.actual_signed_return_bps/100;f['predicted_return_pct']=f.predicted_return_bps/100
        f['actual_direction']=data.direction(f.actual_signed_return_bps);f['predicted_direction']=data.direction(f.predicted_return_bps)
        f['direction_correct']=f.actual_direction==f.predicted_direction;f['error_bps']=f.predicted_return_bps-f.actual_signed_return_bps;comparisons.append(f)
    result.to_csv(out/'predictions.csv',index_label='decision_time');pd.concat(comparisons).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    table=pd.DataFrame(metrics);table.to_csv(out/'cycle_signed_metrics.csv',index=False);pd.DataFrame(sides).to_csv(out/'cycle_magnitude_metrics.csv',index=False)
    mean=table[table.period.isin(['Cycle 2','Cycle 3','Cycle 4'])].groupby('model')[['rmse_bps','mae_bps','rmse_gain_vs_E2','mae_gain_vs_E2']].mean();mean.to_csv(out/'cycle_2_3_4_mean_metrics.csv')
    cols=['period','model','rmse_bps','mae_bps','rank_ic','rmse_gain_vs_E2','mae_gain_vs_E2']
    summary='PACKAGE MEDIAN/RF ARCHITECTURE ADAPTED TO CONDITIONAL ENDPOINT MAGNITUDES\nOrdinarybps; weighted medians are not expected returns. Historical exposed evaluation.\n'+table[cols].to_string(index=False)+'\n\nCycle2-4 unweighted means:\n'+mean.to_string()
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8');b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),replay_passed=True,prediction_sha256=b.sha(out/'predictions.csv'),signature=signature))
    print(summary+'\nSaved to '+str(out),flush=True)


if __name__=='__main__':main()
