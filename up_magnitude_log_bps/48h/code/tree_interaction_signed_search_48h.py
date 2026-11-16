r"""Broad tree/feature/interaction search selected on SIGNED validation MSE.
python .\Code\tree_interaction_signed_search_48h.py --trials 96
No training on import. Historical, exposed evaluation; no performance guarantee.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
import duckdb
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tree_signed_return_search_48h as old

b, price = old.base, old.price
HERE = Path(__file__).resolve().parent
DATA = price.WEEK1 / 'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data'
DIR = b.EXP / 'direction_v45_style_v1/20261112T045009318177Z'
EXTRA = [
    'liquidation__liq_activity_severity_active_severity_percentile',
    'liquidation__liq_side_pressure_percentile',
    'liquidation__liq_temporal_concentration_percentile',
    'liquidation__venue_breadth_percentile',
    'liquidation__cross_asset_liq_activity_raw_active_severity_percentile',
    'flow__flow_pressure_percentile', 'flow__flow_activity_percentile',
    'flow__cvd_path_efficiency_percentile',
    'basis__basis_level_percentile', 'basis__basis_change_percentile',
    'funding__funding_level_percentile', 'funding__funding_change_percentile',
    'oi__leverage_unwind_percentile', 'oi__oi_range_position_percentile',
]
EXTRA = ['F4Z__'+v+'__4h__value' for v in EXTRA]


def interactions(x):
    """Deterministic causal transforms, no learned thresholds or outcome inputs."""
    z = {}
    for i in [0, 1, 5, 8, 10, 12]:
        col = EXTRA[i]
        for hours in [1, 4, 24]:
            lag = pd.Series(x[col].reindex(x.index-pd.Timedelta(hours=hours)).to_numpy(), index=x.index)
            z[f'INT__delta_{i}_{hours}h'] = x[col]-lag
    for i in [0, 1, 5, 8, 10, 12]:
        for j in [0, 1, 5, 8, 10, 12]:
            if j > i:
                z[f'INT__product_{i}_{j}'] = (x[EXTRA[i]]-.5)*(x[EXTRA[j]]-.5)
    for i in [0, 1, 5, 12]:
        for h in [4, 24, 48, 168]:
            z[f'INT__market_{i}_vol_{h}h'] = (x[EXTRA[i]]-.5)*x[f'NEW__rv_{h}h']
    for threshold in [.75, .90, .99]:
        flag = (x[EXTRA[0]] >= threshold).astype(float).where(x[EXTRA[0]].notna())
        z[f'INT__liq_regime_{threshold}'] = flag
        z[f'INT__regime_{threshold}_vol'] = flag*x.NEW__rv_24h
        z[f'INT__regime_{threshold}_move'] = flag*x.NEW__net_24h
    return pd.DataFrame(z, index=x.index)


def make_model(family, c, threads):
    seed = c['seed']
    if family == 'decision_tree':
        return old.DecisionTreeRegressor(max_depth=c['depth'], min_samples_leaf=c['leaf'],
                                        max_features=c['features_fraction'], random_state=seed)
    if family in ['random_forest', 'extra_trees']:
        cls = old.RandomForestRegressor if family == 'random_forest' else old.ExtraTreesRegressor
        return cls(n_estimators=c['trees'], max_depth=c['depth'], min_samples_leaf=c['leaf'],
            max_features=c['features_fraction'], bootstrap=True, max_samples=c['sample'], n_jobs=threads, random_state=seed)
    if family == 'gradient_boosting':
        return old.GradientBoostingRegressor(loss='squared_error', n_estimators=c['trees'],
            learning_rate=c['rate'], max_depth=min(c['depth'], 5), min_samples_leaf=c['leaf'],
            max_features=c['features_fraction'], subsample=c['sample'], random_state=seed)
    if family == 'hist_gradient_boosting':
        return old.HistGradientBoostingRegressor(loss='squared_error', max_iter=c['trees'],
            learning_rate=c['rate'], max_depth=c['depth'], max_leaf_nodes=c['leaves'],
            min_samples_leaf=c['leaf'], l2_regularization=c['l2'], early_stopping=False, random_state=seed)
    cls = getattr(importlib.import_module(family), old.OPTIONAL[family])
    if family == 'xgboost':
        return cls(objective='reg:squarederror', n_estimators=c['trees'], max_depth=c['depth'],
            learning_rate=c['rate'], min_child_weight=c['leaf'], reg_lambda=c['l2'], reg_alpha=c['l1'],
            subsample=c['sample'], colsample_bytree=c['features_fraction'], tree_method='hist',
            n_jobs=threads, random_state=seed)
    if family == 'lightgbm':
        return cls(objective='regression_l2', n_estimators=c['trees'], max_depth=c['depth'],
            num_leaves=min(c['leaves'],2**c['depth']), learning_rate=c['rate'], min_child_samples=c['leaf'],
            reg_lambda=c['l2'], reg_alpha=c['l1'], subsample=c['sample'], subsample_freq=1,
            colsample_bytree=c['features_fraction'], n_jobs=threads, verbosity=-1, random_state=seed)
    return cls(loss_function='RMSE', iterations=c['trees'], depth=c['depth'], learning_rate=c['rate'],
        l2_leaf_reg=c['l2'], bootstrap_type='Bernoulli', subsample=c['sample'],
        rsm=c['features_fraction'], random_strength=c['random_strength'],
        thread_count=threads, random_seed=seed, verbose=False, allow_writing_files=False)


def recipes(columns, families, trials, seed):
    rng = np.random.default_rng(seed)
    pools = {'original': list(b.COLS), 'market': list(b.COLS)+EXTRA,
             'price': list(b.COLS)+[c for c in columns if c.startswith('NEW__')],
             'interaction': list(b.COLS)+EXTRA+[c for c in columns if c.startswith('INT__')],
             'all': list(columns)}
    def pick(values):
        return values[int(rng.integers(len(values)))]
    result=[]
    for i in range(trials):
        family=families[i%len(families)]
        pool='original' if i<len(families) else pick(list(pools))
        fraction=1. if i<len(families) else pick([.4,.65,.85,1.])
        count=max(3,int(np.ceil(len(pools[pool])*fraction)))
        features=sorted(rng.choice(pools[pool],size=count,replace=False).tolist())
        result.append(dict(id=f'trial_{i:03d}_{family}',family=family,columns=features,pool=pool,
            depth=pick([2,3,4,6,8]),leaf=pick([32,64,128,256,512]),trees=pick([96,192,320]),
            rate=pick([.01,.03,.06,.1]),l2=pick([1.,5.,20.,100.]),l1=pick([0.,.1,1.]),
            sample=pick([.6,.8,1.]),features_fraction=pick([.5,.75,1.]),leaves=pick([7,15,31]),
            random_strength=pick([.1,1.,5.]),seed=seed))
    return result


def pair_mse(up, down, probability, actual):
    """All UP/DOWN pair MSEs on the SAME origins; no realized-side routing."""
    a=up*probability[:,None]
    z=down*(1-probability[:,None])+actual[:,None]
    mse=(a*a).mean(axis=0)[:,None]+(z*z).mean(axis=0)[None,:]-2*a.T@z/len(actual)
    return np.maximum(mse,0.)


def load_probabilities():
    done=json.loads((DIR/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE','Direction run incomplete')
    frames=[];audit=[]
    for filename in ['validation_predictions.csv','development_predictions.csv']:
        path=DIR/filename
        b.check(b.sha(path)==done['artifacts'][filename],'Direction prediction hash mismatch')
        f=pd.read_csv(path,index_col='decision_time');f.index=pd.to_datetime(f.index,utc=True)
        groups=f.groupby('fold') if 'fold' in f else [(0,f)]
        for fold,g in groups:
            relative=f'fold_{int(fold)}\\pipeline.joblib' if fold else 'final\\pipeline.joblib'
            path=DIR/relative
            b.check(b.sha(path)==done['artifacts'][relative],'Direction artifact mismatch')
            cutoff=pd.Timestamp(joblib.load(path)['max_label_end'])
            b.check(cutoff<g.index.min(),'Direction labels overlap forecasts')
            audit.append(dict(file=filename,fold=int(fold),max_training_label_end=cutoff,
                              forecast_first=g.index.min(),forecast_last=g.index.max()))
        frames.append(f[['p_up','y']])
    f=pd.concat(frames).sort_index()
    b.check(f.index.is_unique and f.p_up.between(0,1).all(),'Invalid probability index/values')
    return f,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=96,help='Total balanced candidate count, not per family')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=48)
    p.add_argument('--check-data-only',action='store_true',help='Validate real input alignment and causal features, then exit without training')
    p.add_argument('--families',nargs='+',choices=old.FAMILIES,default=old.FAMILIES)
    p.add_argument('--resume',type=Path,help='Resume this exact runner/configuration from an incomplete output folder')
    args=p.parse_args();families=list(dict.fromkeys(args.families))
    b.check(len(families)<=args.trials<=256 and args.threads>0,'Invalid trial/thread count')
    for family in families:
        if family in old.OPTIONAL and not args.check_data_only:
            b.check(hasattr(importlib.import_module(family),old.OPTIONAL[family]),f'Unavailable {family}; see README')
    old.make_model=make_model  # reuse tested train-only preprocessing, target scaling and prediction helper
    refpath=old.REF/'signed_return_predictions.csv'
    done=json.loads((old.REF/'COMPLETE.json').read_text())
    b.check(done['status']=='COMPLETE' and b.sha(refpath)==done['prediction_sha256'],'Ridge reference integrity failed')
    ref=pd.read_csv(refpath,index_col='decision_time');ref.index=pd.to_datetime(ref.index,utc=True)
    ref['issue']=pd.to_datetime(ref.issue,utc=True);ref=ref.sort_index()
    b.check(ref.index.is_unique and ref.symbol.eq('BTCUSDT').all() and ref.horizon_hours.eq(48).all(),'Invalid reference')
    prepared=price.WEEK1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    x,y,dev,_=joblib.load(prepared)
    b.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid prepared index')
    b.check(len(ref)==18481 and pd.DatetimeIndex(dev).sort_values().equals(ref.index),'Evaluation keys differ')
    y=y.copy();y['label_end']=pd.to_datetime(y.label_end,utc=True)
    b.check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Wrong horizon')
    signed=pd.Series(10000*np.expm1(y.signed_return_log_bps/10000),index=y.index);target=signed.abs()
    b.check(np.isfinite(signed).all(),'Nonfinite target')
    np.testing.assert_allclose(signed.loc[ref.index],ref.actual_signed_return_bps,rtol=0,atol=1e-7)
    probabilities,dir_audit=load_probabilities()
    common=y.index.intersection(probabilities.index)
    b.check(np.array_equal(probabilities.loc[common,'y'],(signed.loc[common]>0).astype(int)),'Direction target mismatch')
    np.testing.assert_allclose(probabilities.loc[ref.index,'p_up'],ref.p_up,rtol=0,atol=1e-7)
    print('Loading added market features and causal interactions...',flush=True)
    with duckdb.connect() as con:
        frames=[]
        for name in ['X_train.parquet','X_development.parquet']:
            f=con.execute('SELECT decision_time,'+','.join('"'+c+'"' for c in EXTRA)+' FROM read_parquet(?)',[str(DATA/name)]).fetchdf()
            f['decision_time']=pd.to_datetime(f.decision_time,utc=True);frames.append(f.set_index('decision_time'))
        raw=pd.concat(frames).sort_index()
        # DuckDB exposes TIMESTAMPTZ as microseconds; prepared pandas data uses
        # nanoseconds. Normalize representation without rounding or moving time.
        raw.index=raw.index.as_unit('ns')
        expected_index=pd.DatetimeIndex(x.index).as_unit('ns')
        b.check(raw.index.is_unique and raw.index.equals(expected_index),
                f'Market feature index differs: raw={len(raw)}, prepared={len(expected_index)}, '
                f'missing={len(expected_index.difference(raw.index))}, extra={len(raw.index.difference(expected_index))}')
    x=x.join(raw)
    returnpath=price.WEEK1/'Dataset/canonical_returns_5m.parquet'
    r=price.canonical.load_returns(returnpath);price_audit=price.canonical.validate(r,y)
    price.causal_check(r,x.index);x=x.join(price.price_features(r,x.index))
    cut=x.index[len(x)//2]
    engineered=interactions(x)
    pd.testing.assert_frame_equal(engineered.loc[:cut],interactions(x.loc[:cut]))
    x=x.join(engineered).replace([np.inf,-np.inf],np.nan)
    b.check(len(x)==len(y) and x.index.equals(y.index) and x.columns.is_unique,'Feature assembly changed rows or columns')
    if args.check_data_only:
        print(f'DATA CHECK PASSED: {len(x)} rows, {len(x.columns)} candidate features, '
              f'{len(ref)} evaluation origins; market alignment, labels, direction provenance and causal features checked. No training.',flush=True)
        return
    candidates=recipes(x.columns,families,args.trials,args.seed)
    inputs=[prepared,refpath,returnpath,DATA/'X_train.parquet',DATA/'X_development.parquet',
            DIR/'validation_predictions.csv',DIR/'development_predictions.csv']
    config=dict(candidates=candidates,seed=args.seed,threads=args.threads,
        input_hashes={str(q):b.sha(q) for q in inputs},
        code_hashes={str(q):b.sha(q) for q in [Path(__file__),Path(old.__file__),Path(b.__file__),Path(price.__file__)]})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=args.resume or b.EXP/'tree_interaction_signed_search_48h_v1'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        prior=json.loads((out/'CONTRACT.json').read_text())
        b.check(prior['signature']==signature,'Resume inputs/code/config differ; use original arguments or start a new run')
        b.check(not (out/'COMPLETE.json').exists(),'This run is already complete')
    else:
        out.mkdir(parents=True,exist_ok=False)
        b.dump(out/'CONTRACT.json',dict(signature=signature,config=config,direction_audit=dir_audit,price_audit=price_audit,
            feature_definitions='14 extra 4h percentile fields; exact-time1/4/24h deltas; centered pair products; volatility products; .75/.90/.99 liquidation regimes',
            selection='Joint UP/DOWN pair by pooled SIGNED-return MSE on earlier3windows with purged saved direction forecasts; family-specific pairs and cross-family pair',
            target='Mean positive endpoint magnitudes on each side in ordinarybps, affine scaling, squared-error training',
            preprocessing='Train-only median/standardization/missing flags; standardized features clipped8sigma; negative magnitude forecasts projected to0',
            chronology='All training label_end<cutoff; validation origins first2months after9/6/3month earlier cutoffs, labels mature before outer issue',
            feature_availability='Past-only construction tested; F4Z publication availability inherited from prepared package, not re-certified',
            limitations='Historical exposed evaluation. No independent confirmation; many pair comparisons can overfit validation. No guarantee of beating Ridge.',
            budget=dict(inner_fits=args.trials*24,maximum_outer_fits=(len(families)+1)*8)))
        for q in [Path(__file__),Path(old.__file__),Path(b.__file__),Path(price.__file__),Path(price.canonical.__file__)]:
            (out/q.name).write_bytes(q.read_bytes())
    print(f'Output/checkpoint directory: {out}',flush=True)
    print(f'Candidates={len(candidates)}; inner fits={len(candidates)*24}; candidate inputs={len(x.columns)}',flush=True)
    frames=[];selections=[];fit_audit=[]
    with threadpool_limits(limits=args.threads):
        for issue in b.ISSUES:
            folder=out/f'{issue:%Y-%m}';folder.mkdir(exist_ok=True)
            windows=[]
            for offset in [9,6,3]:
                cutoff=issue-pd.DateOffset(months=offset)
                expected=y.index[(y.index>=cutoff)&(y.index<cutoff+pd.DateOffset(months=2))
                                 &(y.label_end<cutoff+pd.DateOffset(months=3))]
                b.check(expected.isin(probabilities.index).all(),'Missing leakage-safe validation probabilities; no substitution allowed')
                b.check(len(expected)>=100 and y.loc[expected,'label_end'].max()<issue,'Invalid validation window')
                windows.append((cutoff,expected))
            vk=windows[0][1].append([v[1] for v in windows[1:]])
            b.check(vk.is_unique,'Overlapping validation rows')
            val_matrices={}
            for side in ['UP','DOWN']:
                mask=signed>0 if side=='UP' else signed<0
                predictions=[]
                for number,c in enumerate(candidates):
                    path=folder/f'{side}_{c["id"]}_validation.joblib'
                    receipt=path.with_suffix('.json')
                    if path.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256']==b.sha(path):
                        cached=joblib.load(path)
                        b.check(cached['keys'].equals(vk),'Cache timestamp mismatch')
                    else:
                        print(f'{issue.date()} {side} {number+1}/{len(candidates)} {c["id"]} inputs={len(c["columns"])}',flush=True)
                        pp=[];audit=[]
                        for cutoff,va in windows:
                            tr=b.train_keys(y,mask,cutoff,None)
                            a=old.fit(x,target,tr,c['columns'],c['family'],c,args.threads)
                            pp.append(b.predict(a,x.loc[va]))
                            audit.append(dict(cutoff=cutoff,train_rows=len(tr),max_training_label_end=y.loc[tr,'label_end'].max(),
                                              validation_rows=len(va),warnings=a['fit_warnings']))
                        cached=dict(keys=vk,predictions=np.concatenate(pp),audit=audit)
                        joblib.dump(cached,path,compress=3);b.dump(receipt,dict(sha256=b.sha(path)))
                    predictions.append(cached['predictions'])
                val_matrices[side]=np.column_stack(predictions)
            mse=pair_mse(val_matrices['UP'],val_matrices['DOWN'],probabilities.loc[vk,'p_up'].to_numpy(),signed.loc[vk].to_numpy())
            chosen={}
            for family in families+['joint_selected']:
                ids=list(range(len(candidates))) if family=='joint_selected' else [i for i,c in enumerate(candidates) if c['family']==family]
                ii,jj=np.unravel_index(np.argmin(mse[np.ix_(ids,ids)]),(len(ids),len(ids)))
                chosen[family]=(ids[ii],ids[jj])
            pairs=pd.DataFrame(mse,index=[c['id'] for c in candidates],columns=[c['id'] for c in candidates])
            pairs.to_csv(folder/'signed_validation_pair_mse.csv',index_label='UP_candidate')
            selections.append(dict(issue=issue,policies={name:dict(UP=candidates[i],DOWN=candidates[j],
                signed_validation_rmse=float(np.sqrt(mse[i,j]))) for name,(i,j) in chosen.items()}))
            b.dump(out/'SELECTIONS.json',selections)  # saved before final fitting/evaluation
            te=ref.index[ref.issue.eq(issue)]
            b.check(len(te)>0 and (te>=issue).all(),'Invalid evaluation fold')
            f=ref.loc[te,['issue','p_up','p_down','actual_signed_return_bps']].copy()
            f['symbol'],f['horizon_hours']='BTCUSDT',48
            final={}
            for side,pos in [('UP',0),('DOWN',1)]:
                mask=signed>0 if side=='UP' else signed<0
                tr=b.train_keys(y,mask,issue,None)
                for ident in sorted(set(pair[pos] for pair in chosen.values())):
                    c=candidates[ident]
                    a=old.fit(x,target,tr,c['columns'],c['family'],c,args.threads)
                    a.update(issue=issue,side=side,max_training_label_end=y.loc[tr,'label_end'].max())
                    path=folder/f'{side}_{c["id"]}_final.joblib';joblib.dump(a,path,compress=3)
                    pred=b.predict(a,x.loc[te]);np.testing.assert_allclose(pred,b.predict(joblib.load(path),x.loc[te]),rtol=0,atol=1e-6)
                    final[(side,ident)]=pred
                    fit_audit.append(dict(issue=issue,side=side,candidate=c['id'],artifact=str(path.relative_to(out)),
                        sha256=b.sha(path),max_training_label_end=a['max_training_label_end'],training_rows=len(tr)))
            for name,(i,j) in chosen.items():
                f[f'{name}_up_mean_bps']=final[('UP',i)];f[f'{name}_down_mean_bps']=final[('DOWN',j)]
                f[f'{name}_signed_bps']=f.p_up*final[('UP',i)]-f.p_down*final[('DOWN',j)]
            f['ridge_signed_bps']=ref.loc[te,'signed_return_forecast_bps']
            f['side_means_signed_bps']=ref.loc[te,'side_mean_baseline_bps'];f['zero_signed_bps']=0.
            f.to_csv(folder/'predictions.csv',index_label='decision_time');frames.append(f)
            b.dump(out/'FIT_AUDIT.json',fit_audit)
            print(f'DONE {issue.date()}: joint pair {chosen["joint_selected"]}',flush=True)
    result=pd.concat(frames).sort_index();b.check(result.index.equals(ref.index),'Final key mismatch')
    models=families+['joint_selected','ridge','side_means','zero'];scores=[];long=[]
    for period,g in [('ALL',result)]+[(str(k),v) for k,v in result.groupby('issue')]:
        for name in models:
            pred=g[f'{name}_signed_bps'];actual=g.actual_signed_return_bps
            scores.append(dict(period=period,model=name,**b.metric(actual,pred),
                direction_accuracy_pct=100*float((old.direction(pred)==old.direction(actual)).mean()),
                predicted_up_pct=100*float((pred>0).mean())))
    for name in models:
        f=pd.DataFrame(index=result.index);f['model']=name
        f['predicted_return_pct']=result[f'{name}_signed_bps']/100;f['actual_return_pct']=result.actual_signed_return_bps/100
        f['predicted_direction']=old.direction(result[f'{name}_signed_bps']);f['actual_direction']=old.direction(result.actual_signed_return_bps)
        f['direction_correct']=f.predicted_direction==f.actual_direction
        f['error_bps']=result[f'{name}_signed_bps']-result.actual_signed_return_bps;long.append(f)
    table=pd.DataFrame(scores);table.to_csv(out/'signed_metrics.csv',index=False)
    result.to_csv(out/'signed_return_predictions.csv',index_label='decision_time')
    pd.concat(long).to_csv(out/'prediction_vs_actual_with_direction.csv',index_label='decision_time')
    summary=('INTERACTION TREE SEARCH: JOINT SIGNED-RETURN VALIDATION SELECTION\n'
        'Each family and joint selector frozen using earlier forecasts only. Historical exposed evaluation, no automatic promotion.\n'
        +table[table.period.eq('ALL')].sort_values('rmse_bps').to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    b.dump(out/'COMPLETE.json',dict(status='COMPLETE',rows=len(result),training_performed=True,
        direction_retrained=False,model_replay_passed=True,prediction_sha256=b.sha(out/'signed_return_predictions.csv'),
        contract_sha256=b.sha(out/'CONTRACT.json')))
    print(summary);print('Outputs saved:',out)


if __name__=='__main__':
    main()
