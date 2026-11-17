"""Small CPU comparison of median regression, shallow boosting and their blend.
Save in Week 2/Code. Running this script trains NEW models in a separate folder.
Uses existing Week 1 prepared data and raw-model reference predictions.
Dependencies: numpy pandas scipy scikit-learn joblib threadpoolctl.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
      'PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h',
      'F4Z__liquidation__venue_breadth_percentile__1h__value',
      'F4Z__funding__funding_persistence_percentile__5m__value',
      'F4Z__basis__basis_level_rz__1h__value']
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
LINEAR={'linear_light':.001,'linear_stronger':.01}
BOOST={'boost_small':dict(max_iter=100,max_depth=2,max_leaf_nodes=4,min_samples_leaf=128,l2_regularization=5.),
       'boost_regularized':dict(max_iter=200,max_depth=3,max_leaf_nodes=8,min_samples_leaf=256,l2_regularization=10.)}
WEIGHTS=[.1,.25,.5]


def check(ok,message):
    if not ok: raise ValueError(message)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(y.shape==p.shape and len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid scores')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(p-y).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def selection_key(record):
    ic=record['rank_ic']
    return (record['mae_bps'],-ic if ic is not None else float('inf'),record['name'])


def prep(x,keys):
    a=x.loc[keys,COLS];median=a.median().fillna(0);filled=a.fillna(median)
    return dict(columns=COLS,median=median,center=filled.mean(),scale=filled.std(ddof=0).replace(0,1).fillna(1))


def design(x,keys,p):
    a=x.loc[keys,p['columns']]
    z=(a.fillna(p['median'])-p['center'])/p['scale']
    out=np.column_stack([z.to_numpy(float),a.isna().to_numpy(float)])
    check(np.isfinite(out).all(),'Nonfinite design')
    return out


def fit(name,z,y):
    if name in LINEAR:
        model=QuantileRegressor(quantile=.5,alpha=LINEAR[name],solver='highs')
    else:
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,
            early_stopping=False,random_state=48,**BOOST[name])
    # Target scaling is fixed; not fitted calibration. Predictions restored to bps.
    model.fit(z,np.asarray(y,float)/100.)
    return model


def predict(model,z):
    return np.maximum(0.,np.asarray(model.predict(z),float)*100.)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();workspace=args.workspace.resolve();week1=workspace.parent/'Week 1'
    out=args.out or workspace/'Code/experiments/tabular_mae_candidates_v1'
    check(not out.exists(),f'Preserving existing output folder: {out}; use --out with a new directory')
    source=week1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    reference=week1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    x,y,dev,ref=joblib.load(source)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected evaluation rows')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected horizon')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    check(np.isfinite(target).all(),'Invalid targets')
    x=x[COLS].replace([np.inf,-np.inf],np.nan)
    old=pd.read_csv(reference,index_col='decision_time');old.index=pd.to_datetime(old.index,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamp mismatch')
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference direction mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(side.loc[dev].value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected direction counts')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(features=COLS,target='positive ordinary-bps 48h endpoint magnitudes',
        linear=LINEAR,boost=BOOST,blend_weights=WEIGHTS,target_scale=100,
        selection='Lowest earlier validation MAE; IC breaks exact ties. Two linear, two boosting configurations; six blends of selected linear with either boosting configuration, plus linear-only fallback.',
        validation='First two months of preceding quarter, matching original raw-model selection period; label_end before issue',
        training='Expanding side-specific rows, label_end strictly before cutoff',
        limitation='Repeatedly exposed historical tests; exploratory. No promise of improvement; no direction classifier.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    frames=[]; trials=[]; decisions=[]
    for issue in STARTS:
        inner=issue-pd.DateOffset(months=3)
        ik=y.index[y.label_end<inner];ot=y.index[y.label_end<issue]
        vk=y.index[(y.index>=inner)&(y.index<issue-pd.DateOffset(months=1))&(y.label_end<issue)]
        end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(minutes=30)
        te=dev[(dev>=issue)&(dev<end)]
        check(len(te)>0 and len(vk)>0 and y.loc[ik,'label_end'].max()<inner and y.loc[ot,'label_end'].max()<issue,'Purged split failed')
        check(pd.to_datetime(old.loc[te,'issue'],utc=True).eq(issue).all(),'Reference fold mismatch')
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            ti=ik[side.loc[ik].eq(direction)];va=vk[side.loc[vk].eq(direction)];tr=ot[side.loc[ot].eq(direction)]
            check(min(len(ti),len(va),len(tr))>=100,'Insufficient rows')
            print(f'START {issue.date()} {direction}: {len(ti)} inner / {len(tr)} outer rows',flush=True)
            pin=prep(x,ti);zt=design(x,ti,pin);zv=design(x,va,pin)
            candidates=[];vp={}
            for name in list(LINEAR)+list(BOOST):
                model=fit(name,zt,target.loc[ti]);vp[name]=predict(model,zv)
                rec=dict(name=name,**metrics(target.loc[va],vp[name]));candidates.append(rec)
                trials.append(dict(issue=str(issue),side=direction,kind='component',**rec))
                print(f'  {name}: validation MAE {rec["mae_bps"]:.3f}, IC {rec["rank_ic"]}',flush=True)
                del model;gc.collect()
            linear=min([c for c in candidates if c['name'] in LINEAR],key=selection_key)
            boost=min([c for c in candidates if c['name'] in BOOST],key=selection_key)
            blends=[dict(name='linear_only',linear=linear['name'],boost=None,weight=0.,
                         **metrics(target.loc[va],vp[linear['name']]))]
            for name in BOOST:
                for w in WEIGHTS:
                    p=(1-w)*vp[linear['name']]+w*vp[name]
                    blends.append(dict(name=f'{name}_weight_{w}',linear=linear['name'],boost=name,weight=w,
                                       **metrics(target.loc[va],p)))
            for rec in blends: trials.append(dict(issue=str(issue),side=direction,kind='blend',**rec))
            blend=min(blends,key=selection_key)
            # Choose family before any test scores are computed.
            family_records=[dict(name='median_linear',**metrics(target.loc[va],vp[linear['name']])),
                            dict(name='boosting',**metrics(target.loc[va],vp[boost['name']])),
                            dict(name='median_boost_blend',**{k:blend[k] for k in ['rows','mae_bps','rank_ic']})]
            family=min(family_records,key=selection_key)['name']
            decision=dict(issue=str(issue),side=direction,linear=linear,boost=boost,blend=blend,selected_family=family,
                          inner_max_label_end=str(y.loc[ti,'label_end'].max()),outer_max_label_end=str(y.loc[tr,'label_end'].max()))
            decisions.append(decision)
            pout=prep(x,tr);zo=design(x,tr,pout);ze=design(x,te,pout)
            needed={linear['name'],boost['name']}
            if blend['boost'] is not None: needed.add(blend['boost'])
            models={};test={}
            for name in sorted(needed):
                models[name]=fit(name,zo,target.loc[tr]);test[name]=predict(models[name],ze)
            frame[f'{direction}_median_linear']=test[linear['name']]
            frame[f'{direction}_boosting']=test[boost['name']]
            w=blend['weight']
            frame[f'{direction}_median_boost_blend']=(1-w)*test[linear['name']]+(w*test[blend['boost']] if w else 0.)
            frame[f'{direction}_validation_selected']=frame[f'{direction}_{family}']
            for base,value in [('mean',target.loc[tr].mean()),('median',target.loc[tr].median())]:
                frame[f'{direction}_baseline_{base}']=float(value)
                np.testing.assert_allclose(old.loc[te,f'{direction.lower()}_train_{base}_baseline_bps'],value,rtol=0,atol=1e-8)
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
            # Replay the saved representation before declaring this side complete.
            artifact=dict(models=models,prep=pout,decision=decision,train_keys=tr,target_scale=100.)
            apath=out/f'{issue:%Y-%m}_{direction}_models.joblib';joblib.dump(artifact,apath,compress=3)
            replay=joblib.load(apath)
            for name in needed:
                np.testing.assert_allclose(predict(replay['models'][name],design(x,te,replay['prep'])),test[name],rtol=0,atol=1e-8)
            print(f'DONE {direction}: validation-selected family {family}',flush=True)
            del models,replay,artifact,zt,zv,zo,ze;gc.collect()
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
        pd.DataFrame(trials).to_csv(out/'validation_trials.csv',index=False);dump(out/'SELECTIONS.json',decisions)
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Output row mismatch')
    rows=[]
    for fold,f in [('POOLED',pred)]+[(k,v) for k,v in pred.groupby('issue')]:
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)];a=g.actual_magnitude_bps
            means={b:metrics(a,g[f'{direction}_baseline_{b}'])['mae_bps'] for b in ['mean','median']}
            reference_name='existing_mlp' if direction=='UP' else 'existing_hybrid'
            reference_mae=metrics(a,g[f'{direction}_{reference_name}'])['mae_bps']
            for col in [c for c in pred if c.startswith(direction+'_')]:
                met=metrics(a,g[col])
                rows.append(dict(issue=fold,side=direction,model=col[len(direction)+1:],**met,
                    mae_gain_vs_mean=means['mean']-met['mae_bps'],mae_gain_vs_median=means['median']-met['mae_bps'],
                    current_reference=reference_name,mae_gain_vs_current=reference_mae-met['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('NEW TABULAR MAE CANDIDATES\nPositive MAE gain means improvement. Higher positive Rank IC is better.\n'
        'Settings and family selected using earlier validation only; pooled baseline IC is not within-fold ranking.\n'
        'All eight features; no mean calibration. Historical comparisons remain exploratory.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),folds=4,saved_component_replay_passed=True))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
