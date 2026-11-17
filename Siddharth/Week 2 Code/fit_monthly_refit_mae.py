"""Monthly refits versus saved quarterly UP MLP / DOWN hybrid. NEW CPU TRAINING.
Save in Week 2/Code. Requires completed training_windows_mae_v1 (expanding artifacts),
Week 1 original recipe/dependencies, prepared data, RF templates, and raw forecasts.
Within each quarter freeze its original validation-selected epochs / RF blend weight.
Refit on all history with label_end strictly before each monthly origin. No new tuning.
"""
from pathlib import Path
import argparse
import importlib
import sys
import json
import hashlib
import gc
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits

STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)


def check(ok,msg):
    if not ok: raise ValueError(msg)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid score arrays')
    ic=pd.Series(y).corr(pd.Series(p),method='spearman') if len(y)>2 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def predict(recipe,artifact,direction,x,keys):
    if direction=='UP':
        return np.mean([recipe.predict_mlp(a,x,keys) for a in artifact['outer']],axis=0)
    parts=recipe.predict_hybrid(artifact['outer'],x,keys);w=artifact['weight']
    return (1-w)*parts['pair']+w*parts['rf']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--source',type=Path,help='Completed training_windows_mae_v1 folder')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    source=args.source or ws/'Code/experiments/training_windows_mae_v1'
    out=args.out or ws/'Code/experiments/monthly_refit_mae_v1'
    check(not out.exists(),f'Preserving output {out}; use --out with a new folder')
    check((source/'COMPLETE.json').exists(),'Complete training_windows_mae_v1 is required')
    check(json.loads((source/'COMPLETE.json').read_text())['status']=='COMPLETED','Incomplete source experiment')
    recipe_path=w1/'Code/fit_conditional_48h_magnitude_calibrated.py'
    sys.path.insert(0,str(w1/'Code'))
    recipe=importlib.import_module('fit_conditional_48h_magnitude_calibrated')
    check(Path(recipe.__file__).resolve()==recipe_path.resolve(),'Wrong original recipe imported')
    source_contract=json.loads((source/'CONTRACT.json').read_text())
    check(hashlib.sha256(recipe_path.read_bytes()).hexdigest()==source_contract['recipe_sha256'],'Original recipe changed since source experiment')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected test universe')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected horizon')
    check(not np.isinf(x[recipe.COLS].to_numpy(float)).any(),'Unexpected infinite inputs')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    old=pd.read_csv(w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv',index_col='decision_time')
    old.index=pd.to_datetime(old.index,utc=True);old['issue']=pd.to_datetime(old.issue,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamps differ')
    check(set(old.issue.unique())==set(STARTS),'Unexpected original folds')
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Direction mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(old.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(history='expanding',refit='calendar month start',
        settings='Original quarter-specific validation choices frozen throughout that quarter; no monthly tuning',
        purge='label_end strictly before refit timestamp, including earlier test outcomes only once matured',
        first_month='Reuse verified quarterly fit; later months refit weights and training-only preprocessing',
        evaluation='Original test rows unchanged; separate branches scored on matching realized directions',
        limitation='Exposed historical experiment; no significance claim or direction predictor',
        recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest()))
    frames=[];audit=[]
    for issue in STARTS:
        quarter=old.index[old.issue.eq(issue)]
        check(len(quarter)>0 and quarter.min()>=issue,'Invalid reference fold')
        frame=pd.DataFrame({'issue':str(issue),'actual_direction':side.loc[quarter],
                            'actual_magnitude_bps':target.loc[quarter]},index=quarter)
        month_starts=pd.date_range(issue,quarter.max().normalize().replace(day=1),freq='MS')
        frame['refit_time']=None
        for direction in ['UP','DOWN']:
            path=source/f'{issue:%Y-%m}_{direction}_expanding.joblib'
            original=joblib.load(path);note=original['settings']
            tr0=y.index[(y.label_end<issue)&side.eq(direction)]
            check(original['train_keys'].equals(tr0),'Saved quarterly training population mismatch')
            check(pd.Timestamp(note['issue'])==issue and note['side']==direction and note['window']=='expanding','Unexpected source artifact')
            oldcol='mlp_up_magnitude_bps' if direction=='UP' else 'hybrid_down_magnitude_bps'
            control=predict(recipe,original,direction,x,quarter)
            np.testing.assert_allclose(control,old.loc[quarter,oldcol],rtol=0,atol=.001)
            frame[f'{direction}_quarterly']=old.loc[quarter,oldcol]
            if direction=='DOWN':
                check(not note.get('inner_bootstrap_caps') and not note.get('outer_bootstrap_caps'),'Source RF used sample caps; cannot hold recipe fixed')
            for base in ['mean','median']:
                frame[f'{direction}_quarterly_{base}']=old.loc[quarter,f'{direction.lower()}_train_{base}_baseline_bps']
                np.testing.assert_allclose(frame[f'{direction}_quarterly_{base}'],getattr(target.loc[tr0],base)(),rtol=0,atol=1e-8)
            for origin in month_starts:
                keys=quarter[(quarter>=origin)&(quarter<origin+pd.DateOffset(months=1))]
                if len(keys)==0: continue
                train=y.index[(y.label_end<origin)&side.eq(direction)]
                check(y.loc[train,'label_end'].max()<origin,'Purge failed')
                print(f'START {origin.date()} {direction}: {len(train)} training rows',flush=True)
                if origin==issue:
                    # Slice the verified full-quarter replay, preserving original inference batches.
                    pred=pd.Series(control,index=quarter).loc[keys].to_numpy()
                    replay_error=0.;reuse=True
                else:
                    reuse=False
                    if direction=='UP':
                        epochs=note['epochs'];check(len(epochs)==len(recipe.SEEDS),'Invalid seed epoch list')
                        models=[]
                        for seed,epoch in zip(recipe.SEEDS,epochs):
                            print(f'  seed {seed}: fixed {epoch} epochs',flush=True)
                            models.append(recipe.fit_mlp(x,target,train,origin,seed,epochs=epoch))
                        art=dict(outer=models,train_keys=train)
                    else:
                        art=dict(outer=recipe.fit_hybrid(x,target,train,recipe.COLS),weight=original['weight'],train_keys=train)
                    art.update(origin=origin,quarter_issue=issue,frozen_settings=note)
                    pred=predict(recipe,art,direction,x,keys)
                    modelpath=out/f'{origin:%Y-%m}_{direction}.joblib'
                    joblib.dump(art,modelpath,compress=3)
                    loaded=joblib.load(modelpath);replay=predict(recipe,loaded,direction,x,keys)
                    np.testing.assert_allclose(replay,pred,rtol=0,atol=1e-6)
                    replay_error=float(np.max(np.abs(replay-pred)))
                    del art,loaded
                    if direction=='UP': del models
                frame.loc[keys,'refit_time']=str(origin)
                frame.loc[keys,f'{direction}_monthly']=pred
                for base in ['mean','median']:
                    frame.loc[keys,f'{direction}_monthly_{base}']=float(getattr(target.loc[train],base)())
                audit.append(dict(issue=str(issue),origin=str(origin),side=direction,training_rows=len(train),
                    max_training_label_end=str(y.loc[train,'label_end'].max()),reused_quarter_fit=reuse,
                    frozen_epochs=note.get('epochs'),frozen_rf_weight=note.get('rf_weight'),max_replay_error_bps=replay_error))
                print(f'DONE {origin.date()} {direction}',flush=True);gc.collect()
            del original
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
        dump(out/'FIT_AUDIT.json',audit)
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Final index mismatch')
    check(pred.notna().all().all(),'Unassigned predictions')
    rows=[]
    groups=[('pooled','POOLED',pred)]+[('quarter',k,g) for k,g in pred.groupby('issue')]+[('month',k,g) for k,g in pred.groupby('refit_time')]
    for level,period,f in groups:
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)]
            if len(g)==0: continue
            actual=g.actual_magnitude_bps
            names=['quarterly','monthly','quarterly_mean','quarterly_median','monthly_mean','monthly_median']
            met={name:metrics(actual,g[f'{direction}_{name}']) for name in names}
            for name,m in met.items():
                rows.append(dict(level=level,period=period,side=direction,model=name,**m,
                    mae_gain_vs_quarterly_model=met['quarterly']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_quarterly_mean=met['quarterly_mean']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_quarterly_median=met['quarterly_median']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_monthly_mean=met['monthly_mean']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_monthly_median=met['monthly_median']['mae_bps']-m['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('MONTHLY REFIT COMPARISON\nSame models, features and expanding history; original quarter settings frozen within quarter.\n'
        'Monthly refits use only outcomes with label_end strictly before each refit. No monthly tuning.\n'
        'Quarter-start forecasts reuse verified original fits. Later months update parameters and preprocessing.\n'
        'Positive gain means lower MAE. Baselines include original quarterly and refreshed monthly values.\n'
        'Pooled/quarter baseline IC can reflect monthly level shifts; constant-within-month IC is undefined.\n'
        'Historical exposed evaluation; no significance claims or direction predictor.\n\n'
        +scores[scores.level.eq('pooled')].to_string(index=False)+'\n\nBY QUARTER\n'
        +scores[scores.level.eq('quarter')&scores.model.isin(['quarterly','monthly'])].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),monthly_side_checks=len(audit)))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
