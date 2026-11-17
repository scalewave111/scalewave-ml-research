"""Compare expanding, 12-month and 6-month histories without changing models.
UP: original 3-seed MAE MLP. DOWN: original median-regression/RF hybrid.
Save in Week 2/Code. Running this file starts NEW CPU training.
Uses Week 1 source/dependencies, prepared.joblib, RF templates and saved forecasts.
No existing dataset, source or result is modified. No resume into partial folders.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import sys
import json
import gc
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits

WINDOWS={'expanding':None,'rolling12':12,'rolling6':6}
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)


def check(ok,message):
    if not ok: raise ValueError(message)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def keys_for(y,side,direction,cutoff,months):
    mask=(y.label_end<cutoff)&side.eq(direction)
    if months is not None: mask &= y.index>=cutoff-pd.DateOffset(months=months)
    keys=y.index[mask]
    check(len(keys)>=512,f'Insufficient eligible rows: {direction}, {cutoff}, window={months}, rows={len(keys)}')
    check(y.loc[keys,'label_end'].max()<cutoff,'Training purge failed')
    return keys


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(len(y)>0 and y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid scores')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def fit_hybrid(recipe,x,target,keys):
    # Same original recipe, with integer bootstrap sample size capped only when
    # a shorter window has fewer rows than the saved template requested.
    b=recipe.q.b;p=b.prep(x.loc[keys],recipe.COLS);z=b.design(x.loc[keys],p)
    models={};caps=[]
    for kind in ['pair','rf']:
        template=b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib'
        model=clone(joblib.load(template)['model'])
        if kind=='rf':
            model.set_params(n_jobs=1)
            if isinstance(model.max_samples,(int,np.integer)) and model.max_samples>len(keys):
                caps.append(dict(original=int(model.max_samples),used=len(keys)))
                model.set_params(max_samples=len(keys))
        model.fit(z,target.loc[keys].to_numpy(float));models[kind]=model
    return dict(prep=p,models=models,columns=recipe.COLS),caps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();workspace=args.workspace.resolve();week1=workspace.parent/'Week 1'
    out=args.out or workspace/'Code/experiments/training_windows_mae_v1'
    check(not out.exists(),f'Preserving existing output: {out}; use --out with a new folder')
    code=week1/'Code'
    source=code/'fit_conditional_48h_magnitude_calibrated.py'
    check(source.is_file(),f'Missing original recipe: {source}')
    sys.path.insert(0,str(code))
    recipe=importlib.import_module('fit_conditional_48h_magnitude_calibrated')
    check(Path(recipe.__file__).resolve()==source.resolve(),'Imported unexpected recipe')
    data=code/'experiments/eight_feature_exhaustive_v1/prepared.joblib'
    reference=code/'experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    x,y,dev,ref=joblib.load(data)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid index')
    check(dev.equals(ref.index) and len(dev)==18481,'Unexpected development rows')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected label horizon')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    check(np.isfinite(target).all(),'Invalid targets')
    old=pd.read_csv(reference,index_col='decision_time');old.index=pd.to_datetime(old.index,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamps mismatch')
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference direction mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(side.loc[dev].value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    # Preserve original feature preprocessing behavior; functions own their imputation.
    check(set(recipe.COLS).issubset(x.columns),'Missing original features')
    check(not np.isinf(x[recipe.COLS].to_numpy(float)).any(),'Infinite features; original recipe did not specify infinity handling')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'original_recipe_snapshot.py').write_bytes(source.read_bytes())
    dump(out/'CONTRACT.json',dict(windows=WINDOWS,window_definition='Origin timestamp at/after cutoff minus calendar months, and label_end strictly before cutoff',
        models={'UP':'original three-seed MAE MLP','DOWN':'original median regression plus absolute-error RF hybrid'},
        selection='First two months of preceding quarter; choose window by ensemble validation MAE, higher IC breaks exact ties, then prefer expanding/12m/6m in that order',
        inner_selection='Original seed-specific MLP early stopping and RF blend weight selection preserved',
        baselines='Original expanding mean/median plus corresponding window-specific mean/median; selected baseline inherits model-selected window',
        rf_exception='Integer max_samples capped at training-row count if required; every cap logged',
        limitation='Exposed historical evaluation; no guarantee rolling windows improve; no direction classifier',
        recipe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','scikit-learn','joblib','scipy']}))
    frames=[];validation=[];selections=[];settings=[];checks=[]
    for issue in STARTS:
        inner=issue-pd.DateOffset(months=3)
        vk=y.index[(y.index>=inner)&(y.index<issue-pd.DateOffset(months=1))&(y.label_end<issue)]
        end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(minutes=30)
        te=dev[(dev>=issue)&(dev<end)]
        check(len(te)>0 and len(vk)>0,'Empty fold')
        check(pd.to_datetime(old.loc[te,'issue'],utc=True).eq(issue).all(),'Reference fold mismatch')
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            va=vk[side.loc[vk].eq(direction)]
            check(len(va)>0,'No validation outcomes for direction')
            candidates=[]
            oldcol='mlp_up_magnitude_bps' if direction=='UP' else 'hybrid_down_magnitude_bps'
            frame[f'{direction}_existing_reference']=old.loc[te,oldcol]
            for name,months in WINDOWS.items():
                ti=keys_for(y,side,direction,inner,months);tr=keys_for(y,side,direction,issue,months)
                print(f'START {issue.date()} {direction} {name}: {len(ti)} inner / {len(tr)} outer rows',flush=True)
                note=dict(issue=str(issue),side=direction,window=name,months=months,inner_rows=len(ti),outer_rows=len(tr),
                    inner_first_origin=str(ti.min()),outer_first_origin=str(tr.min()),
                    inner_max_label_end=str(y.loc[ti,'label_end'].max()),outer_max_label_end=str(y.loc[tr,'label_end'].max()))
                path=out/f'{issue:%Y-%m}_{direction}_{name}.joblib'
                if direction=='UP':
                    inn=[];vp=[]
                    for seed in recipe.SEEDS:
                        a=recipe.fit_mlp(x,target,ti,inner,seed,valid=va)
                        inn.append(a);vp.append(recipe.predict_mlp(a,x,va))
                        print(f'  seed {seed}: selected {a["epoch"]} epochs',flush=True)
                    v=np.mean(vp,axis=0);outer=[];tp=[]
                    for seed,a in zip(recipe.SEEDS,inn):
                        fitted=recipe.fit_mlp(x,target,tr,issue,seed,epochs=a['epoch'])
                        outer.append(fitted);tp.append(recipe.predict_mlp(fitted,x,te))
                    test=np.mean(tp,axis=0);note['epochs']=[a['epoch'] for a in inn]
                    joblib.dump(dict(outer=outer,settings=note,train_keys=tr),path,compress=3)
                    replay_saved=joblib.load(path)
                    replay=np.mean([recipe.predict_mlp(a,x,te) for a in replay_saved['outer']],axis=0)
                    del inn,outer,replay_saved
                else:
                    hin,capin=fit_hybrid(recipe,x,target,ti)
                    parts=recipe.predict_hybrid(hin,x,va);actual=target.loc[va].to_numpy(float)
                    weight=min(recipe.WEIGHTS,key=lambda w:(float(np.abs((1-w)*parts['pair']+w*parts['rf']-actual).mean()),w))
                    v=(1-weight)*parts['pair']+weight*parts['rf']
                    hout,capout=fit_hybrid(recipe,x,target,tr)
                    parts=recipe.predict_hybrid(hout,x,te);test=(1-weight)*parts['pair']+weight*parts['rf']
                    note.update(rf_weight=weight,inner_bootstrap_caps=capin,outer_bootstrap_caps=capout)
                    joblib.dump(dict(outer=hout,weight=weight,settings=note,train_keys=tr),path,compress=3)
                    replay_saved=joblib.load(path);rp=recipe.predict_hybrid(replay_saved['outer'],x,te)
                    replay=(1-weight)*rp['pair']+weight*rp['rf']
                    del hin,hout,replay_saved
                np.testing.assert_allclose(replay,test,rtol=0,atol=1e-6)
                rec=dict(issue=str(issue),side=direction,window=name,**metrics(target.loc[va],v))
                validation.append(rec);candidates.append(rec);settings.append(note)
                frame[f'{direction}_{name}']=test
                for base,value in [('mean',target.loc[tr].mean()),('median',target.loc[tr].median())]:
                    frame[f'{direction}_{name}_baseline_{base}']=float(value)
                    if name=='expanding':
                        np.testing.assert_allclose(old.loc[te,f'{direction.lower()}_train_{base}_baseline_bps'],value,rtol=0,atol=1e-8)
                if name=='expanding':
                    err=float(np.max(np.abs(test-old.loc[te,oldcol].to_numpy())))
                    checks.append(dict(issue=str(issue),side=direction,expanding_max_error_vs_original_bps=err))
                    # A mismatch would confound history length with a recipe change.
                    if err>1e-3:
                        dump(out/'EXPANDING_MISMATCH.json',checks)
                        raise ValueError(f'Expanding reproduction differs by {err:.6f} bps. Stop and inspect EXPANDING_MISMATCH.json; do not compare windows yet.')
                print(f'DONE {name}: validation MAE {rec["mae_bps"]:.3f}, IC {rec["rank_ic"]}',flush=True)
                gc.collect()
            order=list(WINDOWS)
            win=min(candidates,key=lambda r:(r['mae_bps'],-r['rank_ic'] if r['rank_ic'] is not None else float('inf'),order.index(r['window'])))
            selections.append(win);name=win['window']
            frame[f'{direction}_validation_selected']=frame[f'{direction}_{name}']
            for base in ['mean','median']:
                frame[f'{direction}_selected_baseline_{base}']=frame[f'{direction}_{name}_baseline_{base}']
            print(f'SELECTED {direction}: {name}',flush=True)
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
        pd.DataFrame(validation).to_csv(out/'validation_scores.csv',index=False)
        dump(out/'SELECTIONS.json',selections);dump(out/'FIT_SETTINGS.json',settings);dump(out/'EXPANDING_CHECKS.json',checks)
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Output index mismatch')
    rows=[]
    for fold,f in [('POOLED',pred)]+[(k,v) for k,v in pred.groupby('issue')]:
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)];actual=g.actual_magnitude_bps
            refmae=metrics(actual,g[f'{direction}_existing_reference'])['mae_bps']
            means={b:metrics(actual,g[f'{direction}_expanding_baseline_{b}'])['mae_bps'] for b in ['mean','median']}
            for col in [c for c in pred if c.startswith(direction+'_')]:
                met=metrics(actual,g[col]);rows.append(dict(issue=fold,side=direction,model=col[len(direction)+1:],**met,
                    mae_gain_vs_existing=refmae-met['mae_bps'],mae_gain_vs_expanding_mean=means['mean']-met['mae_bps'],
                    mae_gain_vs_expanding_median=means['median']-met['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    pooled=scores[scores.issue.eq('POOLED')]
    summary=('TRAINING-WINDOW COMPARISON\nSame UP MLP and DOWN hybrid; only eligible history length changes, except logged RF sample caps.\n'
        'Positive gain means lower MAE. IC higher positive is better; constant within-fold IC is undefined.\n'
        'Expanding forecasts reproduced original predictions within 0.001 bps.\n'
        'Validation-selected windows use earlier data only. Fixed-window test winners remain exploratory.\n'
        'Matched-window baselines appear in scores.csv alongside original expanding baselines.\n\n'
        +pooled[pooled.model.isin(list(WINDOWS)+['validation_selected','existing_reference','expanding_baseline_mean','expanding_baseline_median'])].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),folds=4,expanding_reference_checks=len(checks)))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
