"""One versus three validation windows for the original UP MLP and DOWN hybrid.
NEW CPU TRAINING. Save beside fit_mlp_regularization_mae.py in Week 2/Code.
Requires Week 1 original recipe, dependencies, RF templates and prepared data.
MLP: same 8/4 net, frozen median regression, three seeds, optimizer and preprocessing.
Each seed chooses epochs from [0,2,4,8,16,32,48]. Both selectors use the same grid.
Hybrid: same components, choose RF weight from [0,.1,.25,.5].
Original saved references use their original early stopping; single_grid is a NEW
control and is not claimed to reproduce original early stopping exactly.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
import fit_mlp_regularization_mae as net

OFFSETS=[9,6,3]
EPOCHS=[0,2,4,8,16,32,48]
WEIGHTS=[0.,.1,.25,.5]


def pick(records,mode):
    """Select lowest row-weighted MAE; prefer smaller setting on exact ties."""
    subset=records if mode=='three_windows' else [r for r in records if r['offset_months']==3]
    settings=sorted({r['setting'] for r in subset})
    scored=[]
    for value in settings:
        rows=[r for r in subset if r['setting']==value]
        scored.append((sum(r['mae_bps']*r['rows'] for r in rows)/sum(r['rows'] for r in rows),value))
    return min(scored)[1]


def epoch_path(linear,z,target,zv,yv,seed):
    """One training trajectory, scored only at prespecified epochs."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=net.MagnitudeNet('original',linear)
    frozen=copy.deepcopy(model.linear.state_dict())
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':[p for p in params if p.ndim>1],'weight_decay':.01},
                          {'params':[p for p in params if p.ndim==1],'weight_decay':0.}],lr=.001)
    yy=np.asarray(target,np.float32)/100.
    scores=[dict(setting=0,**net.metrics(yv,net.predict(model,zv)))]
    for epoch in range(1,max(EPOCHS)+1):
        model.train();order=rng.permutation(len(yy))
        for start in range(0,len(yy),net.BATCH):
            ids=order[start:start+net.BATCH];opt.zero_grad(set_to_none=True)
            loss=(model(z[ids])-torch.from_numpy(yy[ids])).abs().mean()
            net.check(bool(torch.isfinite(loss)),'Nonfinite MLP loss')
            loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
        if epoch in EPOCHS:
            m=net.metrics(yv,net.predict(model,zv));scores.append(dict(setting=epoch,**m))
            print(f'    seed {seed} epoch {epoch}: validation MAE {m["mae_bps"]:.3f}',flush=True)
    for k,v in frozen.items(): torch.testing.assert_close(v,model.linear.state_dict()[k],rtol=0,atol=0)
    return scores


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();ws=args.workspace.resolve();w1=ws.parent/'Week 1'
    out=args.out or ws/'Code/experiments/original_multiwindow_mae_v1'
    net.check(not out.exists(),f'Preserving {out}; use --out with a new folder')
    sys.path.insert(0,str(w1/'Code'))
    recipe=importlib.import_module('fit_conditional_48h_magnitude_calibrated')
    net.check(Path(recipe.__file__).resolve()==(w1/'Code/fit_conditional_48h_magnitude_calibrated.py').resolve(),'Unexpected recipe imported')
    net.check(recipe.COLS==net.COLS and list(recipe.SEEDS)==net.SEEDS,'Recipe features/seeds differ')
    x,y,dev,ref=joblib.load(w1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    net.check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid index')
    net.check(dev.equals(ref.index) and len(dev)==18481,'Unexpected evaluation population')
    net.check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected horizon')
    net.check(not np.isinf(x[net.COLS].to_numpy(float)).any(),'Unexpected infinite features')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    old=pd.read_csv(w1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv',index_col='decision_time')
    old.index=pd.to_datetime(old.index,utc=True);old['issue']=pd.to_datetime(old.issue,utc=True)
    net.check(old.index.equals(dev.sort_values()),'Reference timestamps differ')
    net.check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference directions differ')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    net.check(old.actual_direction.value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    net.check(set(old.issue.unique())==set(net.STARTS),'Unexpected folds')
    rf_template=joblib.load(recipe.q.b.BASE/'models/2024-01__report_outer_rf__expanding.joblib')['model']
    rf_samples=getattr(rf_template,'max_samples',None)
    splits={}
    for issue in net.STARTS:
        for direction in ['UP','DOWN']:
            for months in OFFSETS:
                cutoff=issue-pd.DateOffset(months=months);end=cutoff+pd.DateOffset(months=3)
                train=y.index[(y.label_end<cutoff)&side.eq(direction)]
                val=y.index[(y.index>=cutoff)&(y.index<cutoff+pd.DateOffset(months=2))&(y.label_end<end)&side.eq(direction)]
                net.check(len(train)>=512 and len(val)>=100,f'Insufficient rows: {issue}, {direction}, offset {months}; protocol unchanged')
                net.check(y.loc[train,'label_end'].max()<cutoff and y.loc[val,'label_end'].max()<end and end<=issue,'Purge failed')
                if direction=='DOWN' and isinstance(rf_samples,(int,np.integer)):
                    net.check(len(train)>=rf_samples,f'RF requires {rf_samples} training rows at {cutoff}; refusing to change its settings')
                splits[(issue,direction,months)]=(cutoff,train,val)
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'mlp_recipe_snapshot.py').write_bytes(Path(net.__file__).read_bytes())
    (out/'original_recipe_snapshot.py').write_bytes(Path(recipe.__file__).read_bytes())
    net.dump(out/'CONTRACT.json',dict(offset_months=OFFSETS,epoch_grid=EPOCHS,rf_weights=WEIGHTS,
        validation='First two months following each inner cutoff; train labels end before cutoff, validation labels end before three-month block end',
        selection='Row-weighted validation MAE; exact ties prefer fewer epochs or lower RF weight. UP selects epochs separately for each original seed.',
        control='single_grid uses same epoch/weight candidates as three_windows; saved original MLP used early stopping and is a separate reference',
        history='Expanding quarterly; full eligible outer history used for final fits; no feature search or new architecture',
        limitation='Historical exposed evaluation; no confidence intervals, selection correction, direction prediction or guarantee of improvement'))
    frames=[];all_trials=[];selections=[];audit=[]
    for issue in net.STARTS:
        te=old.index[old.issue.eq(issue)]
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP','DOWN']:
            print(f'START {issue.date()} {direction}',flush=True)
            trials=[]
            for months in OFFSETS:
                cutoff,train,val=splits[(issue,direction,months)]
                print(f'  Inner cutoff {cutoff.date()}: {len(train)} training / {len(val)} validation',flush=True)
                if direction=='UP':
                    prep=net.prep(x,train);z=net.design(x,train,prep);zv=net.design(x,val,prep)
                    linear=net.linear_fit(z,target.loc[train])
                    for seed in net.SEEDS:
                        for r in epoch_path(linear,z,target.loc[train],zv,target.loc[val],seed):
                            trials.append(dict(issue=str(issue),side=direction,offset_months=months,seed=seed,**r))
                    del z,zv
                else:
                    fitted=recipe.fit_hybrid(x,target,train,recipe.COLS)
                    parts=recipe.predict_hybrid(fitted,x,val)
                    for w in WEIGHTS:
                        m=net.metrics(target.loc[val],(1-w)*parts['pair']+w*parts['rf'])
                        trials.append(dict(issue=str(issue),side=direction,offset_months=months,seed=None,setting=w,**m))
                    del fitted
                gc.collect()
            all_trials.extend(trials)
            tr=y.index[(y.label_end<issue)&side.eq(direction)]
            net.check(y.loc[tr,'label_end'].max()<issue,'Outer purge failed')
            modes=['single_grid','three_windows']
            if direction=='UP':
                chosen={mode:[int(pick([r for r in trials if r['seed']==seed],mode)) for seed in net.SEEDS] for mode in modes}
                prep=net.prep(x,tr);z=net.design(x,tr,prep);ze=net.design(x,te,prep)
                linear=net.linear_fit(z,target.loc[tr]);cache={}
                for seed_index,seed in enumerate(net.SEEDS):
                    for epoch in sorted({chosen[m][seed_index] for m in modes}):
                        print(f'  Outer seed {seed}: {epoch} epochs',flush=True)
                        model,a=net.train('original',linear,z,target.loc[tr].to_numpy(),seed,epochs=epoch)
                        p=net.predict(model,ze)
                        path=out/f'{issue:%Y-%m}_UP_seed{seed}_epoch{epoch}.joblib'
                        joblib.dump(dict(artifact=a,prep=prep,train_keys=tr,issue=issue),path,compress=3)
                        saved=joblib.load(path);replay=net.predict(net.restore(saved['artifact']),net.design(x,te,saved['prep']))
                        np.testing.assert_allclose(replay,p,rtol=0,atol=1e-6)
                        cache[(seed,epoch)]=p
                for mode in modes:
                    frame[f'{direction}_{mode}']=np.mean([cache[(seed,epoch)] for seed,epoch in zip(net.SEEDS,chosen[mode])],axis=0)
                del z,ze,cache,model
            else:
                chosen={mode:float(pick(trials,mode)) for mode in modes}
                outer=recipe.fit_hybrid(x,target,tr,recipe.COLS);parts=recipe.predict_hybrid(outer,x,te)
                path=out/f'{issue:%Y-%m}_DOWN_components.joblib'
                joblib.dump(dict(outer=outer,train_keys=tr,issue=issue),path,compress=3)
                replay=recipe.predict_hybrid(joblib.load(path)['outer'],x,te)
                for k in ['pair','rf']: np.testing.assert_allclose(parts[k],replay[k],rtol=0,atol=1e-6)
                for mode,w in chosen.items(): frame[f'{direction}_{mode}']=(1-w)*parts['pair']+w*parts['rf']
                del outer
            selections.append(dict(issue=str(issue),side=direction,settings=chosen))
            audit.append(dict(issue=str(issue),side=direction,outer_rows=len(tr),outer_max_label_end=str(y.loc[tr,'label_end'].max()),replay='PASS'))
            col='mlp_up_magnitude_bps' if direction=='UP' else 'hybrid_down_magnitude_bps'
            frame[f'{direction}_original_reference']=old.loc[te,col]
            for b in ['mean','median']:
                frame[f'{direction}_baseline_{b}']=old.loc[te,f'{direction.lower()}_train_{b}_baseline_bps']
                np.testing.assert_allclose(frame[f'{direction}_baseline_{b}'],getattr(target.loc[tr],b)(),rtol=0,atol=1e-8)
            print(f'DONE {direction}: {chosen}',flush=True);gc.collect()
        frames.append(frame);frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
        pd.DataFrame(all_trials).to_csv(out/'validation_trials.csv',index=False)
        net.dump(out/'SELECTIONS.json',selections);net.dump(out/'FIT_AUDIT.json',audit)
    pred=pd.concat(frames).sort_index();net.check(pred.index.equals(old.index),'Final index mismatch')
    rows=[]
    for issue,f in [('POOLED',pred)]+list(pred.groupby('issue')):
        for direction in ['UP','DOWN']:
            g=f[f.actual_direction.eq(direction)];actual=g.actual_magnitude_bps
            names=['single_grid','three_windows','original_reference','baseline_mean','baseline_median']
            met={n:net.metrics(actual,g[f'{direction}_{n}']) for n in names}
            for name,m in met.items():
                rows.append(dict(issue=issue,side=direction,model=name,**m,
                    mae_gain_vs_single_grid=met['single_grid']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_original=met['original_reference']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_mean=met['baseline_mean']['mae_bps']-m['mae_bps'],
                    mae_gain_vs_median=met['baseline_median']['mae_bps']-m['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False);pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('ORIGINAL MODELS: ONE VERSUS THREE VALIDATION WINDOWS\n'
        'Same UP MLP and DOWN hybrid components. Only settings selection changes.\n'
        'single_grid and three_windows use identical candidates; original_reference retains original early stopping.\n'
        'UP chooses epochs per seed; DOWN chooses RF weight. Selection minimizes validation MAE, not IC.\n'
        'Positive MAE gains mean improvement. Baseline within-fold IC undefined; pooled IC includes fold shifts.\n'
        'Historical exposed experiment; no significance claims or direction predictor.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False)+'\n\nBY FOLD\n'
        +scores[scores.issue.ne('POOLED')&scores.model.isin(['single_grid','three_windows','original_reference'])].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    net.dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),fold_side_checks=len(audit)))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1): main()
