"""Controlled UP MLP regularization comparison; NEW TRAINING on execution.

Save in Week 2/Code. Uses existing Week 1 prepared data and raw forecasts.
Original 8/4 MLP; matrix weight decay .01/.05/.10; original three seeds.
UP only. No architecture changes, feature search, scaling, or new model blends.
Dependencies: numpy pandas scipy scikit-learn torch joblib threadpoolctl.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import QuantileRegressor
import joblib
from threadpoolctl import threadpool_limits

COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
      'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
      'PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h',
      'F4Z__liquidation__venue_breadth_percentile__1h__value',
      'F4Z__funding__funding_persistence_percentile__5m__value',
      'F4Z__basis__basis_level_rz__1h__value']
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
SETTINGS={'original': .01, 'decay_005': .05, 'decay_010': .10}
SEEDS=[48,149,250]
MAX_EPOCHS=48
PATIENCE=8
BATCH=512


def check(ok,message):
    if not ok: raise ValueError(message)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    check(y.shape==p.shape and len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid score arrays')
    ic=pd.Series(p).corr(pd.Series(y),method='spearman') if np.ptp(p)>0 and np.ptp(y)>0 else np.nan
    return dict(rows=len(y),mae_bps=float(np.abs(y-p).mean()),rank_ic=float(ic) if np.isfinite(ic) else None)


def prep(x,keys):
    a=x.loc[keys,COLS];med=a.median().fillna(0.);filled=a.fillna(med)
    scale=filled.quantile(.75)-filled.quantile(.25)
    scale=scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    return dict(columns=COLS,median=med,scale=scale)


def design(x,keys,p):
    a=x.loc[keys,p['columns']]
    z=((a.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    check(np.isfinite(z).all(),'Nonfinite design')
    return torch.from_numpy(z)


def linear_fit(z,target):
    # Match original MLP's float32 target scaling and quantile recipe.
    yy=np.asarray(target,np.float32)/100.
    model=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs')
    model.fit(z.numpy().astype(float),yy.astype(float))
    return dict(coef=model.coef_.copy(),intercept=float(model.intercept_))


class MagnitudeNet(nn.Module):
    def __init__(self,setting,linear):
        super().__init__();n=len(linear['coef'])
        self.linear=nn.Linear(n,1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(linear['coef'],dtype=torch.float32).reshape(1,n))
            self.linear.bias.fill_(linear['intercept'])
        self.linear.requires_grad_(False)
        self.body=nn.Sequential(nn.Linear(n,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
        self.head=nn.Linear(4,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0.)


def predict(model,z):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(z[i:i+4096]).numpy()*100. for i in range(0,len(z),4096)]).astype(float)


def train(setting,linear,z,y,seed,valid=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=MagnitudeNet(setting,linear)
    frozen=copy.deepcopy(model.linear.state_dict())
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([
        {'params':[p for p in params if p.ndim>1],'weight_decay':SETTINGS[setting]},
        {'params':[p for p in params if p.ndim==1],'weight_decay':0.}],lr=.001)
    yy=np.asarray(y,np.float32)/100.
    state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;history=[]
    if valid is not None:
        vz,vy=valid;best=metrics(vy,predict(model,vz))['mae_bps']
        history.append(dict(epoch=0,validation_mae_bps=best))
    else: best=None
    limit=MAX_EPOCHS if epochs is None else int(epochs)
    check(limit>=0,'Negative epoch count')
    for epoch in range(1,limit+1):
        model.train();order=rng.permutation(len(y))
        for start in range(0,len(y),BATCH):
            ids=order[start:start+BATCH];opt.zero_grad(set_to_none=True)
            loss=(model(z[ids])-torch.from_numpy(yy[ids])).abs().mean()
            check(bool(torch.isfinite(loss)),'Nonfinite neural-network loss')
            loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
        if valid is None:
            best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        else:
            val=metrics(vy,predict(model,vz))['mae_bps']
            history.append(dict(epoch=epoch,validation_mae_bps=val))
            if val<best-.01:
                best=val;best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
            else: stale+=1
            if stale>=PATIENCE: break
        if epoch==1 or epoch%8==0:
            print(f'    seed {seed}, epoch {epoch}/{limit}'+(f', validation MAE {val:.3f}' if valid is not None else ''),flush=True)
    model.load_state_dict(state)
    for name,value in frozen.items():
        torch.testing.assert_close(model.linear.state_dict()[name],value,rtol=0,atol=0)
    artifact=dict(setting=setting,linear=linear,seed=seed,epochs=best_epoch,
                  state={k:v.cpu().numpy().copy() for k,v in state.items()},history=history,
                  validation_mae_bps=best,trainable_parameters=sum(p.numel() for p in params))
    return model,artifact


def restore(artifact):
    model=MagnitudeNet(artifact['setting'],artifact['linear'])
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in artifact['state'].items()})
    return model.eval()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();workspace=args.workspace.resolve();week1=workspace.parent/'Week 1'
    out=args.out or workspace/'Code/experiments/mlp_regularization_mae_v1'
    check(not out.exists(),f'Preserving output: {out}; use --out with a new folder')
    data=week1/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    oldpath=week1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/predictions.csv'
    x,y,dev,ref=joblib.load(data)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,'Invalid dataset index')
    check(len(dev)==18481 and dev.equals(ref.index),'Unexpected evaluation universe')
    check((y.label_end==y.index+pd.Timedelta(hours=48)).all(),'Unexpected horizon')
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(10000*np.expm1(signed.to_numpy()/10000)),index=y.index)
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    check(np.isfinite(target).all(),'Invalid targets')
    x=x[COLS].replace([np.inf,-np.inf],np.nan)
    old=pd.read_csv(oldpath,index_col='decision_time');old.index=pd.to_datetime(old.index,utc=True)
    check(old.index.equals(dev.sort_values()),'Reference timestamps mismatch')
    np.testing.assert_allclose(old.actual_magnitude_bps,target.loc[old.index],rtol=0,atol=1e-8)
    check(np.array_equal(old.actual_direction,side.loc[old.index]),'Reference direction mismatch')
    check(side.loc[dev].value_counts().to_dict()=={'UP':10191,'DOWN':8290},'Unexpected side counts')
    gru_path=workspace/'Code/experiments/gru_conditional_mae_v1/predictions.csv'
    gru=None
    if gru_path.exists():
        gru=pd.read_csv(gru_path,index_col='decision_time');gru.index=pd.to_datetime(gru.index,utc=True)
        check(gru.index.equals(old.index),'GRU timestamps mismatch')
        np.testing.assert_allclose(gru.actual_magnitude_bps,old.actual_magnitude_bps,rtol=0,atol=1e-8)
        check(np.array_equal(gru.actual_direction,old.actual_direction),'GRU directions mismatch')
        check(pd.to_datetime(gru.issue,utc=True).equals(pd.to_datetime(old.issue,utc=True)),'GRU fold mismatch')
    out.mkdir(parents=True)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'CONTRACT.json',dict(settings=SETTINGS,features=COLS,seeds=SEEDS,
        target='positive ordinary-bps 48h endpoint magnitude conditional on realized side',
        objective='MAE',epochs=MAX_EPOCHS,patience=PATIENCE,minimum_validation_improvement_bps=.01,
        optimizer='AdamW lr=.001; matrix decay from SETTINGS; vector decay=0; gradient clip=1',dropout=.05,
        linear='Frozen QuantileRegressor quantile=.5 alpha=.0001; fitted once per side/phase and shared across settings and seeds',
        selection='Each seed selects epochs on first two months of previous quarter; mean-of-original-three-seeds validation MAE selects decay; IC breaks exact ties, then prefer weaker decay',
        purge='Training label_end strictly before inner/outer cutoff; validation label_end before outer cutoff',
        preprocessing='Training-only median and IQR/std; clip [-8,8]; matches original MLP',
        limitation='Repeatedly exposed historical tests. Regularization search is exploratory; no guarantee of improvement.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','joblib']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    frames=[];validation=[];decisions=[];audit=[]
    for issue in STARTS:
        inner=issue-pd.DateOffset(months=3)
        ik=y.index[y.label_end<inner];ot=y.index[y.label_end<issue]
        vk=y.index[(y.index>=inner)&(y.index<issue-pd.DateOffset(months=1))&(y.label_end<issue)]
        end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(minutes=30)
        te=dev[(dev>=issue)&(dev<end)]
        check(len(te)>0 and len(vk)>0 and y.loc[ik,'label_end'].max()<inner and y.loc[ot,'label_end'].max()<issue,'Invalid purged fold')
        check(pd.to_datetime(old.loc[te,'issue'],utc=True).eq(issue).all(),'Reference fold differs')
        frame=pd.DataFrame(dict(issue=str(issue),actual_direction=side.loc[te],actual_magnitude_bps=target.loc[te]),index=te)
        for direction in ['UP']:
            ti=ik[side.loc[ik].eq(direction)];va=vk[side.loc[vk].eq(direction)];tr=ot[side.loc[ot].eq(direction)]
            check(min(len(ti),len(va),len(tr))>=100,'Too few rows')
            print(f'START {issue.date()} {direction}',flush=True)
            pin=prep(x,ti);pout=prep(x,tr)
            zi=design(x,ti,pin);zv=design(x,va,pin);zo=design(x,tr,pout);ze=design(x,te,pout)
            print('  Fitting shared inner/outer median regression',flush=True)
            li=linear_fit(zi,target.loc[ti]);lo=linear_fit(zo,target.loc[tr])
            candidates=[]
            for setting in SETTINGS:
                print(f'  {setting}: inner validation fits',flush=True)
                inner_art=[];vp=[]
                for seed in SEEDS:
                    model,a=train(setting,li,zi,target.loc[ti].to_numpy(),seed,valid=(zv,target.loc[va].to_numpy()))
                    vp.append(predict(model,zv));inner_art.append(a)
                    print(f'    seed {seed}: selected {a["epochs"]} epochs',flush=True)
                val=metrics(target.loc[va],np.mean(vp,axis=0))
                rec=dict(issue=str(issue),side=direction,setting=setting,**val)
                validation.append(rec);candidates.append(rec)
                print(f'  {setting}: outer fits',flush=True)
                outer_art=[];tp=[]
                for seed,ia in zip(SEEDS,inner_art):
                    model,a=train(setting,lo,zo,target.loc[tr].to_numpy(),seed,epochs=ia['epochs'])
                    tp.append(predict(model,ze));outer_art.append(a)
                prediction=np.mean(tp,axis=0)
                frame[f'{direction}_{setting}']=prediction
                if setting=='original':
                    reference=old.loc[te,f'mlp_{direction.lower()}_magnitude_bps'].to_numpy()
                    discrepancy=float(np.max(np.abs(reference-prediction)))
                    check(discrepancy<=.001,
                          f'Original MLP reproduction failed: max difference {discrepancy:.6f} bps. '
                          'Stop and inspect dependency versions / data; do not compare changed recipes.')
                else: discrepancy=None
                with torch.no_grad():
                    initial=predict(MagnitudeNet(setting,lo),ze)
                correction=float(np.mean(np.abs(prediction-initial)))
                path=out/f'{issue:%Y-%m}_{direction}_{setting}.joblib'
                joblib.dump(dict(inner=inner_art,outer=outer_art,inner_prep=pin,outer_prep=pout,
                    inner_train_keys=ti,outer_train_keys=tr,validation_keys=va,issue=issue,
                    inner_max_label_end=y.loc[ti,'label_end'].max(),outer_max_label_end=y.loc[tr,'label_end'].max()),path,compress=3)
                saved=joblib.load(path)
                replay=np.mean([predict(restore(a),design(x,te,saved['outer_prep'])) for a in saved['outer']],axis=0)
                error=float(np.max(np.abs(replay-prediction)))
                np.testing.assert_allclose(replay,prediction,rtol=0,atol=1e-6)
                audit.append(dict(issue=str(issue),side=direction,setting=setting,max_replay_error_bps=error,
                    original_reference_max_error_bps=discrepancy,
                    mean_abs_adjustment_from_initial_bps=correction))
                del model,saved,inner_art,outer_art;gc.collect()
            winner=min(candidates,key=lambda r:(r['mae_bps'],-r['rank_ic'] if r['rank_ic'] is not None else float('inf'),SETTINGS[r['setting']]))
            frame[f'{direction}_validation_selected']=frame[f'{direction}_{winner["setting"]}']
            decisions.append(winner)
            for name in ['mlp','hybrid']:
                frame[f'{direction}_existing_{name}']=old.loc[te,f'{name}_{direction.lower()}_magnitude_bps']
            if gru is not None: frame[f'{direction}_existing_gru']=gru.loc[te,f'{direction}_gru']
            for base,value in [('mean',target.loc[tr].mean()),('median',target.loc[tr].median())]:
                frame[f'{direction}_baseline_{base}']=float(value)
                np.testing.assert_allclose(old.loc[te,f'{direction.lower()}_train_{base}_baseline_bps'],value,rtol=0,atol=1e-8)
            print(f'DONE {direction}: validation selected {winner["setting"]}',flush=True)
            del zi,zv,zo,ze;gc.collect()
        frame.to_csv(out/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time');frames.append(frame)
        pd.DataFrame(validation).to_csv(out/'validation_scores.csv',index=False)
        pd.DataFrame(audit).to_csv(out/'replay_audit.csv',index=False);dump(out/'SELECTIONS.json',decisions)
    pred=pd.concat(frames).sort_index();check(pred.index.equals(old.index),'Final timestamps mismatch')
    rows=[]
    for fold,f in [('POOLED',pred)]+[(k,v) for k,v in pred.groupby('issue')]:
        for direction in ['UP']:
            g=f[f.actual_direction.eq(direction)];a=g.actual_magnitude_bps
            baseline={b:metrics(a,g[f'{direction}_baseline_{b}'])['mae_bps'] for b in ['mean','median']}
            current='existing_mlp' if direction=='UP' else 'existing_hybrid'
            refmae=metrics(a,g[f'{direction}_{current}'])['mae_bps']
            for col in [c for c in pred if c.startswith(direction+'_')]:
                met=metrics(a,g[col]);rows.append(dict(issue=fold,side=direction,model=col[len(direction)+1:],**met,
                    mae_gain_vs_mean=baseline['mean']-met['mae_bps'],mae_gain_vs_median=baseline['median']-met['mae_bps'],
                    current_reference=current,mae_gain_vs_current=refmae-met['mae_bps']))
    scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
    pred.to_csv(out/'predictions.csv',index_label='decision_time')
    summary=('UP SMALL MLP REGULARIZATION COMPARISON\nPositive MAE gain means improvement; higher positive Rank IC is better.\n'
        'Original 8/4 architecture and frozen median regression; original three seeds retained.\n'
        'Zero epochs retains the initial linear prediction. Validation selects epochs and weight decay.\n'
        'No test-based tuning inside runner; these historical periods have prior exposure.\n'
        'Higher decay penalizes matrix weights; smaller forecast adjustments are not guaranteed.\nPooled baseline IC is not within-fold ranking; no significance claims.\n\n'
        +scores[scores.issue.eq('POOLED')].to_string(index=False))
    (out/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    dump(out/'COMPLETE.json',dict(status='COMPLETED',rows=len(pred),folds=4,replay_checks=len(audit),gru_reference_included=gru is not None))
    print(summary);print(f'Outputs: {out}')


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1): main()
