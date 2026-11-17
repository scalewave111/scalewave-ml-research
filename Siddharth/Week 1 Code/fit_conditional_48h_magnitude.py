"""Train/evaluate separate UP and DOWN 48h magnitude models.

Prepared for local PowerShell execution; do not infer results until the run completes.
"""
from pathlib import Path
import copy
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.base import clone
from sklearn.linear_model import QuantileRegressor
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
OUT=ROOT/'Code/experiments/conditional_48h_magnitude_v2'
COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h','F4Z__liquidation__venue_breadth_percentile__1h__value','F4Z__funding__funding_persistence_percentile__5m__value','F4Z__basis__basis_level_rz__1h__value']
SEEDS=[48,149,250]
MAX_EPOCHS=48
PATIENCE=8
BATCH=512
WEIGHTS=[0.,.1,.25,.5]
SIDES=['UP','DOWN']

def simple_bps(log_bps):
    return 10000*np.expm1(np.asarray(log_bps,dtype=float)/10000)

def mlp_prep(x,keys):
    a=x.loc[keys,COLS]
    med=a.median().fillna(0.)
    filled=a.fillna(med)
    scale=filled.quantile(.75)-filled.quantile(.25)
    scale=scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    return {'columns':COLS,'median':med,'scale':scale}

def mlp_design(x,keys,p):
    a=x.loc[keys,p['columns']]
    z=((a.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    assert np.isfinite(z).all()
    return torch.from_numpy(z)

class MagnitudeNet(nn.Module):
    """Existing median-linear skip plus 8/4 SiLU MLP; magnitude output >= 0."""
    def __init__(self,coef,intercept):
        super().__init__()
        n=len(coef)
        self.linear=nn.Linear(n,1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(coef,dtype=torch.float32).reshape(1,n))
            self.linear.bias.fill_(float(intercept))
        self.linear.requires_grad_(False)
        self.body=nn.Sequential(nn.Linear(n,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
        self.head=nn.Linear(4,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x):
        return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0.)

def infer_mlp(model,z):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(z[i:i+4096]).cpu().numpy()*100. for i in range(0,len(z),4096)]).astype(float)

def fit_mlp(x,target,keys,cutoff,seed,valid=None,epochs=None):
    """Fit the existing MLP recipe to positive ordinary-bps magnitudes."""
    assert len(keys)>0 and target.loc[keys].notna().all()
    assert target.index.equals(x.index)
    prep=mlp_prep(x,keys)
    tx=mlp_design(x,keys,prep)
    y=target.loc[keys].to_numpy(np.float32)
    y_scaled=y/100.
    baseline=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs')
    baseline.fit(tx.numpy().astype(float),y_scaled.astype(float))
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=MagnitudeNet(baseline.coef_,baseline.intercept_)
    initial_linear=copy.deepcopy(model.linear.state_dict())
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([
        {'params':[p for p in params if p.ndim>1],'weight_decay':.01},
        {'params':[p for p in params if p.ndim==1],'weight_decay':0.}],lr=.001)
    vx=vy=None
    if valid is not None:
        assert len(valid)>0 and target.loc[valid].notna().all()
        vx=mlp_design(x,valid,prep)
        vy=target.loc[valid].to_numpy(float)
    limit=MAX_EPOCHS if epochs is None else int(epochs)
    best_epoch=0
    best_state=copy.deepcopy(model.state_dict())
    best=float(np.mean(np.abs(infer_mlp(model,vx)-vy))) if vx is not None else float('inf')
    stale=0
    for epoch in range(1,limit+1):
        model.train()
        order=rng.permutation(len(keys))
        for start in range(0,len(keys),BATCH):
            ids=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            loss=torch.mean(torch.abs(model(tx[ids])-torch.from_numpy(y_scaled[ids])))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MLP training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(params,1.)
            opt.step()
        if vx is None:
            best_epoch=epoch
            best_state=copy.deepcopy(model.state_dict())
            continue
        val=float(np.mean(np.abs(infer_mlp(model,vx)-vy)))
        if val<best-.01:
            best=val
            best_epoch=epoch
            best_state=copy.deepcopy(model.state_dict())
            stale=0
        else:
            stale+=1
        if stale>=PATIENCE:
            break
    model.load_state_dict(best_state)
    for name,value in initial_linear.items():
        torch.testing.assert_close(model.linear.state_dict()[name],value,atol=0,rtol=0)
    return {'model':model,'prep':prep,'epoch':best_epoch,'validation_mae_bps':best if vx is not None else None}

def predict_mlp(artifact,x,keys):
    return infer_mlp(artifact['model'],mlp_design(x,keys,artifact['prep']))

def fit_hybrid(x,target,keys,cols):
    """Refit the established median-regression + absolute-error RF components."""
    b=q.b
    prep=b.prep(x.loc[keys],cols)
    design=b.design(x.loc[keys],prep)
    yy=target.loc[keys].to_numpy(float)
    models={}
    for kind in ['pair','rf']:
        template_path=b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib'
        if not template_path.exists():
            raise FileNotFoundError(f'Required existing hybrid template not found: {template_path}')
        model=clone(joblib.load(template_path)['model'])
        if kind=='rf':
            model.set_params(n_jobs=1)
        model.fit(design,yy)
        models[kind]=model
    return {'prep':prep,'models':models,'columns':cols}

def predict_hybrid(artifact,x,keys):
    b=q.b
    design=b.design(x.loc[keys],artifact['prep'])
    out={}
    for kind,model in artifact['models'].items():
        if kind=='rf':
            model.n_jobs=1
        out[kind]=np.asarray(model.predict(design),dtype=float)
    return out

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving nonempty output folder: {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    x,y,dev,ref=joblib.load(SRC)
    assert x.index.equals(y.index) and dev.equals(ref.index)
    assert all(c in x.columns for c in COLS) and len(COLS)==8
    assert len(dev)==18481
    assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    signed=y['signed_return_log_bps'].astype(float)
    signed_simple=simple_bps(signed)
    magnitude=pd.Series(np.abs(signed_simple),index=y.index,name='ordinary_magnitude_bps')
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index,name='realized_direction')
    fold_predictions=[]
    fold_records=[]
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        assert len(vk)>0 and len(te)>0
        assert y.loc[ik,'label_end'].max()<inner
        assert y.loc[vk,'label_end'].max()<issue
        assert y.loc[ot,'label_end'].max()<issue
        frame=pd.DataFrame(index=te)
        frame.index.name='decision_time'
        frame['issue']=str(issue)
        frame['actual_direction']=side.loc[te]
        frame['actual_magnitude_bps']=magnitude.loc[te]
        notes={'issue':str(issue),'inner_train_rows':len(ik),'validation_rows':len(vk),'outer_train_rows':len(ot),'test_rows':len(te),'sides':{}}
        for direction in SIDES:
            train_inner=ik[side.loc[ik].to_numpy()==direction]
            valid_side=vk[side.loc[vk].to_numpy()==direction]
            train_outer=ot[side.loc[ot].to_numpy()==direction]
            test_side=te[side.loc[te].to_numpy()==direction]
            assert len(train_inner)>0 and len(valid_side)>0 and len(train_outer)>0 and len(test_side)>0
            target=magnitude
            # MLP: select epoch count by matching-direction validation MAE, then refit.
            inner_mlp=[fit_mlp(x,target,train_inner,inner,seed,valid=valid_side) for seed in SEEDS]
            epochs=[a['epoch'] for a in inner_mlp]
            outer_mlp=[fit_mlp(x,target,train_outer,issue,seed,epochs=ep) for seed,ep in zip(SEEDS,epochs)]
            mlp_pred=np.mean([predict_mlp(a,x,te) for a in outer_mlp],axis=0)
            frame[f'mlp_{direction.lower()}_magnitude_bps']=mlp_pred
            # Hybrid: select RF blend weight on matching-direction validation rows.
            inner_hybrid=fit_hybrid(x,target,train_inner,COLS)
            val_parts=predict_hybrid(inner_hybrid,x,valid_side)
            truth_valid=target.loc[valid_side].to_numpy(float)
            weight=min(WEIGHTS,key=lambda w:(float(np.mean(np.abs((1-w)*val_parts['pair']+w*val_parts['rf']-truth_valid))),w))
            outer_hybrid=fit_hybrid(x,target,train_outer,COLS)
            test_parts=predict_hybrid(outer_hybrid,x,te)
            hybrid_pred=(1-weight)*test_parts['pair']+weight*test_parts['rf']
            frame[f'hybrid_{direction.lower()}_magnitude_bps']=hybrid_pred
            # Train-only constant benchmarks for this side.
            values=target.loc[train_outer].to_numpy(float)
            frame[f'{direction.lower()}_train_mean_baseline_bps']=float(values.mean())
            frame[f'{direction.lower()}_train_median_baseline_bps']=float(np.median(values))
            notes['sides'][direction]={'inner_direction_rows':len(train_inner),'validation_direction_rows':len(valid_side),'outer_direction_rows':len(train_outer),'test_direction_rows':len(test_side),'mlp_epochs_by_seed':epochs,'hybrid_rf_weight':weight,'training_mean_magnitude_bps':float(values.mean()),'training_median_magnitude_bps':float(np.median(values)),'max_train_label_end':str(y.loc[train_outer,'label_end'].max())}
        fold_predictions.append(frame)
        fold_records.append(notes)
        print(f'Fitted both conditional magnitude sides for fold {issue.date()}',flush=True)
    predictions=pd.concat(fold_predictions).sort_index()
    assert predictions.index.equals(dev.sort_values())
    rows=[]
    for direction in SIDES:
        mask=predictions.actual_direction.eq(direction)
        actual=predictions.loc[mask,'actual_magnitude_bps'].to_numpy(float)
        for model,col in [('MLP',f'mlp_{direction.lower()}_magnitude_bps'),('Hybrid RF + median regression',f'hybrid_{direction.lower()}_magnitude_bps'),('Training mean baseline',f'{direction.lower()}_train_mean_baseline_bps'),('Training median baseline',f'{direction.lower()}_train_median_baseline_bps')]:
            pred=predictions.loc[mask,col].to_numpy(float)
            rank=pd.Series(pred).corr(pd.Series(actual),method='spearman') if np.ptp(pred)>0 else np.nan
            rows.append({'side':direction,'model':model,'rows':len(actual),'mae_ordinary_bps':float(np.mean(np.abs(pred-actual))),'rank_ic':float(rank) if np.isfinite(rank) else None,'mean_predicted_magnitude_bps':float(np.mean(pred)),'mean_actual_magnitude_bps':float(np.mean(actual))})
    scores=pd.DataFrame(rows)
    predictions.to_csv(OUT/'predictions.csv',index_label='decision_time')
    scores.to_csv(OUT/'scores.csv',index=False)
    window_rows=[]
    for issue,g in predictions.groupby('issue',sort=True):
        for direction in SIDES:
            mask=g.actual_direction.eq(direction)
            actual=g.loc[mask,'actual_magnitude_bps'].to_numpy(float)
            for model,col in [('MLP',f'mlp_{direction.lower()}_magnitude_bps'),('Hybrid RF + median regression',f'hybrid_{direction.lower()}_magnitude_bps'),('Training mean baseline',f'{direction.lower()}_train_mean_baseline_bps'),('Training median baseline',f'{direction.lower()}_train_median_baseline_bps')]:
                pred=g.loc[mask,col].to_numpy(float)
                window_rows.append({'issue':issue,'side':direction,'model':model,'rows':len(actual),'mae_ordinary_bps':float(np.mean(np.abs(pred-actual)))})
    pd.DataFrame(window_rows).to_csv(OUT/'window_scores.csv',index=False)
    contract={'status':'RUNNER_EXECUTED','target':'ordinary simple-return move magnitude in bps, trained separately on realized UP and DOWN training rows','up_target':'10000*expm1(signed_log_bps/10000) for positive-return rows','down_target':'abs(10000*expm1(signed_log_bps/10000)) for negative-return rows','features':COLS,'data_source':str(SRC),'split':'existing four expanding quarterly folds; training and validation require label_end strictly before the relevant issuance cutoff','mlp':'Existing median-regression linear skip plus 8/4 SiLU MLP, dropout .05, three seeds, MAE training, validation-selected epochs, nonnegative output','hybrid':'Existing median-regression pair plus absolute-error RF; validation selects RF blend weight from 0,.1,.25,.5','evaluation':'MAE and rank IC in ordinary bps, separately among realized UP and DOWN test rows; includes train-only mean and median baselines','limitation':'Historical development rows have prior research exposure; this is a historical evaluation, not fresh independent confirmation or a trading result. The existing MAE/median-regression recipes estimate median-like conditional magnitude, not a conditional mean; assess mean calibration before using these outputs as expected sizes.'}
    (OUT/'CONTRACT.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    summary={'status':'EXECUTED','scores':rows,'folds':fold_records,'limitation':contract['limitation']}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False)+'\n',encoding='utf-8')
    print(scores.to_string(index=False),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
