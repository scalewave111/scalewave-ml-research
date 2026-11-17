"""Compare 4/5/8-feature mean-targeting UP and DOWN 48h magnitude models.

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
from sklearn.linear_model import Ridge
import hashlib
import platform
import importlib.metadata
from threadpoolctl import threadpool_limits
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Week 1'/'Code'))
import lstm_walk_forward as q

WORKSPACE=Path(__file__).resolve().parents[1]
ROOT=WORKSPACE.parent/'Week 1'
SRC=ROOT/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
OUT=WORKSPACE/'Code/experiments/conditional_48h_mean_feature_comparison_v1'
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
    """Mean-linear skip plus 8/4 SiLU MLP; magnitude output >= 0."""
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
    """Fit the squared-error MLP recipe to positive ordinary-bps magnitudes."""
    assert len(keys)>0 and target.loc[keys].notna().all()
    assert target.index.equals(x.index)
    prep=mlp_prep(x,keys)
    tx=mlp_design(x,keys,prep)
    y=target.loc[keys].to_numpy(np.float32)
    y_scaled=y/100.
    baseline=Ridge(alpha=1.0)
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
    best=float(np.mean((infer_mlp(model,vx)-vy)**2)) if vx is not None else float('inf')
    stale=0
    for epoch in range(1,limit+1):
        model.train()
        order=rng.permutation(len(keys))
        for start in range(0,len(keys),BATCH):
            ids=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            loss=torch.mean((model(tx[ids])-torch.from_numpy(y_scaled[ids]))**2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MLP training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(params,1.)
            opt.step()
        if vx is None:
            best_epoch=epoch
            best_state=copy.deepcopy(model.state_dict())
            continue
        val=float(np.mean((infer_mlp(model,vx)-vy)**2))
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
    return {'model':model,'prep':prep,'epoch':best_epoch,'validation_mse_bps2':best if vx is not None else None}

def predict_mlp(artifact,x,keys):
    return infer_mlp(artifact['model'],mlp_design(x,keys,artifact['prep']))

def fit_hybrid(x,target,keys,cols):
    """Fit Ridge mean regression + squared-error RF components."""
    b=q.b
    prep=b.prep(x.loc[keys],cols)
    design=b.design(x.loc[keys],prep)
    yy=target.loc[keys].to_numpy(float)
    models={}
    for kind in ['pair','rf']:
        if kind=='pair':
            model=Ridge(alpha=1.0)
        else:
            template_path=b.BASE/'models'/'2024-01__report_outer_rf__expanding.joblib'
            model=clone(joblib.load(template_path)['model'])
            model.set_params(criterion='squared_error',n_jobs=1)
            if isinstance(model.max_samples,int):
                model.set_params(max_samples=min(model.max_samples,len(keys)))
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


ALL_COLS=list(COLS)
PACKS={'four':[ALL_COLS[i] for i in [0,1,3,5]],
       'five':[ALL_COLS[i] for i in [0,1,3,5,4]],'eight':ALL_COLS}

def metrics(actual,pred):
    actual=np.asarray(actual,dtype=float);pred=np.asarray(pred,dtype=float)
    assert len(actual)>0 and np.isfinite(actual).all() and np.isfinite(pred).all()
    err=pred-actual
    rank=pd.Series(pred).corr(pd.Series(actual),method='spearman') if np.ptp(pred)>0 else np.nan
    return dict(rows=len(actual),mae_bps=float(np.abs(err).mean()),
                mse_bps2=float((err**2).mean()),rmse_bps=float(np.sqrt((err**2).mean())),
                rank_ic=float(rank) if np.isfinite(rank) else None,
                mean_predicted_bps=float(pred.mean()),mean_actual_bps=float(actual.mean()),
                mean_bias_bps=float(err.mean()))

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def main():
    global COLS
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving existing output: {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    contract=json.loads((ROOT/'Code/experiments/eight_feature_exhaustive_v1/CONTRACT.json').read_text())
    assert contract['features']==ALL_COLS
    x,y,dev,ref=joblib.load(SRC)
    assert x.index.equals(y.index) and dev.equals(ref.index) and x.index.is_unique
    assert len(dev)==18481 and (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    x=x.replace([np.inf,-np.inf],np.nan)
    signed=y.signed_return_log_bps.astype(float)
    target=pd.Series(np.abs(simple_bps(signed)),index=y.index)
    assert np.isfinite(target).all()
    side=pd.Series(np.where(signed>0,'UP',np.where(signed<0,'DOWN','FLAT')),index=y.index)
    dump('CONTRACT.json',dict(status='STARTED',features=PACKS,seeds=SEEDS,
         target='positive ordinary-bps magnitude conditional on realized direction',
         selection='MLP epochs, hybrid weight, and feature pack selected by preceding validation MSE',
         mlp='Ridge-initialized frozen linear skip plus 8/4 SiLU, squared-error loss',
         hybrid='Ridge alpha=1 plus squared-error RF; existing RF template settings',
         weights=WEIGHTS,purge='label_end strictly before inner/outer cutoff',
         limitation='Mean-targeting objective does not guarantee calibrated means. Previously exposed historical evaluation.',
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    predictions=[];diagnostics=[];selections=[];validation=[]
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner<=vk.min()
        assert y.loc[vk,'label_end'].max()<issue<=te.min()
        assert y.loc[ot,'label_end'].max()<issue
        frame=pd.DataFrame(index=te)
        frame['issue']=str(issue);frame['actual_direction']=side.loc[te]
        frame['actual_magnitude_bps']=target.loc[te]
        for direction in SIDES:
            ti=ik[side.loc[ik].eq(direction)];vs=vk[side.loc[vk].eq(direction)]
            tr=ot[side.loc[ot].eq(direction)];ts=te[side.loc[te].eq(direction)]
            assert min(map(len,[ti,vs,tr,ts]))>0
            for name,constant in [('mean',target.loc[tr].mean()),('median',target.loc[tr].median())]:
                frame[f'{direction}_baseline_{name}']=float(constant)
            candidates={'mlp':[],'hybrid':[]}
            for pack,cols in PACKS.items():
                COLS=list(cols)
                tag=f'{issue:%Y-%m}_{direction}_{pack}'
                print(f'START {tag}: {len(ti)} inner / {len(tr)} outer training rows',flush=True)
                inn=[]
                for seed in SEEDS:
                    a=fit_mlp(x,target,ti,inner,seed,valid=vs);inn.append(a)
                    print(f'  MLP validation seed {seed}, selected epochs {a["epoch"]}',flush=True)
                vp=np.mean([predict_mlp(a,x,vs) for a in inn],axis=0)
                vm=metrics(target.loc[vs],vp)
                candidates['mlp'].append((vm['mse_bps2'],len(cols),pack))
                validation.append(dict(issue=str(issue),side=direction,pack=pack,model='mlp',**vm))
                outer=[fit_mlp(x,target,tr,issue,s,epochs=a['epoch']) for s,a in zip(SEEDS,inn)]
                trainp=np.mean([predict_mlp(a,x,tr) for a in outer],axis=0)
                testp=np.mean([predict_mlp(a,x,te) for a in outer],axis=0)
                frame[f'{direction}_mlp_{pack}']=testp
                joblib.dump(outer,OUT/f'{tag}_mlp.joblib',compress=3)
                hin=fit_hybrid(x,target,ti,cols)
                parts=predict_hybrid(hin,x,vs)
                w=min(WEIGHTS,key=lambda w:(metrics(target.loc[vs],np.maximum(0,(1-w)*parts['pair']+w*parts['rf']))['mse_bps2'],w))
                hv=np.maximum(0,(1-w)*parts['pair']+w*parts['rf'])
                hm=metrics(target.loc[vs],hv)
                candidates['hybrid'].append((hm['mse_bps2'],len(cols),pack))
                validation.append(dict(issue=str(issue),side=direction,pack=pack,model='hybrid',rf_weight=w,**hm))
                hout=fit_hybrid(x,target,tr,cols)
                def hp(keys):
                    p=predict_hybrid(hout,x,keys)
                    return np.maximum(0,(1-w)*p['pair']+w*p['rf'])
                htrain=hp(tr);htest=hp(te)
                frame[f'{direction}_hybrid_{pack}']=htest
                joblib.dump(dict(artifact=hout,weight=w),OUT/f'{tag}_hybrid.joblib',compress=3)
                for model,tpr,vpr,epr in [('mlp',trainp,vp,testp),('hybrid',htrain,hv,htest)]:
                    for stage,actual,pr in [('train_in_sample',target.loc[tr],tpr),('validation_selection',target.loc[vs],vpr),('test',target.loc[ts],epr[side.loc[te].eq(direction).to_numpy()])]:
                        diagnostics.append(dict(issue=str(issue),side=direction,pack=pack,model=model,stage=stage,**metrics(actual,pr)))
                dump(f'{tag}_settings.json',dict(columns=cols,epochs=[a['epoch'] for a in inn],rf_weight=w,inner_max_label_end=y.loc[ti,'label_end'].max(),outer_max_label_end=y.loc[tr,'label_end'].max(),rf_params=hout['models']['rf'].get_params()))
                print(f'DONE {tag}',flush=True)
            for model in ['mlp','hybrid']:
                mse,_,winner=min(candidates[model])
                frame[f'{direction}_{model}_selected']=frame[f'{direction}_{model}_{winner}']
                selections.append(dict(issue=str(issue),side=direction,model=model,pack=winner,validation_mse_bps2=mse))
        frame.to_csv(OUT/f'{issue:%Y-%m}_predictions.csv',index_label='decision_time')
        predictions.append(frame)
        pd.DataFrame(diagnostics).to_csv(OUT/'fit_diagnostics.csv',index=False)
        pd.DataFrame(validation).to_csv(OUT/'validation_scores.csv',index=False)
        dump('SELECTIONS.json',selections)
        print(f'Completed fold {issue.date()}',flush=True)
    pred=pd.concat(predictions).sort_index()
    assert pred.index.equals(dev.sort_values())
    pred.to_csv(OUT/'predictions.csv',index_label='decision_time')
    rows=[];windows=[]
    for direction in SIDES:
        columns=[c for c in pred.columns if c.startswith(direction+'_')]
        for col in columns:
            m=pred.actual_direction.eq(direction)
            rows.append(dict(side=direction,model=col[len(direction)+1:],**metrics(pred.loc[m,'actual_magnitude_bps'],pred.loc[m,col])))
            for issue,g in pred.groupby('issue'):
                m=g.actual_direction.eq(direction)
                windows.append(dict(issue=issue,side=direction,model=col[len(direction)+1:],**metrics(g.loc[m,'actual_magnitude_bps'],g.loc[m,col])))
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/'scores.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'window_scores.csv',index=False)
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False),encoding='utf-8')
    dump('COMPLETE.json',dict(status='COMPLETED',rows=len(pred),folds=len(q.STARTS),artifact_replay_audited=False))
    print(scores.to_string(index=False),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
