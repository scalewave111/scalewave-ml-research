"""Fresh 48-hour signed-return comparison; prepared, not executed by this session."""
from pathlib import Path
import copy, json
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
OUT=ROOT/'Code/experiments/signed_48h_mlp_hybrid_v1'
COLS=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h','F4Z__liquidation__venue_breadth_percentile__1h__value','F4Z__funding__funding_persistence_percentile__5m__value','F4Z__basis__basis_level_rz__1h__value']
SEEDS=[48,149,250]
MAX_EPOCHS=48
PATIENCE=8
BATCH=512
WEIGHTS=[0.,.1,.25,.5]

def to_bps(v): return 10000*np.expm1(np.asarray(v,dtype=float)/10000)
def score_bps(actual_log,pred_log): return float(np.mean(np.abs(to_bps(pred_log)-to_bps(actual_log))))

def prep_mlp(x,keys):
    a=x.loc[keys,COLS]; med=a.median().fillna(0.); filled=a.fillna(med)
    scale=(filled.quantile(.75)-filled.quantile(.25)); scale=scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    return {'columns':COLS,'median':med,'scale':scale}
def design_mlp(x,keys,p):
    a=x.loc[keys,p['columns']]; z=((a.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    assert np.isfinite(z).all(); return torch.from_numpy(z)

class SignedNet(nn.Module):
    def __init__(self,coef,intercept):
        super().__init__(); n=len(coef); self.linear=nn.Linear(n,1)
        with torch.no_grad(): self.linear.weight.copy_(torch.as_tensor(coef,dtype=torch.float32).reshape(1,n)); self.linear.bias.fill_(float(intercept))
        self.linear.requires_grad_(False)
        self.body=nn.Sequential(nn.Linear(n,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
        self.head=nn.Linear(4,1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x): return (self.linear(x)+self.head(self.body(x))).squeeze(1)
def infer_mlp(model,z):
    model.eval()
    with torch.no_grad(): return np.concatenate([model(z[i:i+4096]).cpu().numpy()*100. for i in range(0,len(z),4096)]).astype(float)

def fit_mlp(x,y,keys,cutoff,seed,valid=None,epochs=None):
    assert y.loc[keys,'label_end'].max()<cutoff
    p=prep_mlp(x,keys); tx=design_mlp(x,keys,p); raw=y.loc[keys,'signed_return_log_bps'].to_numpy(np.float32); target=raw/100.
    median=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs').fit(tx.numpy().astype(float),target.astype(float))
    torch.manual_seed(seed); rng=np.random.default_rng(seed); model=SignedNet(median.coef_,median.intercept_)
    initial=copy.deepcopy(model.linear.state_dict()); trainable=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW([{'params':[v for v in trainable if v.ndim>1],'weight_decay':.01},{'params':[v for v in trainable if v.ndim==1],'weight_decay':0.}],lr=.001)
    vx=vy=None
    if valid is not None: vx=design_mlp(x,valid,p); vy=y.loc[valid,'signed_return_log_bps'].to_numpy(float)
    limit=MAX_EPOCHS if epochs is None else int(epochs); best_epoch=0; best_state=copy.deepcopy(model.state_dict()); stale=0
    best=score_bps(vy,infer_mlp(model,vx)) if vx is not None else float('inf')
    for ep in range(1,limit+1):
        model.train(); order=rng.permutation(len(keys))
        for start in range(0,len(keys),BATCH):
            ids=order[start:start+BATCH]; opt.zero_grad(set_to_none=True)
            loss=torch.mean(torch.abs(model(tx[ids])-torch.from_numpy(target[ids])))
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite MLP loss')
            loss.backward(); nn.utils.clip_grad_norm_(trainable,1.); opt.step()
        if vx is None: best_epoch=ep; best_state=copy.deepcopy(model.state_dict()); continue
        val=score_bps(vy,infer_mlp(model,vx))
        if val<best-.01: best=val; best_epoch=ep; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=PATIENCE: break
    model.load_state_dict(best_state)
    for k,v in initial.items(): torch.testing.assert_close(model.linear.state_dict()[k],v,atol=0,rtol=0)
    return {'model':model,'prep':p,'epoch':best_epoch}
def predict_mlp(a,x,keys): return infer_mlp(a['model'],design_mlp(x,keys,a['prep']))

def fit_hybrid(x,y,keys,cols):
    b=q.b; p=b.prep(x.loc[keys],cols); z=b.design(x.loc[keys],p); target=y.loc[keys,'signed_return_log_bps'].to_numpy(float); models={}
    for kind in ['pair','rf']:
        path=b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib'
        if not path.exists(): raise FileNotFoundError(f'Missing existing hybrid template: {path}')
        model=clone(joblib.load(path)['model'])
        if kind=='rf': model.set_params(n_jobs=1)
        model.fit(z,target); models[kind]=model
    return {'prep':p,'models':models}
def predict_hybrid(a,x,keys):
    b=q.b; z=b.design(x.loc[keys],a['prep']); out={}
    for k,m in a['models'].items():
        if k=='rf': m.n_jobs=1
        out[k]=np.asarray(m.predict(z),dtype=float)
    return out

def main():
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f'Preserving nonempty output folder: {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    x,y,dev,ref=joblib.load(SRC); assert x.index.equals(y.index) and dev.equals(ref.index)
    assert len(COLS)==8 and all(c in x.columns for c in COLS) and len(dev)==18481
    assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    cutoff=pd.Timestamp('2024-01-01',tz='UTC'); old=y.index[(y.index<cutoff)&(y.label_end<cutoff)]
    frozen_base=float(to_bps(y.loc[old,'signed_return_log_bps']).mean()); predictions=[]; fold_notes=[]; covered=[]
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue); assert len(vk)>0 and len(te)>0
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
        covered.extend(te)
        hi=fit_hybrid(x,y,ik,COLS); hv=predict_hybrid(hi,x,vk); vy=y.loc[vk,'signed_return_log_bps'].to_numpy(float)
        w=min(WEIGHTS,key=lambda a:(score_bps(vy,(1-a)*hv['pair']+a*hv['rf']),a))
        ho=fit_hybrid(x,y,ot,COLS); ht=predict_hybrid(ho,x,te); hp=(1-w)*ht['pair']+w*ht['rf']
        iv=[fit_mlp(x,y,ik,inner,s,valid=vk) for s in SEEDS]; eps=[a['epoch'] for a in iv]
        ov=[fit_mlp(x,y,ot,issue,s,epochs=e) for s,e in zip(SEEDS,eps)]
        mp=np.mean([predict_mlp(a,x,te) for a in ov],axis=0)
        actual=y.loc[te,'signed_return_log_bps'].to_numpy(float)
        fold_mean=float(to_bps(y.loc[ot,'signed_return_log_bps']).mean())
        frozen_log=float(10000*np.log1p(frozen_base/10000)); fold_log=float(10000*np.log1p(fold_mean/10000))
        f=pd.DataFrame({'issue':str(issue),'actual_signed_log_bps':actual,'actual_signed_bps':to_bps(actual),
          'baseline_fixed_log_bps':frozen_log,'baseline_fixed_bps':frozen_base,'baseline_expanding_log_bps':fold_log,'baseline_expanding_bps':fold_mean,
          'mlp_signed_log_bps':mp,'mlp_signed_bps':to_bps(mp),'hybrid_signed_log_bps':hp,'hybrid_signed_bps':to_bps(hp)},index=te)
        f.index.name='decision_time'; predictions.append(f)
        fold_notes.append({'issue':str(issue),'inner_train_rows':len(ik),'validation_rows':len(vk),'outer_train_rows':len(ot),'test_rows':len(te),'hybrid_rf_weight':w,'mlp_epochs_by_seed':eps,'expanding_mean_baseline_bps':fold_mean,'max_train_label_end':str(y.loc[ot,'label_end'].max())})
        print(f'Finished fitting fold {issue.date()}',flush=True)
    p=pd.concat(predictions).sort_index(); assert p.index.equals(dev.sort_values())
    model_cols=[('Fixed pre-2024 mean','baseline_fixed_bps','baseline_fixed_log_bps'),('Expanding mean','baseline_expanding_bps','baseline_expanding_log_bps'),('MLP','mlp_signed_bps','mlp_signed_log_bps'),('Hybrid RF + median regression','hybrid_signed_bps','hybrid_signed_log_bps')]
    rows=[]; windows=[]
    for name,bcol,lcol in model_cols:
        ab=p.actual_signed_bps.to_numpy(float); al=p.actual_signed_log_bps.to_numpy(float); pb=p[bcol].to_numpy(float); pl=p[lcol].to_numpy(float)
        rows.append({'model':name,'rows':len(p),'mae_ordinary_bps':float(np.mean(np.abs(pb-ab))),'mae_signed_log_bps':float(np.mean(np.abs(pl-al))),
          'direction_accuracy':float(np.mean((pl>0)==(al>0))),'rank_ic':float(pd.Series(pb).corr(pd.Series(ab),method='spearman')),'mean_predicted_bps':float(pb.mean()),'mean_actual_bps':float(ab.mean()),'predicted_up_rate':float(np.mean(pl>0)),'actual_up_rate':float(np.mean(al>0))})
        for issue,g in p.groupby('issue',sort=True): windows.append({'issue':issue,'model':name,'rows':len(g),'mae_ordinary_bps':float(np.mean(np.abs(g[bcol]-g.actual_signed_bps))),'direction_accuracy':float(np.mean((g[lcol]>0)==(g.actual_signed_log_bps>0)))})
    scores=pd.DataFrame(rows); p.to_csv(OUT/'predictions.csv',index_label='decision_time'); scores.to_csv(OUT/'scores.csv',index=False); pd.DataFrame(windows).to_csv(OUT/'window_scores.csv',index=False)
    contract={'target':'signed BTCUSDT 48h endpoint log return in bps','features':COLS,'source':str(SRC),'split':'existing four expanding quarterly folds; strict label_end < issuance for training and validation','ordinary_bps_conversion':'per row: 10000*expm1(log_bps/10000)','mlp':'Existing median-regression linear skip plus 8/4 SiLU correction MLP, dropout .05; three seeds; validation selects epoch count; signed output unclamped','hybrid':'Existing median-regression pair plus absolute-error RF; validation selects RF blend weight from 0,.1,.25,.5','metrics':['ordinary-bps MAE','signed log-bps MAE','direction accuracy','rank IC'],'baselines':['fixed pre-2024 mean ordinary return','expanding training mean ordinary return'],'limitation':'Saved later period has prior research exposure; this is a historical evaluation, not fresh independent confirmation or a trading result.'}
    (OUT/'CONTRACT.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    summary={'status':'EXECUTED','fixed_pre2024_mean_baseline_bps':frozen_base,'scores':rows,'folds':fold_notes,'limitation':contract['limitation']}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); (OUT/'RESULTS.txt').write_text(scores.to_string(index=False)+'\n',encoding='utf-8')
    print(scores.to_string(index=False),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main()
