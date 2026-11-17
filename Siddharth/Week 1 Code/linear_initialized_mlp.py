"""Three-feature fitted-linear skip plus zero-initialized neural correction."""
from pathlib import Path
import copy,json,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import QuantileRegressor
import joblib
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
SRC=HERE/'experiments/baseline_single_feature_v6'
OUT=HERE/'experiments/linear_initialized_mlp_v1'
COLS=q.b.PAIR+['PRICE__vol_ratio_4h_24h']
SEEDS=[48,149,250]
MAX_EPOCHS=48

def design(x,keys,p):
    z=((x.loc[keys,COLS].fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    assert z.shape==(len(keys),3) and np.isfinite(z).all()
    return torch.from_numpy(z)

class Net(nn.Module):
    def __init__(self,coef,intercept):
        super().__init__()
        self.linear=nn.Linear(3,1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(coef,dtype=torch.float32).reshape(1,3))
            self.linear.bias.fill_(float(intercept))
        self.linear.requires_grad_(False)
        self.body=nn.Sequential(nn.Linear(3,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
        self.head=nn.Linear(4,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0)

def infer(model,tx):
    model.eval()
    with torch.no_grad():return np.concatenate([model(tx[i:i+4096]).numpy().astype('float64')*100 for i in range(0,len(tx),4096)])

def linear_fit(x,y,keys,cut):
    path=OUT/'linear'/f'{cut:%Y%m%d}.joblib'
    if path.exists():
        a=joblib.load(path);assert a['train_keys'].equals(keys) and a['cutoff']==cut;return a
    assert y.loc[keys,'label_end'].max()<cut
    p=q.prep(x,keys,COLS);tx=design(x,keys,p)
    ty=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32)/100
    model=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs')
    model.fit(tx.numpy().astype(float),ty.astype(float))
    a=dict(prep=p,coef=model.coef_,intercept=float(model.intercept_),model=model,train_keys=keys,cutoff=cut)
    joblib.dump(a,path,compress=3);print('LINEAR FIT',cut.date(),len(keys),'rows',flush=True);return a

def linear_predict(a,x,keys):
    # Same float32 arithmetic as the network skip; zero residual must match exactly.
    return infer(Net(a['coef'],a['intercept']),design(x,keys,a['prep']))

def fit(x,y,base,seed,path,vk=None,epochs=None):
    if path.exists():return torch.load(path,weights_only=False)
    keys=base['train_keys'];tx=design(x,keys,base['prep']);target=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32)
    ty=torch.from_numpy(target/100);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=Net(base['coef'],base['intercept']);initial=copy.deepcopy(model.linear.state_dict())
    np.testing.assert_allclose(infer(model,tx),linear_predict(base,x,keys),atol=0,rtol=0)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':[p for p in trainable if p.ndim>1],'weight_decay':.01},{'params':[p for p in trainable if p.ndim==1],'weight_decay':0}],lr=.001)
    vx=design(x,vk,base['prep']) if vk is not None else None
    vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
    best=float(abs(infer(model,vx)-vy).mean()) if vx is not None else float('inf')
    state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;history=[dict(epoch=0,validation_mae=best if vx is not None else None)]
    for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
        model.train();order=rng.permutation(len(keys));total=0.
        for offset in range(0,len(keys),512):
            ids=order[offset:offset+512];opt.zero_grad(set_to_none=True)
            loss=abs(model(tx[ids])-ty[ids]).mean();assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(trainable,1);opt.step();total+=float(loss.detach())*len(ids)*100
        value=float(abs(infer(model,vx)-vy).mean()) if vx is not None else None
        history.append(dict(epoch=epoch,train_mae=total/len(keys),validation_mae=value))
        if vx is None or value<best-.01:
            best=value if vx is not None else total/len(keys);best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if vx is not None and stale>=8:break
    for name,value in initial.items():torch.testing.assert_close(model.linear.state_dict()[name],value,atol=0,rtol=0)
    a=dict(state=state,prep=base['prep'],coef=base['coef'],intercept=base['intercept'],seed=seed,train_keys=keys,cutoff=base['cutoff'],validation_keys=vk,epoch=best_epoch,history=history)
    torch.save(a,path);return a

def predict(a,x,keys):
    model=Net(a['coef'],a['intercept']);model.load_state_dict(a['state']);return infer(model,design(x,keys,a['prep']))

def run():
    start=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'linear').mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    contract=dict(features=COLS,seeds=SEEDS,architecture='Frozen fitted 3-input linear skip plus 8/4 SiLU MLP, dropout .05, zero initial correction head; output clipped at zero rather than softplus to reproduce linear predictor exactly.',
        linear='Training-only median regression quantile=.5 alpha=.0001 on training-scaled/clipped inputs and target/100; fits cached only for exactly identical cutoff and training rows.',
        optimizer='AdamW lr .001 weight_decay .01 on nonlinear weights only, gradient clip1, batch512, MAE loss target/100',
        early_stopping='Prior purged validation-quarter MAE, include epoch0, max48, patience8, minimum improvement .01bps. Each outer seed refitted for its chosen epoch count.',
        preprocessing='Training median/IQR with std fallback, clipping [-8,8], exactly three values, no missingness flags or lag features.',
        comparison='Same linear predictor alone, original baseline, three-feature regression/RF, and three-seed mean initialized MLP; same expanding quarterly windows and 18481 historical test rows.',
        limitation='No untouched holdout. Comparison against own linear model isolates added nonlinear correction; comparison against earlier softplus MLP also changes output link and freezes linear branch. No test-driven retries.',source_sha=q.sha(__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    cached=joblib.load(SRC/'prepared.joblib');x=cached[0][COLS];y,dev,ref=cached[1:4]
    old=pd.read_csv(SRC/'predictions.csv',index_col=0,parse_dates=True);val=pd.read_csv(SRC/'validation_predictions.csv',index_col=0,parse_dates=True)
    assert x.index.equals(y.index) and old.index.equals(dev) and len(dev)==18481
    np.testing.assert_allclose(old.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
    np.testing.assert_allclose(old.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    paths=[SRC/n for n in ['prepared.joblib','predictions.csv','validation_predictions.csv','AUDIT.json']];hashes={str(p):q.sha(p) for p in paths}
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework');docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    if not (OUT/'MANIFEST.json').exists():q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']},dependencies={str(p):q.sha(p) for p in [Path(q.__file__),Path(q.b.__file__)]}))
    pieces=[];validations=[];trains=[];decisions=[];seedrows=[];count=0
    for issue in q.STARTS:
        folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        inner,ik,vk,ot,te=q.windows(y,dev,issue);assert y.loc[vk,'label_end'].max()<issue
        q.dump(OUT/'STATUS.json',dict(stage='linear_fit',issue=issue,cutoff=inner))
        bi=linear_fit(x,y,ik,inner)
        vv=val.loc[pd.to_datetime(val.issue,utc=True)==issue,['baseline','vol_ratio','truth','issue']].copy();assert vv.index.equals(vk)
        vv['linear_only']=linear_predict(bi,x,vk)
        inners=[]
        for seed in SEEDS:
            q.dump(OUT/'STATUS.json',dict(stage='inner_mlp',issue=issue,seed=seed,completed_checkpoints=count))
            a=fit(x,y,bi,seed,folder/f'{seed}_inner.pt',vk=vk);inners.append(a);count+=1
            print('MLP',count,'/24',issue.date(),'inner',seed,'epoch',a['epoch'],flush=True)
        d=dict(issue=str(issue),seeds=SEEDS,epochs=[a['epoch'] for a in inners]);q.dump(folder/'selection.json',d);decisions.append(d)
        vv['initialized_mlp']=np.mean([predict(a,x,vk) for a in inners],axis=0)
        q.dump(OUT/'STATUS.json',dict(stage='linear_fit',issue=issue,cutoff=issue))
        bo=linear_fit(x,y,ot,issue)
        pp=old.loc[te,['baseline','vol_ratio','truth','issue']].copy();pp['linear_only']=linear_predict(bo,x,te)
        tt=pd.DataFrame(dict(truth=y.loc[ot,'magnitude_log_bps'],linear_only=linear_predict(bo,x,ot),issue=str(issue)),index=ot)
        preds=[];training=[]
        for a in inners:
            seed=a['seed'];q.dump(OUT/'STATUS.json',dict(stage='outer_mlp',issue=issue,seed=seed,completed_checkpoints=count))
            b=fit(x,y,bo,seed,folder/f'{seed}_outer.pt',epochs=a['epoch']);count+=1
            pred=predict(b,x,te);preds.append(pred);training.append(predict(b,x,ot));seedrows.append(dict(issue=str(issue),seed=seed,**q.b.metric(pred,pp.truth)))
            print('MLP',count,'/24',issue.date(),'outer',seed,'epoch',b['epoch'],flush=True)
        pp['initialized_mlp']=np.mean(preds,axis=0);tt['initialized_mlp']=np.mean(training,axis=0)
        pieces.append(pp);validations.append(vv);trains.append(tt)
    p=pd.concat(pieces);v=pd.concat(validations);t=pd.concat(trains);assert p.index.equals(dev)
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time');q.dump(OUT/'SELECTIONS.json',decisions)
    names=['baseline','vol_ratio','linear_only','initialized_mlp']
    scores=pd.DataFrame([dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]+[dict(split='training_in_sample',model=n,**q.b.metric(t[n],t.truth)) for n in ['linear_only','initialized_mlp']])
    scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'seed_window_scores.csv',index=False)
    extreme=pd.DataFrame([dict(model=n,**q.b.metric(p.loc[p.truth>500,n],p.loc[p.truth>500,'truth'])) for n in names]);extreme.to_csv(OUT/'extreme_scores.csv',index=False)
    uncertainty=[]
    for n in ['linear_only','initialized_mlp']:
        for reference in ['baseline','vol_ratio']+(['linear_only'] if n=='initialized_mlp' else []):
            delta=abs(p[n]-p.truth)-abs(p[reference]-p.truth);block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(model=n,reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    q.dump(OUT/'STATUS.json',dict(stage='audit'));saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True);maxerror=0.
    for d in decisions:
        issue=pd.Timestamp(d['issue']);inner,ik,vk,ot,te=q.windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d')
        for phase,keys,cut,pk,df in [('inner',ik,inner,vk,v),('outer',ot,issue,te,saved)]:
            base=joblib.load(OUT/'linear'/f'{cut:%Y%m%d}.joblib');assert base['train_keys'].equals(keys) and base['cutoff']==cut
            assert y.loc[keys,'label_end'].max()<cut and base['prep']['columns']==COLS
            expected=q.prep(x,keys,COLS)
            for f in ['median','scale']:pd.testing.assert_series_equal(base['prep'][f],expected[f])
            lp=linear_predict(base,x,pk);np.testing.assert_allclose(lp,df.loc[pk,'linear_only'],rtol=0,atol=1e-8)
            preds=[]
            for j,seed in enumerate(SEEDS):
                a=torch.load(folder/f'{seed}_{phase}.pt',weights_only=False)
                assert a['train_keys'].equals(keys) and a['cutoff']==cut and a['seed']==seed and a['epoch']==d['epochs'][j]
                for f in ['median','scale']:pd.testing.assert_series_equal(a['prep'][f],expected[f])
                initial=Net(base['coef'],base['intercept'])
                for f in ['weight','bias']:torch.testing.assert_close(a['state']['linear.'+f],initial.linear.state_dict()[f],atol=0,rtol=0)
                if phase=='inner':
                    assert a['validation_keys'].equals(vk);best=a['history'][0]['validation_mae'];best_epoch=0
                    for row in a['history'][1:]:
                        if row['validation_mae']<best-.01:best=row['validation_mae'];best_epoch=row['epoch']
                    assert a['epoch']==best_epoch
                else:assert a['validation_keys'] is None
                pred=predict(a,x,pk)
                if a['epoch']==0:np.testing.assert_allclose(pred,lp,atol=0,rtol=0)
                preds.append(pred)
            rp=np.mean(preds,axis=0);maxerror=max(maxerror,float(abs(rp-df.loc[pk,'initialized_mlp']).max()));np.testing.assert_allclose(rp,df.loc[pk,'initialized_mlp'],atol=1e-8,rtol=0)
    assert count==24 and all(q.sha(Path(path))==h for path,h in hashes.items())
    audit=dict(status='PASS',neural_checkpoints=count,linear_fits=len(list((OUT/'linear').glob('*.joblib'))),rows=len(p),three_features=True,strict_label_purges=True,training_only_preprocessing=True,linear_initialization_exact=True,frozen_linear_preserved=True,epoch_selection_and_prediction_replay=True,max_replay_error=maxerror,inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='Three-feature linear-initialized residual MLP: completed on real historical data\n\n'+scores.to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nExtreme-move scores:\n'+extreme.to_string(index=False)+'\n\nDescriptive block bootstrap:\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nSelection:\n'+json.dumps(decisions,indent=2)+'\n\nHistorical exposed evaluation; no untouched holdout. Training scores pool repeated timestamps. Linear skip frozen; zero epochs reproduce the fitted linear model.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8');q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,scores=scores.to_dict('records'),seconds=time.monotonic()-start));q.dump(OUT/'STATUS.json',dict(stage='complete'));print(report,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):run()
