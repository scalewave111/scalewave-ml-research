"""Fixed three-feature residual MLP versus small patch-based transformer."""
from pathlib import Path
import copy,json,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
SRC=HERE/'experiments/baseline_single_feature_v6'
OUT=HERE/'experiments/neural_snapshot_vs_sequence_v1'
COLS=q.b.PAIR+['PRICE__vol_ratio_4h_24h']
SEEDS=[48,149,250]
KINDS=['residual_mlp','patch_transformer']
STEPS=48
PATCH=8
MAX_EPOCHS=48
PATIENCE=8
BATCH=512

class Net(nn.Module):
    def __init__(self,kind,median):
        super().__init__();self.kind=kind
        self.linear=nn.Linear(3,1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias,float(np.log(np.expm1(median/100))))
        if kind=='residual_mlp':
            self.body=nn.Sequential(nn.Linear(3,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
            self.head=nn.Linear(4,1)
        else:
            self.embed=nn.Linear(PATCH,16)
            self.position=nn.Parameter(torch.zeros(1,STEPS//PATCH,16))
            nn.init.normal_(self.position,std=.02)
            layer=nn.TransformerEncoderLayer(d_model=16,nhead=2,dim_feedforward=32,dropout=.05,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
            self.head=nn.Linear(3*(STEPS//PATCH)*16,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        now=x[:,-1,:]
        if self.kind=='residual_mlp':z=self.body(now)
        else:
            # Shared channel encoder, nonoverlapping temporal patches; no future inputs.
            z=x.transpose(1,2).reshape(-1,STEPS//PATCH,PATCH)
            z=self.encoder(self.embed(z)+self.position).reshape(len(x),-1)
        return nn.functional.softplus(self.linear(now)+self.head(z)).squeeze(1)

def design(x,keys,p,kind):
    steps=STEPS if kind=='patch_transformer' else 1
    stamps=keys.asi8[:,None]-np.arange(steps-1,-1,-1)[None,:]*pd.Timedelta(minutes=30).value
    assert (stamps<=keys.asi8[:,None]).all()
    raw=x[COLS].reindex(pd.to_datetime(stamps.ravel(),utc=True))
    z=((raw.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    assert np.isfinite(z).all()
    return torch.from_numpy(z.reshape(len(keys),steps,3))

def infer(model,x):
    model.eval()
    with torch.no_grad():return np.concatenate([model(x[i:i+1024]).numpy().astype('float64')*100 for i in range(0,len(x),1024)])

def fit(x,y,keys,cut,kind,seed,path,vk=None,epochs=None):
    if path.exists():return torch.load(path,weights_only=False)
    assert y.loc[keys,'label_end'].max()<cut
    p=q.prep(x,keys,COLS);tx=design(x,keys,p,kind)
    target=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32);ty=torch.from_numpy(target/100)
    median=float(np.median(target));torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=Net(kind,median)
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>1],'weight_decay':.01},{'params':[p for p in model.parameters() if p.ndim==1],'weight_decay':0}],lr=.001)
    vx=design(x,vk,p,kind) if vk is not None else None
    vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
    best=float(abs(infer(model,vx)-vy).mean()) if vx is not None else float('inf')
    state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0
    history=[dict(epoch=0,validation_mae=best if vx is not None else None)]
    begin=time.monotonic()
    for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
        model.train();order=rng.permutation(len(keys));total=0.
        for offset in range(0,len(keys),BATCH):
            ids=order[offset:offset+BATCH];opt.zero_grad(set_to_none=True)
            loss=abs(model(tx[ids])-ty[ids]).mean();assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=float(loss.detach())*len(ids)*100
        value=float(abs(infer(model,vx)-vy).mean()) if vx is not None else None
        history.append(dict(epoch=epoch,train_mae=total/len(keys),validation_mae=value))
        if vx is None or value<best-.01:
            best=value if vx is not None else total/len(keys);best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if vx is not None and stale>=PATIENCE:break
    a=dict(kind=kind,seed=seed,median=median,prep=p,state=state,train_keys=keys,cutoff=cut,validation_keys=vk,epoch=best_epoch,history=history,parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-begin)
    torch.save(a,path);return a

def predict(a,x,keys):
    model=Net(a['kind'],a['median']);model.load_state_dict(a['state'])
    return infer(model,design(x,keys,a['prep'],a['kind']))

def run():
    begin=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    contract=dict(features=COLS,seeds=SEEDS,architectures={'residual_mlp':'3 -> 8 -> 4 -> 1 SiLU, linear skip, softplus nonnegative output','patch_transformer':'48 half-hour observations ending at origin; 6 nonoverlapping patches of 8; shared channel encoder; d_model16, heads2, one encoder layer, feedforward32; channel concatenation head plus current-value linear skip; softplus output'},
        target='BTCUSDT absolute 48-hour endpoint log return, log bps',optimizer='AdamW lr .001, weight_decay .01 weights only, gradient clip1, dropout .05, MAE loss on target/100',
        preprocessing='Training-only median/IQR with std fallback; clip [-8,8]; no extra features or missingness indicators; unavailable historical timestamps imputed with training median',
        early_stopping=dict(max_epochs=MAX_EPOCHS,patience=PATIENCE,min_improvement_bps=.01,include_epoch_zero=True),
        selection='Fixed architectures and settings; each seed chooses epoch by prior purged validation-quarter MAE; outer fit repeats chosen epoch count on expanding history; average three seeds. No model/test-driven hyperparameter search.',
        scope='Controlled architectural comparison, not an exact reproduction of full PatchTST; differences include sequence input and capacity. Sequence benefit cannot be attributed solely to attention.',
        limitation='Already-examined historical evaluation; no untouched holdout. GARCH not included in this neural comparison.',source_sha=q.sha(__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    cached=joblib.load(SRC/'prepared.joblib');x=cached[0][COLS].copy();y,dev,ref=cached[1:4];del cached
    old=pd.read_csv(SRC/'predictions.csv',index_col=0,parse_dates=True)
    val=pd.read_csv(SRC/'validation_predictions.csv',index_col=0,parse_dates=True)
    assert x.index.equals(y.index) and x.index.is_unique and old.index.equals(dev) and len(dev)==18481
    assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
    np.testing.assert_allclose(old.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(old.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
    paths=[SRC/n for n in ['prepared.joblib','predictions.csv','validation_predictions.csv','AUDIT.json']]
    hashes={str(p):q.sha(p) for p in paths}
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    if not (OUT/'MANIFEST.json').exists():q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','joblib']},dependencies={str(p):q.sha(p) for p in [Path(q.__file__),Path(q.b.__file__)]}))
    # Verify temporal input construction by prefix truncation and future perturbation.
    cut=dev[len(dev)//2];keys=pd.DatetimeIndex([cut]);prep=q.prep(x,y.index[y.label_end<q.STARTS[0]],COLS)
    for kind in KINDS:
        expected=design(x,keys,prep,kind)
        torch.testing.assert_close(design(x.loc[:cut],keys,prep,kind),expected,rtol=0,atol=0)
        changed=x.copy();changed.loc[changed.index>cut]=1e10
        torch.testing.assert_close(design(changed,keys,prep,kind),expected,rtol=0,atol=0)
        model=Net(kind,200);pred=model(expected);pred.sum().backward();assert torch.isfinite(pred).all()
    q.dump(OUT/'PREFLIGHT.json',dict(status='PASS',sequence_prefix=True,future_perturbation=True,forward_backward=True))
    pieces=[];validations=[];trains=[];decisions=[];seed_rows=[];fits=0
    for issue in q.STARTS:
        folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        assert y.loc[vk,'label_end'].max()<issue
        pp=old.loc[te,['baseline','vol_ratio','truth','issue']].copy()
        vv=val.loc[pd.to_datetime(val.issue,utc=True)==issue,['baseline','vol_ratio','truth','issue']].copy();assert vv.index.equals(vk)
        tt=pd.DataFrame(dict(truth=y.loc[ot,'magnitude_log_bps'],issue=str(issue)),index=ot)
        for kind in KINDS:
            inners=[];outer_preds=[];train_preds=[]
            for seed in SEEDS:
                q.dump(OUT/'STATUS.json',dict(stage='inner_training',issue=issue,kind=kind,seed=seed,completed_fits=fits))
                a=fit(x,y,ik,inner,kind,seed,folder/f'{kind}_{seed}_inner.pt',vk=vk);inners.append(a);fits+=1
                print('FIT',fits,'/48',issue.date(),kind,seed,'inner epoch',a['epoch'],flush=True)
            decision=dict(issue=str(issue),kind=kind,seeds=SEEDS,epochs=[a['epoch'] for a in inners],parameters=inners[0]['parameters'])
            q.dump(folder/f'{kind}_selection.json',decision);decisions.append(decision)
            vv[kind]=np.mean([predict(a,x,vk) for a in inners],axis=0)
            for a in inners:
                seed=a['seed'];q.dump(OUT/'STATUS.json',dict(stage='outer_training',issue=issue,kind=kind,seed=seed,completed_fits=fits))
                b=fit(x,y,ot,issue,kind,seed,folder/f'{kind}_{seed}_outer.pt',epochs=a['epoch']);fits+=1
                prediction=predict(b,x,te);outer_preds.append(prediction);train_preds.append(predict(b,x,ot))
                seed_rows.append(dict(issue=str(issue),model=kind,seed=seed,**q.b.metric(prediction,pp.truth)))
                print('FIT',fits,'/48',issue.date(),kind,seed,'outer epoch',b['epoch'],flush=True)
            pp[kind]=np.mean(outer_preds,axis=0);tt[kind]=np.mean(train_preds,axis=0)
        pieces.append(pp);validations.append(vv);trains.append(tt)
    p=pd.concat(pieces);v=pd.concat(validations);t=pd.concat(trains);assert p.index.equals(dev)
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
    names=['baseline','vol_ratio']+KINDS
    scores=pd.DataFrame([dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]+[dict(split='training_in_sample',model=n,**q.b.metric(t[n],t.truth)) for n in KINDS])
    scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(OUT/'seed_window_scores.csv',index=False);q.dump(OUT/'SELECTIONS.json',decisions)
    extreme=pd.DataFrame([dict(model=n,**q.b.metric(p.loc[p.truth>500,n],p.loc[p.truth>500,'truth'])) for n in names]);extreme.to_csv(OUT/'extreme_scores.csv',index=False)
    uncertainty=[]
    for n in KINDS:
        for reference in ['baseline','vol_ratio']+(['residual_mlp'] if n=='patch_transformer' else []):
            delta=abs(p[n]-p.truth)-abs(p[reference]-p.truth)
            block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count'])
            ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            uncertainty.append(dict(model=n,reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    q.dump(OUT/'STATUS.json',dict(stage='replay_audit',completed_fits=fits))
    count=0;maxerror=0.
    saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d')
        for kind in KINDS:
            replay=[];vr=[]
            decision=json.loads((folder/f'{kind}_selection.json').read_text())
            for j,seed in enumerate(SEEDS):
                for phase,keys,cut in [('inner',ik,inner),('outer',ot,issue)]:
                    a=torch.load(folder/f'{kind}_{seed}_{phase}.pt',weights_only=False)
                    assert a['train_keys'].equals(keys) and a['cutoff']==cut and y.loc[keys,'label_end'].max()<cut
                    assert a['kind']==kind and a['seed']==seed and a['prep']['columns']==COLS
                    expected=q.prep(x,keys,COLS)
                    for f in ['median','scale']:pd.testing.assert_series_equal(a['prep'][f],expected[f])
                    assert a['epoch']==decision['epochs'][j]
                    if phase=='outer':assert a['validation_keys'] is None;replay.append(predict(a,x,te))
                    else:
                        assert a['validation_keys'].equals(vk)
                        best=a['history'][0]['validation_mae'];best_epoch=0
                        for row in a['history'][1:]:
                            if row['validation_mae']<best-.01:best=row['validation_mae'];best_epoch=row['epoch']
                        assert best_epoch==a['epoch'];vr.append(predict(a,x,vk))
                    count+=1
            rp=np.mean(replay,axis=0);maxerror=max(maxerror,float(abs(rp-saved.loc[te,kind]).max()))
            np.testing.assert_allclose(rp,saved.loc[te,kind],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean(vr,axis=0),v.loc[vk,kind],atol=1e-8,rtol=0)
    assert count==fits==48 and all(q.sha(Path(path))==h for path,h in hashes.items())
    audit=dict(status='PASS',checkpoints=count,rows=len(p),three_source_features=True,strict_label_purges=True,training_only_preprocessing=True,causal_sequences=True,epoch_selection_replayed=True,max_prediction_replay_error=maxerror,original_inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='Three-feature neural comparison: residual MLP versus small PatchTST-style transformer\n\n'+scores.to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nExtreme-move scores (>500 bps):\n'+extreme.to_string(index=False)+'\n\nDescriptive block bootstrap:\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nHistorical exposed evaluation; no fresh holdout. Training scores pool repeated timestamps across expanding windows. Small custom patch model, not a full PatchTST reproduction.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8')
    q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,seconds=time.monotonic()-begin,scores=scores.to_dict('records')))
    q.dump(OUT/'STATUS.json',dict(stage='complete'))
    print(report,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):run()
