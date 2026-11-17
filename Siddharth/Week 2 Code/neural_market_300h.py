"""Compact MAE MLP/GRU search; causal snapshot sequences and train-only preprocessing."""
import numpy as np
import pandas as pd
import torch
from torch import nn
import focused_market_trees_300h as tree

b=tree.b
PACKS=['flow_liquidation']
WEIGHTS=[.1,.25,.5,1.]
CONFIG={
 'mlp16':dict(kind='mlp',hidden=16,lr=.002,decay=.1),
 'mlp32_16':dict(kind='mlp',hidden=32,lr=.001,decay=.03),
 'gru8':dict(kind='gru',hidden=8,lr=.002,decay=.1),
 'gru16':dict(kind='gru',hidden=16,lr=.001,decay=.03)}
EPOCHS=24
CAP=4096
LAGS=[90,60,30,0]


def pack_columns(x,pack):return tree.pack_columns(x,pack)


def augment(x):
    cols=list(dict.fromkeys(c for pack in tree.PACKS for c in pack_columns(x,pack)))
    seq={f'SEQ{lag}__{c}':x[c].reindex(x.index-pd.Timedelta(hours=lag)).to_numpy() for lag in LAGS for c in cols}
    out=x.join(pd.DataFrame(seq,index=x.index));out.attrs=x.attrs.copy()
    return out


class Network(nn.Module):
    def __init__(self,n,cfg):
        super().__init__();self.kind=cfg['kind'];h=cfg['hidden']
        if self.kind=='gru':
            self.body=nn.GRU(n,h,batch_first=True);self.head=nn.Linear(h,1)
        else:
            layers=[nn.Linear(n,h),nn.ReLU()]
            if h==32:layers.extend([nn.Linear(h,16),nn.ReLU()]);h=16
            self.body=nn.Sequential(*layers);self.head=nn.Linear(h,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,z):
        z=self.body(z)[0][:,-1] if self.kind=='gru' else self.body(z)
        return .75*torch.tanh(self.head(z)).squeeze(-1)


def inputs(x,model):
    a=x[model['features']].to_numpy(float)
    z=np.clip(b.transform(a,model['prep']),-8,8).astype(np.float32)
    if model['config']['kind']=='gru':
        n=len(model['features']);z=np.concatenate([z[:,:n].reshape(len(z),4,-1),z[:,n:].reshape(len(z),4,-1)],axis=2)
    return torch.from_numpy(z)


def fit(x,y,spec):
    median=float(np.median(y))
    if spec['model']=='median':return dict(constant=median,features=[])
    cfg=CONFIG[spec['config']]
    cols=pack_columns(x,spec.get('pack','flow_liquidation'))
    if cfg['kind']=='gru':cols=[f'SEQ{lag}__{c}' for lag in LAGS for c in cols]
    # Keep the GRU sequence dimensions fixed; reject a pack if any input has insufficient history.
    quality=[dict(feature=c,coverage=float(np.isfinite(x[c]).mean()),eligible=bool(np.isfinite(x[c]).sum()>=1000 and np.isfinite(x[c]).mean()>=.8)) for c in cols]
    if not all(q['eligible'] for q in quality):return dict(constant=median,features=[],quality=quality)
    model=dict(features=cols,prep=b.transform_fit(x[cols].to_numpy(float)),config=cfg,median=median,unit=max(median,1.),quality=quality,loss=spec.get('loss','absolute_error'))
    rows=np.unique(np.linspace(0,len(y)-1,min(CAP,len(y)),dtype=int))
    z=inputs(x.iloc[rows],model);target=torch.tensor((np.asarray(y)[rows]-median)/model['unit'],dtype=torch.float32)
    torch.manual_seed(48);torch.use_deterministic_algorithms(True)
    net=Network(z.shape[-1],cfg)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    generator=torch.Generator().manual_seed(48)
    for _ in range(EPOCHS):
        for ids in torch.randperm(len(rows),generator=generator).split(512):
            optimizer.zero_grad(set_to_none=True)
            error=net(z[ids])-target[ids]
            loss=error.square().mean() if model['loss']=='squared_error' else error.abs().mean()
            b.check(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
    model['state']={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    model['estimator_training_rows']=len(rows)
    return model


def predict(model,x,weight=1.):
    if 'constant' in model:return np.full(len(x),model['constant'])
    z=inputs(x,model);net=Network(z.shape[-1],model['config'])
    net.load_state_dict({k:torch.from_numpy(v) for k,v in model['state'].items()});net.eval()
    with torch.no_grad():adjustment=net(z).numpy().astype(float)*model['unit']
    coverage=np.isfinite(x[model['features']].to_numpy(float)).mean(axis=1)>=.8
    return np.maximum(0,model['median']+weight*coverage*adjustment)

