"""Small deterministic MLPs, original-unit MAE, causal volatility scaling."""
from itertools import product
import numpy as np
import pandas as pd
import torch
from torch import nn
import discover_excursion_features_120h as b

SCALE_COLUMN='LONG__vol_per_sqrt_hour__120h'
PACKS={
 'volatility':['vol_per_sqrt_hour__120h','vol_per_sqrt_hour__720h','vol_ratio_120__240h','up_variance_share__120h'],
 'trend':['normalized_momentum__120h','trend_efficiency__168h','drift_change_120__240h','net__720h'],
 'range':['range_position__120h','drawdown_vol_units__336h','range__240h','range__1440h'],
 'mixed':['vol_per_sqrt_hour__720h','normalized_momentum__120h','range_position__240h','up_variance_share__1440h'],
}
PACKS={k:['LONG__'+c for c in v] for k,v in PACKS.items()}
CONFIGS={'small':dict(hidden=[8],epochs=40,decay=.1),'two_layer':dict(hidden=[16,8],epochs=60,decay=.1)}
SEEDS=[48,148]
WEIGHTS=[.25,.5,.75,1.]


def scales(x,mode,params=None):
    raw=x[SCALE_COLUMN].to_numpy(float)*np.sqrt(120)
    if params is None:
        finite=raw[np.isfinite(raw)&(raw>0)]
        fallback=float(np.median(finite)) if len(finite) else 1.
        floor=max(float(np.quantile(finite,.1)) if len(finite) else 1.,1.)
        params=dict(fallback=fallback,floor=floor)
    out=np.maximum(np.where(np.isfinite(raw),raw,params['fallback']),params['floor'])
    return (out if mode=='volatility' else np.ones(len(x))),params


def forward(weights,z):
    for i,(w,bias) in enumerate(weights):
        z=torch.nn.functional.linear(z,torch.from_numpy(w),torch.from_numpy(bias))
        if i<len(weights)-1:z=torch.relu(z)
    return torch.nn.functional.softplus(z)


def fit(x,y,spec):
    if spec.get('model')=='median':return dict(constant=float(np.median(y)))
    columns=spec['features']; a=x[columns].to_numpy(float)
    prep=b.transform_fit(a)
    inputs=torch.from_numpy(np.clip(b.transform(a,prep),-8,8).astype(np.float32))
    factor,params=scales(x,spec['mode'])
    unit=max(float(np.median(y)),1.)
    normalized=y/factor
    output_unit=max(float(np.median(normalized)),1e-3)
    target=torch.from_numpy((y/unit).astype(np.float32)).reshape(-1,1)
    multiplier=torch.from_numpy((factor*output_unit/unit).astype(np.float32)).reshape(-1,1)
    cfg=CONFIGS[spec['config']]; ensembles=[]
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        dims=[inputs.shape[1]]+cfg['hidden']+[1]; layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2:layers.append(nn.ReLU())
        layers.append(nn.Softplus());model=nn.Sequential(*layers)
        with torch.no_grad():
            model[-2].weight.zero_();model[-2].bias.fill_(float(np.log(np.expm1(1.))))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=cfg['decay'])
        generator=torch.Generator().manual_seed(seed)
        for _ in range(cfg['epochs']):
            for rows in torch.randperm(len(y),generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                # Original-unit MAE: normalized errors are weighted by causal scale.
                loss=(model(inputs[rows])*multiplier[rows]-target[rows]).abs().mean()
                b.check(bool(torch.isfinite(loss)),'Nonfinite MLP loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        weights=[(m.weight.detach().numpy().copy(),m.bias.detach().numpy().copy()) for m in model if isinstance(m,nn.Linear)]
        with torch.no_grad():
            np.testing.assert_array_equal(model(inputs[:64]).numpy(),forward(weights,inputs[:64]).numpy())
        ensembles.append(weights)
    return dict(prep=prep,spec=spec,scale_params=params,output_unit=output_unit,weights=ensembles,
                median=float(np.median(y)),unit=unit)


def predict(model,x):
    if 'constant' in model:return np.full(len(x),model['constant'])
    spec=model['spec']
    z=torch.from_numpy(np.clip(b.transform(x[spec['features']].to_numpy(float),model['prep']),-8,8).astype(np.float32))
    factor,_=scales(x,spec['mode'],model['scale_params'])
    with torch.no_grad():
        predictions=[forward(w,z).numpy().ravel().astype(float)*factor*model['output_unit'] for w in model['weights']]
    raw=np.mean(predictions,axis=0)
    weight=spec.get('weight',1.)
    result=weight*raw+(1-weight)*model['median']
    b.check(np.isfinite(result).all() and (result>=0).all(),'Invalid MLP output')
    return result


def gate(errors,baseline,weights):
    return (np.average(errors,weights=weights)<np.average(baseline,weights=weights)*.999
            and int((errors<baseline).sum())>=2 and np.all(errors<=baseline*1.02))


def search(x,y,folds,args,folder):
    baseline=np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr,va in folds])
    counts=[len(va) for _,va in folds]; quality=[];admitted={}
    for pack,cols in PACKS.items():
        valid=all(x.iloc[tr][c].notna().mean()>=.8 and x.iloc[tr][c].notna().sum()>=1000
                  and x.iloc[tr][c].nunique()>1 for tr,_ in folds for c in cols)
        quality.append(dict(pack=pack,eligible=valid,features=cols))
        if valid:admitted[pack]=cols
    trials=[];best=None
    for pack,config,mode in product(admitted,CONFIGS,['raw','volatility']):
        spec=dict(pack=pack,config=config,mode=mode,features=admitted[pack],weight=1.)
        predictions=[]
        for tr,va in folds:
            bundle=fit(x.iloc[tr],y[tr],spec)
            predictions.append(predict(bundle,x.iloc[va]))
        for weight in WEIGHTS:
            errors=np.array([np.abs(y[va]-(weight*p+(1-weight)*np.median(y[tr]))).mean() for (tr,va),p in zip(folds,predictions)])
            eligible=gate(errors,baseline,counts)
            row=dict(**{**spec,'weight':weight},validation_mae=float(np.average(errors,weights=counts)),
                window_mae=errors.tolist(),eligible=bool(eligible))
            trials.append(row)
            if eligible and (best is None or row['validation_mae']<best['validation_mae']):best=row
        print(f'  MLP {pack}/{config}/{mode} complete',flush=True)
    pd.DataFrame(quality).to_csv(folder/'quality.csv',index=False)
    b.dump(folder/'validation_trials.json',trials)
    if best is None:best=dict(model='median',features=[],validation_mae=float(np.average(baseline,weights=counts)))
    else:best=dict(model='mlp',**best)
    best['baseline_window_mae']=baseline.tolist()
    b.dump(folder/'selection.json',best)
    return best
