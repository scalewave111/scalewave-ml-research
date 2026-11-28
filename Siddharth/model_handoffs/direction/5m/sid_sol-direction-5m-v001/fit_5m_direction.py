"""Seven chronological SOL five-minute fits across the guarded six-family universe."""
from pathlib import Path
import json
import sys
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
V1=OUT.parent/"SOL_5M_ROUTER_V01"
CACHE=OUT.parent/"btc15_broad_inputs_20261126"
STRIDE=ROOT/"Datasets/SOL_60M_ALLFEATURES_V01"
sys.path.insert(0,str(V1))
from sol5_experts import SHARED, ISSUES, M, labels, digest
sys.path.insert(0,str(ROOT/"Code"))
from eth15af_common import day_slices
FAMILIES=("F1","F2","F4","F4Z","F5","F5Z")
SEED=5281
torch.set_num_threads(3)


def save(name,value):
    (OUT/name).write_text(json.dumps(value,default=str,indent=2),encoding="utf8")


def universe():
    names=json.loads((CACHE/"schema.json").read_text())["columns"]
    allowed=set(json.loads((CACHE/"venue_policy.json").read_text())["allowed_fields"])
    stride=json.loads((STRIDE/"hist_s5_manifest.json").read_text())["feature_names"]
    lookup={name:i for i,name in enumerate(stride)}
    rows=[]
    for i,name in enumerate(names):
        asset,key=name.split("__",1)
        family=key.split("__")[0]
        if family in FAMILIES:
            rows.append(dict(name=name,family=family,asset=asset,physical=i,stride=lookup.get(name,-1),admitted=key in allowed))
    return pd.DataFrame(rows)


def raw_context(split):
    paths=sorted(SHARED.glob(split+"_????-??.parquet"))
    cols=[a+"_"+v for a in ("BTC","ETH","SOL") for v in
          ("valid_price","scale","r1","r5","flow","liq_long","liq_short","oi","vol_ratio","reported_pressure")]
    return pd.concat([pd.read_parquet(p,columns=cols) for p in paths],verify_integrity=True)


def engineer(d):
    """Rolling hedge and impact coefficients are lagged; no fit touches a future row."""
    logp={a:np.log(d[a+"_valid_price"].where(d[a+"_valid_price"]>0).astype(float)) for a in ("BTC","ETH","SOL")}
    s=logp["SOL"].diff()*10000
    scale=d.SOL_scale.clip(lower=.2)
    peer=(logp["ETH"].diff()+logp["BTC"].diff())*5000
    def slope(a,b,w):
        return (a.rolling(w,min_periods=w//2).cov(b)/b.rolling(w,min_periods=w//2).var().clip(lower=1e-8)).shift(1).clip(-4,4)
    beta=slope(s,peer,180)
    residual=s-beta*peer
    flow=d.SOL_flow
    dv=flow-(d.BTC_flow+d.ETH_flow)/2
    impact=slope(s,flow,180)
    err=s-impact*flow
    out={"scale":scale,"beta_peer":beta,"flow_impact":impact,"sol_return":s/scale,
         "peer_lead":peer.shift(1)/scale,"peer_gap":(10000*(logp["BTC"]-logp["ETH"]).diff())/scale,
         "vol_ratio":d.SOL_vol_ratio,"sol_flow":flow,"relative_flow":dv,"open_interest":d.SOL_oi,
         "forced_side":d.SOL_liq_short-d.SOL_liq_long,"pressure":d.SOL_reported_pressure}
    for w in (2,5,15,60):
        diff=residual.rolling(w).sum()/scale
        common=peer.rolling(w).sum()/scale
        path=s.rolling(w).sum()/s.abs().rolling(w).sum().clip(lower=.01)
        f=flow.rolling(w,min_periods=max(1,w//2)).mean()
        out.update({f"hedged_move_{w}":diff, f"peer_wave_{w}":common,f"path_efficiency_{w}":path,
                    f"residual_curvature_{w}":diff-diff.shift(w),f"unexplained_impact_{w}":err.rolling(w).sum()/scale,
                    f"peer_catchup_{w}":common*(1-path.abs()),f"pressure_without_progress_{w}":f*(1-path.abs()),
                    f"flow_mismatch_{w}":f-np.tanh(diff/np.sqrt(w)),
                    f"vol_adjusted_reversal_{w}":-diff*d.SOL_vol_ratio.clip(0,4)})
    for lag in (1,3,8):
        out[f"lagged_residual_{lag}"]=residual.shift(lag)/scale
        out[f"historical_autocorr_{lag}"]=residual.rolling(180,min_periods=90).corr(residual.shift(lag))
    return pd.DataFrame(out,index=d.index).replace([np.inf,-np.inf],np.nan).astype("float32")


def context(split):
    d=raw_context(split)
    if split=="test":
        warm=raw_context("validation").iloc[-1440:]
        full=pd.concat([warm,d],verify_integrity=True)
    else:
        full=d
    e=engineer(full)
    np.testing.assert_allclose(e.iloc[999],engineer(full.iloc[:1000]).iloc[-1],equal_nan=True,atol=1e-6)
    return e.loc[d.index],d.SOL_valid_price.gt(0)&d.SOL_scale.gt(0)


def screen(x,ret,names):
    # Similarity to direction and movement in disjoint historical slices, with minority-tail coverage.
    scores=[]
    for rows in np.array_split(np.arange(len(ret)),3):
        y=ret[rows]
        outcomes=np.column_stack((np.sign(y),np.clip(y,-5,5),np.minimum(abs(y),8)))
        outcomes=(outcomes-outcomes.mean(0))/outcomes.std(0).clip(.01)
        part=[]
        for j in range(0,x.shape[1],100):
            a=x[rows,j:j+100]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                low,mid,high=np.nanpercentile(a,[10,50,90],axis=0)
                z=np.nan_to_num(((a-mid)/np.where(high>low,high-low,np.nan)).clip(-5,5)).astype("f")
            z-=z.mean(0)
            part.append(z.T@outcomes/(len(rows)*z.std(0).clip(.02)[:,None]))
        scores.append(np.concatenate(part))
    measures=np.maximum(0,np.abs(np.mean(scores,axis=0))-.45*np.std(scores,axis=0))
    rank=pd.DataFrame(dict(name=names,direction=measures[:,:2].max(1),size=measures[:,2]))
    rank["family"]=[n.split("__")[1] for n in names]
    picks=[]
    for fam in FAMILIES:
        v=rank[rank.family.eq(fam)]
        picks+=v.nlargest(12,"direction").name.tolist()+v.nlargest(5,"size").name.tolist()
    picks+=rank.nlargest(32,"direction").name.tolist()
    picks=list(dict.fromkeys(picks))
    rank["selected"]=rank.name.isin(picks)
    return picks,rank


def standardized(a,m):
    return np.nan_to_num(((a-m["center"])/m["width"]).clip(-8,8)).astype("float32")


def net(n):
    return nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),
                         nn.Linear(96,48),nn.SiLU(),nn.Linear(48,4))


def fit(a,r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        q=np.nanpercentile(a,[10,50,90],axis=0)
    m=dict(center=np.nan_to_num(q[1]).astype("f"),width=np.where(q[2]>q[0],q[2]-q[0],1).astype("f"))
    z=standardized(a,m)
    kw=dict(n_estimators=125,num_leaves=17,min_child_samples=130,learning_rate=.045,colsample_bytree=.75,
            reg_lambda=16,verbosity=-1,n_jobs=3,deterministic=True,force_col_wise=True,random_state=SEED)
    m["tree"]=lgb.LGBMClassifier(**kw).fit(a,r>0)
    m["size"]=lgb.LGBMRegressor(**kw).fit(a,abs(r).clip(0,14))
    m["pay"]=lgb.LGBMRegressor(**kw).fit(a,r.clip(-10,10))
    # Four return bins permit different typical magnitudes on up and down moves.
    cat=np.digitize(r,[-1,0,1])
    m["bin_means"]=np.array([r[cat==i].clip(-12,12).mean() for i in range(4)])
    torch.manual_seed(SEED)
    model=net(a.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.00085,weight_decay=.015)
    inp=torch.from_numpy(z)
    target=torch.from_numpy(cat).long()
    rng=np.random.default_rng(SEED)
    for epoch in range(11):
        model.train()
        for ids in np.array_split(rng.permutation(len(r)),max(1,len(r)//768)):
            logits=model(inp[ids])
            loss=nn.functional.cross_entropy(logits,target[ids])
            opt.zero_grad();loss.backward();opt.step()
    m["net"]=model.state_dict()
    return m


def predict(a,scale,m,model):
    prob=m["tree"].booster_.predict(a,num_threads=3)
    size=m["size"].booster_.predict(a,num_threads=3).clip(.01)*scale
    pay=m["pay"].booster_.predict(a,num_threads=3)*scale
    z=standardized(a,m)
    with torch.no_grad():
        bins=torch.softmax(model(torch.from_numpy(z)),dim=1).numpy()
    neural_p=bins[:,2:].sum(1)
    neural_size=bins@np.abs(m["bin_means"])*scale
    neural_pay=bins@m["bin_means"]*scale
    return np.column_stack([prob,size,pay,neural_p,neural_size,neural_pay])


def forecasts(split,e,valid,m):
    model=net(len(m["names"])+e.shape[1])
    model.load_state_dict(m["net"]);model.eval()
    parts=[]
    for day,sl,rows in day_slices(e.index):
        file=CACHE/split/(day+".npy")
        x=np.load(file,mmap_mode="r")[np.ix_(rows,m["physical"])]
        frame=e.iloc[sl]
        a=np.column_stack([x,frame.to_numpy()])
        parts.append(predict(a,frame.scale.to_numpy(),m,model))
    f=pd.DataFrame(np.vstack(parts),index=e.index,columns=["tp","ts","ty","np","ns","ny"])
    f["valid"]=valid
    return f


def train():
    reg=universe();reg.to_csv(OUT/"universe.csv",index=False)
    allowed=reg[reg.admitted].set_index("name")
    assert allowed.stride.ge(0).all()
    names=allowed.index.tolist()
    e,valid=context("historical")
    idx=pd.DatetimeIndex(pd.read_feather(STRIDE/"hist_s5_meta.feather",columns=["timestamp_utc"]).timestamp_utc).as_unit("ns")
    s=np.load(STRIDE/"hist_s5_X.npy",mmap_mode="r")
    sample_ids=np.arange(0,len(idx),8)
    sample=s[np.ix_(sample_ids,allowed.stride)]
    y=labels("historical").reindex(idx)
    r=y.actual_signed_return_bps.to_numpy()/e.scale.reindex(idx).to_numpy()
    print("Historical feature screen",allowed.family.value_counts().to_dict(),flush=True)
    save("engineered.json",list(e))
    ranks=[];parts=[];logs=[]
    for q in ISSUES:
        ok=(idx>=q-pd.DateOffset(years=1))&y.label_available_at.lt(q).to_numpy()&np.isfinite(r)&valid.reindex(idx).to_numpy()
        pos=np.flatnonzero(ok)
        prefix=sample_ids[ok[sample_ids]]
        picks,rank=screen(sample[ok[sample_ids]],r[prefix],names)
        ranks.append(rank.assign(issue=str(q.date())))
        cols=allowed.loc[picks]
        a=np.column_stack([s[np.ix_(pos,cols.stride)],e.reindex(idx[pos]).to_numpy()])
        m=fit(a,r[pos]);m.update(names=picks,physical=cols.physical.tolist(),engineered=list(e))
        path=OUT/"models"/f"{q.date()}.joblib"
        joblib.dump(m,path,compress=2)
        log=dict(issue=str(q),rows=len(pos),screened=len(names),selected=len(picks),latest_origin=idx[pos[-1]],
                 latest_maturity=y.label_available_at.iloc[pos].max(),sha256=digest(path))
        assert log["latest_maturity"]<q
        logs.append(log)
        print("Fitted",q.date(),len(pos),"rows",len(picks),"selected fields",flush=True)
        take=(e.index>=q)&(e.index<q+pd.DateOffset(months=3))
        parts.append(forecasts("historical",e[take],valid[take],m))
        print("Scored",q.date(),flush=True)
    save("fits.json",logs)
    pd.concat(ranks).to_csv(OUT/"screening.csv",index=False)
    pd.concat(parts,verify_integrity=True).to_parquet(OUT/"historical.parquet")


if __name__=="__main__":
    with threadpool_limits(limits=2,user_api="blas"):
        train()
