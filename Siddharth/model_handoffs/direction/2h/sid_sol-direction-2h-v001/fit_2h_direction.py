"""Quarterly GRU sign, magnitude and conditional payoff forecasts for two-hour SOL."""
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from two_hour_models import OUT,MODEL,START,END,ISSUES,M,H,MECH

COLS=["ret1","vol_ratio","flow","relative","trend","imbalance","liq_intensity","oi_change","lead_BTC","lead_ETH","peer_flow_BTC","peer_flow_ETH"]
LAGS=np.array([120,60,30,15,5,0])


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(len(COLS),18,batch_first=True)
        self.output=nn.Linear(18,3)

    def forward(self,x):
        _,h=self.gru(x)
        return self.output(h[-1])


def raw(splits):
    f=pd.concat([pd.read_parquet(MECH/f"SOL_{s}.parquet",columns=COLS+["vol","price_valid"]) for s in splits],verify_integrity=True)
    assert f.index.to_series().diff().dropna().eq(M).all()
    return f,f[COLS].to_numpy(dtype="float32"),(f.vol*np.sqrt(H)).clip(3,650).where(f.price_valid.eq(1))


def tensor(arr,ids,model):
    seq=arr[ids[:,None]-LAGS]
    return torch.from_numpy(np.nan_to_num(np.clip((seq-model["center"])/model["width"],-5,5),nan=0.).astype("float32"))


def predict(arr,ids,model,scale):
    net=Temporal()
    net.load_state_dict(model["state"])
    net.eval()
    parts=[]
    with torch.no_grad():
        for batch in np.array_split(ids,max(1,(len(ids)+8191)//8192)):
            out=net(tensor(arr,batch,model))
            p=torch.sigmoid(out[:,0])
            mag=nn.functional.softplus(out[:,1])
            signed=((2*p-1)*mag).numpy()*scale.iloc[batch].to_numpy()
            payoff=out[:,2].numpy()*scale.iloc[batch].to_numpy()
            parts.append(pd.DataFrame(dict(gru=signed,gru_payoff=payoff,gru_p=p.numpy()),index=scale.index[batch]))
    return pd.concat(parts,verify_integrity=True)


def train():
    torch.set_num_threads(1)
    f,arr,scale=raw(["historical"])
    y=pd.read_parquet(OUT/"historical_labels.parquet").reindex(f.index)
    rr=(y.r/scale).clip(-6,6)
    parts,records=[],[]
    for q in ISSUES:
        mask=(f.index<q)&(f.index>=q-pd.DateOffset(years=1))&y.avail.lt(q)&rr.notna()
        ids=np.flatnonzero(mask&((f.index.as_unit("ns").asi8//M.value)%23==0))
        ids=ids[ids>=LAGS.max()]
        lo,center,hi=np.nanpercentile(arr[ids],[10,50,90],axis=0)
        m=dict(center=np.nan_to_num(center).astype("float32"),width=np.where(hi>lo,hi-lo,1).astype("float32"))
        torch.manual_seed(22677)
        net=Temporal()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.012)
        x=tensor(arr,ids,m)
        target=torch.tensor(rr.iloc[ids].to_numpy(dtype="float32"))
        rng=np.random.default_rng(22677)
        for epoch in range(8):
            for b in np.array_split(rng.permutation(len(ids)),max(1,len(ids)//512)):
                out=net(x[b])
                r=target[b]
                weights=(.65+.35*r.abs().clamp(max=3))
                loss=(nn.functional.binary_cross_entropy_with_logits(out[:,0],(r>0).float(),reduction="none")*weights).mean()
                loss+=.35*nn.functional.smooth_l1_loss(nn.functional.softplus(out[:,1]),r.abs())+.4*nn.functional.smooth_l1_loss(out[:,2],r)
                opt.zero_grad()
                loss.backward()
                opt.step()
        m["state"]=net.state_dict()
        joblib.dump(m,MODEL/f"gru_{q.date()}.joblib",compress=2)
        ix=np.flatnonzero((f.index>=q)&(f.index<q+pd.DateOffset(months=3)))
        parts.append(predict(arr,ix,m,scale))
        records.append(dict(issue=q,rows=len(ids),last_origin=f.index[ids[-1]],last_available=y.avail.iloc[ids].max()))
        print("Two-hour GRU",q.date(),flush=True)
    pd.DataFrame(records).to_csv(OUT/"gru_fits.csv",index=False)
    pd.concat(parts,verify_integrity=True).to_parquet(OUT/"historical_gru.parquet")


def external_gru(ix):
    torch.set_num_threads(1)
    f,arr,scale=raw(["validation","test"])
    ids=f.index.get_indexer(ix)
    valid=ids>=LAGS.max()
    return predict(arr,ids[valid],joblib.load(MODEL/f"gru_{ISSUES[-1].date()}.joblib"),scale).reindex(ix)


if __name__=="__main__":train()
