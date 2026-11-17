"""Past-only direction-state features; separate from magnitude feature families."""
import numpy as np
import pandas as pd
TOKENS=['flow_pressure_percentile','cvd_path_efficiency_percentile','funding_change_percentile','basis_change_percentile','basis_persistence_percentile','oi_range_position_percentile','leverage_unwind_percentile']
def build(x,inventory):
    grid=pd.date_range(x.index.min(),x.index.max(),freq='30min')
    if not x.index.is_unique or not x.index.isin(grid).all():raise ValueError('Invalid half-hour direction grid')
    sources=[]
    for token in TOKENS:
        match=[c for c in inventory if token in c and '__1h__' in c]
        if len(match)!=1:raise ValueError('Expected one direction source for '+token)
        sources.append(match[0])
    data=x[sources].reindex(grid);v={};defs=[]
    def add(name,a,formula):
        v[name]=a.reindex(x.index);defs.append(dict(feature=name,formula=formula))
    for token,col in zip(TOKENS,sources):
        s=data[col];lag=s.shift(1)
        add('DIR48__'+token+'__lag30m',lag,col+' at t-30min; no fill across missing grid rows')
        for h in [24,48,120]:
            roll=s.rolling(f'{h}h',closed='left',min_periods=int(.8*2*h))
            mean=roll.mean();std=roll.std(ddof=0)
            for tag,a,f in [('mean',mean,'historical mean'),('deviation',lag-mean,'lag30m minus historical mean'),('z',((lag-mean)/std.clip(lower=1e-6)).clip(-5,5),'(lag30m-mean)/max(populationstd,1e-6),clip +/-5')]:
                add(f'DIR48__{token}__{tag}_{h}h',a,f'{f} of {col} in [t-{h}h,t);80%expected half-hour coverage')
            add(f'DIR48__{token}__change_{h}h',lag-lag.shift(2*h),f'{col}(t-30min)-{col}(t-{h}h-30min); exact grid positions')
    # Centered products are candidate state interactions,not guaranteed signed effects.
    for a,b in [(0,3),(0,6),(2,3)]:
        add(f'DIR48__interaction_{a}_{b}',(data[sources[a]].shift(1)-.5)*(data[sources[b]].shift(1)-.5),f'({sources[a]}(t-30min)-.5)*({sources[b]}(t-30min)-.5)')
    return pd.DataFrame(v,index=x.index).replace([np.inf,-np.inf],np.nan),pd.DataFrame(defs)
