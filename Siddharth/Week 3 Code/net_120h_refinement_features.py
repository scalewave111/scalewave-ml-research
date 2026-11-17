"""Past-only endpoint distribution and multiscale persistence features."""
import numpy as np
import pandas as pd

def build(r,origins):
    hour=r.resample('1h',closed='left',label='right').sum(min_count=12)
    count=r.resample('1h',closed='left',label='right').count()
    hour=hour.where(count.eq(12))
    magnitude=hour.rolling(120,min_periods=120).sum().abs()
    values={};dictionary=[]
    def add(name,series,formula,hourly=True):
        key='REF120__'+name
        values[key]=series.reindex(origins,method='ffill',tolerance=pd.Timedelta(minutes=59,seconds=59)) if hourly else series.reindex(origins)
        dictionary.append(dict(feature=key,family='endpoint_refinement',formula=formula))
    for h in [120,720]:
        for quantile in [.5,.75,.9]:
            add(f'endpoint_q{int(quantile*100)}_{h}h',magnitude.rolling(h,min_periods=h).quantile(quantile),f'Q{quantile} of last {h} completed hourly observations of abs(sum past120 completed hourly returns); overlapping historical windows, no future labels')
    for h in [60,120,240,720]:
        q5=r.pow(2).rolling(f'{h}h',closed='left',min_periods=12*h).sum()
        q1=hour.pow(2).rolling(h,min_periods=h).sum()
        denominator=q5.reindex(hour.index)
        add(f'hour_to_5m_energy_{h}h',q1/denominator.clip(lower=1e-8),f'sum past{h} completed hourly return squared / max(sum prior{h}h5min squared returns,1e-8); measures aggregation/persistence, not future variance')
    q={h:r.pow(2).rolling(f'{h}h',closed='left',min_periods=12*h).sum()/h for h in [60,240,720]}
    for h in [60,240]:
        add(f'vol_regime_{h}_720',np.sqrt(q[h]/q[720].clip(lower=1e-8)),f'sqrt((Q{h}/{h})/(Q720/720)); 5min endpoints strictly before origin',False)
    return pd.DataFrame(values,index=origins).replace([np.inf,-np.inf],np.nan),pd.DataFrame(dictionary)
