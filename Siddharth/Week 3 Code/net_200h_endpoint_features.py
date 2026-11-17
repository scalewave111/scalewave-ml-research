"""Past-only endpoint magnitude hypotheses; no labels or fitted parameters."""
import numpy as np
import pandas as pd

HOURS=[100,200,400,720]
MEASURES=['rv5m','zero_drift_abs_proxy','shrunk_drift_abs_proxy','drift_to_noise']

def endpoint_columns():
    return [f'END200__{name}_{h}h' for h in HOURS for name in MEASURES]

def endpoint_features(r,origins,price):
    from scipy.special import erf
    frame=price.copy();definitions=[]
    def add(key,value,formula):
        frame[key]=value.reindex(origins)
        definitions.append(dict(feature=key,family='endpoint200',formula=formula))
    for h in HOURS:
        roll=r.rolling(f'{h}h',closed='left',min_periods=12*h)
        n=roll.sum();q=r.pow(2).rolling(f'{h}h',closed='left',min_periods=12*h).sum()
        # Long-horizon drift is noisy: fixed 25% shrink, chosen before validation.
        mu=.25*n*(200/h)
        sigma=np.sqrt((q-n*n/(12*h)).clip(lower=0)*(200/h)).clip(lower=1e-8)
        z=mu/sigma
        proxy=sigma*np.sqrt(2/np.pi)*np.exp(-.5*z*z)+mu*erf(z/np.sqrt(2))
        for name,value,formula in [
            ('rv5m',np.sqrt(q),'sqrt(Q)'),
            ('zero_drift_abs_proxy',sigma*np.sqrt(2/np.pi),'sigma*sqrt(2/pi)'),
            ('shrunk_drift_abs_proxy',proxy,'mu=.25*N*200/h; EabsGaussian=sigma*sqrt(2/pi)*exp(-mu^2/(2*sigma^2))+mu*erf(mu/(sqrt(2)*sigma))'),
            ('drift_to_noise',z,'(.25*N*200/h)/sigma')]:
            add(f'END200__{name}_{h}h',value,f'{formula}; N=sum r,Q=sum r^2 over [t-{h}h,t); require {12*h} finite 5min returns; sigma=sqrt(max(Q-N^2/(12*h),0)*200/h),floor1e-8; historical hypothesis, not future truth')
    # Completed hours contain only 5m endpoints strictly before their timestamp.
    hour=r.resample('1h',closed='left',label='right').sum(min_count=12)
    count=r.resample('1h',closed='left',label='right').count()
    hour=hour.where(count.eq(12))
    magnitude=hour.rolling(200,min_periods=200).sum().abs()
    for h in [168,720]:
        rolling=magnitude.rolling(h,min_periods=h)
        mean=rolling.mean();rms=np.sqrt(magnitude.pow(2).rolling(h,min_periods=h).mean())
        for name,value,formula in [('past_endpoint_mean',mean,'mean(abs(sum last200completed hourly returns))'),('past_endpoint_rms',rms,'sqrt(mean(abs(sum last200completed hourly returns)^2))'),('past_endpoint_cv',rolling.std(ddof=0)/mean.clip(lower=1e-8),'std(past200h endpoint magnitude)/max(mean,1e-8)')]:
            aligned=value.reindex(origins,method='ffill',tolerance=pd.Timedelta(minutes=59,seconds=59))
            add(f'END200__{name}_{h}h',aligned,f'{formula}; aggregate last {h} completed hourly endpoints; overlapping historical blocks; no future outcomes')
    return frame.replace([np.inf,-np.inf],np.nan),pd.DataFrame(definitions)
