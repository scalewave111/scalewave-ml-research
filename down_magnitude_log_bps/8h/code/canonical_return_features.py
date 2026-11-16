"""Verified canonical five-minute returns, causal volatility and multi-horizon labels."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import duckdb
EXPECTED_SHA256='ed122ee8f1582b669957f8b477e449134e30e020f83094f1e0dda28a0af64036'
DRIVE_URL='https://drive.google.com/file/d/13SEH8reObQsWLsECJbrONejd12pdInNU/view'
def load_returns(path):
 path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=EXPECTED_SHA256:raise ValueError('Canonical return source hash mismatch')
 con=duckdb.connect()
 q=con.execute("SELECT decision_time,signed_return_log_bps,label_status,horizon_minutes FROM read_parquet(?) WHERE canonical_symbol='BTCUSDT' AND EXTRACT(minute FROM decision_time)::INTEGER % 5=0 ORDER BY decision_time",[str(path)]).fetchdf()
 con.close()
 assert (q.horizon_minutes==5).all()
 q.decision_time=pd.to_datetime(q.decision_time,utc=True)
 idx=pd.DatetimeIndex(q.decision_time).as_unit('ns')+pd.Timedelta(minutes=5)
 assert idx.is_unique and (np.diff(idx.asi8)==pd.Timedelta(minutes=5).value).all()
 r=pd.Series(q.signed_return_log_bps.to_numpy(),index=idx)
 r.loc[q.label_status.to_numpy()!='VALID']=np.nan
 return r
def make_labels(r,index,hours):
 count=hours*12
 sums=r.rolling(f'{hours}h',closed='right',min_periods=count).sum()
 end=index+pd.Timedelta(hours=hours);signed=sums.reindex(end).to_numpy()
 return pd.DataFrame(dict(label_end=end,signed_return_log_bps=signed,magnitude_log_bps=abs(signed)),index=index).dropna()
def features(r,index):
 values={}
 for h in [4,24,168]:
  roll=r.rolling(f'{h}h',closed='left',min_periods=h*12)
  values[f'PRICE__realized_vol_{h}h']=np.sqrt(r.pow(2).rolling(f'{h}h',closed='left',min_periods=h*12).sum())
  values[f'PRICE__mean_absolute_return_{h}h']=r.abs().rolling(f'{h}h',closed='left',min_periods=h*12).mean()
  values[f'PRICE__absolute_net_move_{h}h']=roll.sum().abs()
 values['PRICE__vol_ratio_4h_24h']=(values['PRICE__realized_vol_4h']/np.sqrt(4))/(values['PRICE__realized_vol_24h']/np.sqrt(24)).clip(lower=1e-6)
 return pd.DataFrame(values).reindex(index)
def validate(r,original_y):
 lab=make_labels(r,original_y.index,48)
 assert lab.index.equals(original_y.index)
 error=float(np.max(abs(lab.signed_return_log_bps-original_y.signed_return_log_bps)))
 if error>1e-8:raise ValueError('New source does not reproduce original48h labels')
 cut=original_y.index[len(original_y)//2]
 pd.testing.assert_frame_equal(features(r,original_y.index[original_y.index<=cut]),features(r.loc[:cut],original_y.index[original_y.index<=cut]))
 changed=r.copy();changed.loc[changed.index>=cut]+=100000
 np.testing.assert_allclose(features(r,pd.DatetimeIndex([cut])),features(changed,pd.DatetimeIndex([cut])),rtol=0,atol=0)
 # Independent direct-window sums for sampled actual48h targets.
 for origin in original_y.index[::1000]:
  window=r[(r.index>origin)&(r.index<=origin+pd.Timedelta(hours=48))]
  assert len(window)==576 and window.notna().all()
  assert abs(float(window.sum())-original_y.loc[origin,'signed_return_log_bps'])<1e-8
 return dict(status='PASS',source_sha256=EXPECTED_SHA256,source_url=DRIVE_URL,rows_5min=len(r),missing_5min=int(r.isna().sum()),
  matched_original_labels=len(lab),max48h_error_log_bps=error,causal_prefix=True,future_return_perturbation=True,
  availability='Each5m forward return is indexed at its endpoint. Features use only endpoints strictly before the forecast origin.',
  interpretation='Price-based absolute volatility derived from canonical nonoverlapping5m log returns; not a reconstructed dollar price series.')
