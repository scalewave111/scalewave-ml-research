"""Saved controlled-hybrid inference from causal feature history; never fits."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
import controlled_feature_extension as m
def forecast(selection,raw,returns,keys):
 issue=pd.Timestamp(selection['issue']);keys=pd.DatetimeIndex(keys)
 if raw.index.tz is None or not raw.index.is_unique:raise ValueError('Unique UTC-aware feature history required')
 if len(keys)==0 or (keys<issue).any():raise ValueError('Forecast timestamps must be at or after artifact issuance')
 needed=list(dict.fromkeys(m.PAIR+selection['engineering_parents']+[c for c in selection['features'] if c.startswith('F4Z__')]))
 missing=[c for c in needed if c.startswith('F4Z__') and c not in raw]
 if missing:raise ValueError('Missing raw columns: '+str(missing))
 x=pd.concat([raw.replace([np.inf,-np.inf],np.nan),m.price_features(returns,raw.index)],axis=1)
 x=pd.concat([x,m.engineering(x,selection['engineering_parents'])],axis=1)
 pair=joblib.load(selection['pair_artifact']);forest=joblib.load(selection['artifact'])
 for a in [pair,forest]:
  if pd.Timestamp(a['issuance'])!=issue:raise ValueError('Artifact issuance mismatch')
 result=(1-selection['weight'])*m.predict(pair,x.loc[keys])+selection['weight']*m.predict(forest,x.loc[keys])
 return pd.DataFrame(dict(predicted_magnitude_log_bps=result,forecast_end=keys+pd.Timedelta(hours=48),model_issuance=str(issue),rf_weight=selection['weight']),index=keys)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--returns',type=Path,required=True);ap.add_argument('--start',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 raw=pd.read_csv(args.features,index_col='decision_time',parse_dates=True);raw.index=pd.to_datetime(raw.index,utc=True);raw=raw.sort_index()
 start=pd.Timestamp(args.start);start=start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
 if raw.index.min()>start-pd.Timedelta(hours=24):raise ValueError('At least24h raw feature history required for engineering')
 # Inference allows newly supplied past-return data; unlike research loader, it does not pin a historical-file hash.
 r=pd.read_csv(args.returns,index_col='return_end',parse_dates=True)['signed_return_log_bps'];r.index=pd.to_datetime(r.index,utc=True);r=r.sort_index()
 if not r.index.is_unique or (r.index.to_series().diff().dropna()!=pd.Timedelta(minutes=5)).any():raise ValueError('Continuous5m return endpoint grid required; retain missing observations as NaN')
 if r.index.min()>start-pd.Timedelta(hours=168):raise ValueError('At least168h past5m returns required')
 out=forecast(json.loads(args.selection.read_text()),raw,r,raw.index[raw.index>=start]);out.to_csv(args.output,index_label='decision_time')
 print('Forecasts written',len(out))
if __name__=='__main__':main()
