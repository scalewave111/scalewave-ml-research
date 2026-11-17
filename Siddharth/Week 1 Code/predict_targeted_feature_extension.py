"""Inference for the38-column targeted48h hybrid; no labels or fitting."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
import targeted_feature_extension as m
def forecast(selection,raw,keys):
 issue=pd.Timestamp(selection['issue']);keys=pd.DatetimeIndex(keys)
 if raw.index.tz is None or not raw.index.is_unique:raise ValueError('Unique timezone-aware feature timestamps required')
 if len(keys)==0 or (keys<issue).any():raise ValueError('Forecast rows cannot precede model issuance')
 raw=raw.replace([np.inf,-np.inf],np.nan).astype('float32')
 needed=list(dict.fromkeys(m.PAIR+selection['engineering_parents']+[c for c in selection['features'] if c.startswith('F4Z__')]))
 missing=set(needed)-set(raw)
 if missing:raise ValueError('Missing input columns: '+str(sorted(missing)))
 frame=pd.concat([raw,m.engineering(raw,selection['engineering_parents'])],axis=1)
 pair=joblib.load(selection['pair_artifact']);forest=joblib.load(selection['artifact'])
 for a in [pair,forest]:
  if pd.Timestamp(a['issuance'])!=issue:raise ValueError('Artifact issuance mismatch')
 prediction=(1-selection['weight'])*m.predict(pair,frame.loc[keys])+selection['weight']*m.predict(forest,frame.loc[keys])
 return pd.DataFrame(dict(predicted_magnitude_log_bps=prediction,forecast_end=keys+pd.Timedelta(hours=48),model_issuance=str(issue),rf_weight=selection['weight']),index=keys)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--start',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 raw=pd.read_csv(args.features,index_col='decision_time',parse_dates=True);raw.index=pd.to_datetime(raw.index,utc=True);raw=raw.sort_index()
 start=pd.Timestamp(args.start);start=start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
 if raw.index.min()>start-pd.Timedelta(hours=24):raise ValueError('Provide at least24h raw feature history for causal transformations')
 result=forecast(json.loads(args.selection.read_text()),raw,raw.index[raw.index>=start])
 result.to_csv(args.output,index_label='decision_time');print('Rows',len(result))
if __name__=='__main__':main()
