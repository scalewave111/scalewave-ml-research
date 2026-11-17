"""Predict from saved three-feature walk-forward artifacts; no fitting."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
import three_feature_tree_tuning as t
def forecast(selection,frame):
 if frame.index.tz is None or not frame.index.is_unique:raise ValueError('Unique timezone-aware timestamps required')
 issue=pd.Timestamp(selection['issue'])
 if (frame.index<issue).any():raise ValueError('Rows precede model issuance')
 missing=set(t.COLS)-set(frame)
 if missing:raise ValueError('Missing features: '+str(sorted(missing)))
 x=frame[t.COLS].replace([np.inf,-np.inf],np.nan).astype('float32')
 pair=joblib.load(selection['pair_artifact']);standalone=joblib.load(selection['standalone_artifact']);blend=joblib.load(selection['hybrid_artifact'])
 for a in [pair,standalone,blend]:
  if pd.Timestamp(a['issuance'])!=issue:raise ValueError('Artifact issuance mismatch')
 w=selection['hybrid']['weight']
 return pd.DataFrame(dict(standalone_magnitude_log_bps=t.m.predict(standalone,x),hybrid_magnitude_log_bps=(1-w)*t.m.predict(pair,x)+w*t.m.predict(blend,x),forecast_end=x.index+pd.Timedelta(hours=48),model_issuance=str(issue)),index=x.index)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 x=pd.read_csv(args.features,index_col='decision_time',parse_dates=True);x.index=pd.to_datetime(x.index,utc=True);x=x.sort_index()
 forecast(json.loads(args.selection.read_text()),x).to_csv(args.output,index_label='decision_time')
if __name__=='__main__':main()
