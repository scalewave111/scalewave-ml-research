"""Predict with one saved horizon-specific tree model; no labels or refitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
import tree_magnitude_filter as m

def predict_frame(artifact,raw,start=None):
 if not isinstance(raw.index,pd.DatetimeIndex):raise ValueError('Features need a UTC DatetimeIndex')
 raw=raw.sort_index().copy()
 if raw.index.tz is None:raise ValueError('Timezone-aware timestamps required')
 raw.index=raw.index.tz_convert('UTC').as_unit('ns')
 if not raw.index.is_unique:raise ValueError('Duplicate feature timestamps')
 parents=artifact.get('engineering_parents',artifact['columns'])
 missing=set(parents)-set(raw)
 if missing:raise ValueError('Missing source features: '+str(sorted(missing)))
 raw=raw.replace([np.inf,-np.inf],np.nan).astype('float32')
 eng=m.engineered(raw,parents)
 frame=pd.concat([raw,eng],axis=1)
 if start is not None:
  stamp=pd.Timestamp(start)
  stamp=stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
  if frame.index.min()>stamp-pd.Timedelta(hours=24):raise ValueError('Supply at least24h of feature history before prediction start')
  frame=frame.loc[frame.index>=stamp]
 return pd.DataFrame({'horizon_hours':artifact['horizon_hours'],'magnitude_log_bps':m.predict(artifact,frame)},index=frame.index)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--start',required=True)
 args=p.parse_args();path=Path(args.features)
 if path.suffix.lower()=='.csv':
  raw=pd.read_csv(path,index_col=0,parse_dates=True)
 else:raw=m.old.read_frame(path)
 artifact=joblib.load(args.model)
 predict_frame(artifact,raw,args.start).to_csv(args.output)
if __name__=='__main__':main()
