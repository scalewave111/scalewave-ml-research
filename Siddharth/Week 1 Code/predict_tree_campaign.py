"""Inference for a saved monthly tree ensemble. No target labels or refitting."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import joblib
import tree_improvement_campaign as m
def predict_frame(folder,raw,end_returns,start=None):
 folder=Path(folder);decision=json.loads((folder/'DONE.json').read_text())
 arts={name:joblib.load(folder/(name+'.joblib')) for name in decision['weights']}
 parents=next(iter(arts.values()))['engineering_parents']
 if not isinstance(raw.index,pd.DatetimeIndex) or raw.index.tz is None:raise ValueError('UTC-aware feature timestamps required')
 if not raw.index.is_unique or not end_returns.index.is_unique:raise ValueError('Duplicate timestamps')
 if end_returns.index.tz is None:raise ValueError('UTC-aware return-end timestamps required')
 if len(end_returns)>1 and not (np.diff(end_returns.sort_index().index.asi8)==pd.Timedelta(minutes=5).value).all():raise ValueError('Supply regular, nonoverlapping5minute returns indexed at their end; gaps must be explicit NaNs')
 missing=set(parents)-set(raw)
 if missing:raise ValueError('Missing source columns: '+str(sorted(missing)))
 raw=raw.sort_index().astype('float32');raw.index=raw.index.tz_convert('UTC').as_unit('ns')
 end_returns=end_returns.sort_index();end_returns.index=end_returns.index.tz_convert('UTC').as_unit('ns')
 eng=m.engineering(raw,parents);states=m.state_frame(end_returns,raw.index)
 frame=pd.concat([raw,eng,states],axis=1).replace([np.inf,-np.inf],np.nan).astype('float32')
 if start is not None:
  stamp=pd.Timestamp(start);stamp=stamp.tz_localize('UTC') if stamp.tz is None else stamp.tz_convert('UTC')
  if raw.index.min()>stamp-pd.Timedelta(hours=24):raise ValueError('Provide24h preceding feature history')
  if end_returns.index.min()>stamp-pd.Timedelta(hours=168):raise ValueError('Provide168h preceding return-end history')
  frame=frame.loc[frame.index>=stamp]
 p=sum(decision['weights'][name]*m.predict(art,frame) for name,art in arts.items())
 return pd.DataFrame({'horizon_hours':decision['horizon_hours'],'magnitude_log_bps':p},index=frame.index)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-folder',required=True);p.add_argument('--features',required=True);p.add_argument('--return-ends-csv');p.add_argument('--start',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 path=Path(args.features)
 raw=pd.read_csv(path,index_col=0,parse_dates=True) if path.suffix.lower()=='.csv' else m.old.old.read_frame(path)
 if args.return_ends_csv:
  q=pd.read_csv(args.return_ends_csv);q['return_end']=pd.to_datetime(q.return_end,utc=True)
  r=q.set_index('return_end')['log_return_bps']
 else:r=m.crf.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet')
 predict_frame(args.model_folder,raw,r,args.start).to_csv(args.output)
if __name__=='__main__':main()
