"""Apply a saved hybrid weight vector to aligned48h component forecasts."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
def predict(frame,weights):
 if abs(sum(weights.values())-1)>1e-8 or min(weights.values())<0:raise ValueError('Invalid convex weights')
 missing=set(weights)-set(frame)
 if missing:raise ValueError('Missing component predictions: '+str(sorted(missing)))
 if not np.isfinite(frame[list(weights)].to_numpy()).all():raise ValueError('Required forecasts contain missing or invalid values')
 return sum(float(w)*frame[name] for name,w in weights.items())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selections',required=True);ap.add_argument('--issue',required=True);ap.add_argument('--forecasts',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 choices=json.loads(Path(args.selections).read_text());date=pd.Timestamp(args.issue)
 date=date.tz_localize('UTC') if date.tz is None else date.tz_convert('UTC')
 chosen=next(a for a in choices if pd.Timestamp(a['issue'])==date)
 frame=pd.read_csv(args.forecasts,index_col=0,parse_dates=True)
 if frame.index.tz is None or not frame.index.is_unique:raise ValueError('Unique timezone-aware forecast timestamps required')
 if (frame.index<date).any() or (frame.index>=date+pd.DateOffset(months=1)).any():raise ValueError('Input rows must belong to the saved weight-issuance month')
 pd.DataFrame({'magnitude_log_bps':predict(frame,chosen['weights'])},index=frame.index).to_csv(args.output)
if __name__=='__main__':main()
