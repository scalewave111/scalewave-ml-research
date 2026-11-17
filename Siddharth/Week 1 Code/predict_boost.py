"""Replay the latest selected walk-forward model on timestamped feature CSVs."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nested_boost as nb
import pandas as pd
import numpy as np
import joblib

def forecast(run,features_csv,output):
 contract=json.loads((run/'CONTRACT.json').read_text())
 selection_path=sorted((run/'outer').glob('*_selection.json'))[-1]
 selection=json.loads(selection_path.read_text());issuance=pd.Timestamp(selection['outer'])
 frame=pd.read_csv(features_csv,index_col='decision_time')
 frame.index=pd.to_datetime(frame.index,utc=True).as_unit('ns')
 frame=frame.sort_index()
 if not frame.index.is_unique:raise ValueError('Duplicate feature timestamps')
 mapping=json.loads((nb.ROOT/'Report/ic_rf_sources/magnitude_feature_mapping.json').read_text())
 report=[v['column'] for v in mapping if v['mapped']]
 # Engineered changes require at least 24h of historical features, even for a single forecast.
 if frame.index.max()-frame.index.min()<pd.Timedelta(hours=24):
  raise ValueError('Supply at least 24h of causal feature history for delta features.')
 x=nb.features(frame,report);keys=x.index[x.index>=issuance]
 if len(keys)==0:raise ValueError('No forecast timestamps at or after the saved model issuance')
 name=selection['selected']
 def predict_model(modelname):
  artifact=joblib.load(run/'models'/f'{issuance:%Y-%m}__{modelname}.joblib')
  return nb.predict(artifact,x.loc[keys])
 if '_blend' in name:
  modelname,w=name.rsplit('_blend',1);w=float(w)
  values=(1-w)*predict_model('rolling_pair')+w*predict_model(modelname)
 else:values=predict_model(name)
 result=pd.DataFrame(dict(predicted_magnitude_log_bps=values,horizon_hours=contract['horizon_hours'],
  forecast_end=keys+pd.Timedelta(hours=contract['horizon_hours']),model_issuance=str(issuance),selected_recipe=name),index=keys)
 if output.exists():raise FileExistsError(output)
 result.to_csv(output,index_label='decision_time')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--run',type=Path,required=True)
 p.add_argument('--features',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();print(forecast(a.run,a.features,a.output).tail().to_string())
