"""Independent saved-prediction and split audit; no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/nested_ic_rf_v3'

def read(path):
 d=pd.read_csv(path,index_col='decision_time'); d.index=pd.to_datetime(d.index,utc=True)
 return d

def audit():
 contract=json.loads((OUT/'CONTRACT.json').read_text())
 alltest=read(OUT/'all_predictions.csv')
 saved=read(OUT/'predictions.csv')
 overall=pd.read_csv(OUT/'overall_training_testing.csv')
 train=read(OUT/'training_predictions.csv')
 assert alltest.index.is_unique and alltest.index.equals(saved.index) and len(alltest)==3937
 selection_checks=[]
 for p in sorted((OUT/'outer').glob('*_selection.json')):
  s=json.loads(p.read_text()); outer=pd.Timestamp(s['outer'])
  inner=read(OUT/'inner'/f'{outer:%Y-%m}_predictions.csv')
  assert (inner.index+pd.Timedelta(hours=48)<outer).all()
  expected_months=pd.date_range(outer-pd.DateOffset(months=3),outer-pd.DateOffset(months=1),freq='MS')
  assert s['inner_starts']==list(map(str,expected_months))
  pure=[c['id'] for c in contract['configs'] if c['target']!='residual']
  ranks=sorted(pure,key=lambda c:(float(abs(inner[c]-inner.truth).mean()),-inner[c].corr(inner.truth,method='spearman'),c))
  assert ranks[0]==s['best_raw_rf'] and ranks[:3]==s['top_three']
  cands={c:inner[c].to_numpy() for c in pure+['residual_ic64']}
  cands['rf_top3_ensemble']=inner[ranks[:3]].mean(axis=1).to_numpy()
  cal={}
  for name,values in cands.items():
   scores=[]
   for scale in contract['calibration_scales']:
    for offset in contract['calibration_offsets']:
     loss=float(np.mean(abs(np.maximum(0,scale*values+offset)-inner.truth.to_numpy())))
     scores.append((loss,abs(scale-1)+abs(offset)/1000,scale,offset))
   loss,_,scale,offset=min(scores)
   assert abs(loss-s['calibrations'][name]['mae'])<1e-9
   assert scale==s['calibrations'][name]['scale'] and offset==s['calibrations'][name]['offset']
   cal[name]=loss
  assert min(cal,key=lambda c:(cal[c],c))==s['selected_forecast']
  assert min(pure+['rf_top3_ensemble'],key=lambda c:(cal[c],c))==s['selected_standalone']
  selection_checks.append(dict(month=str(outer),rows=len(inner),status='PASS'))
 for _,r in overall.iterrows():
  frame=train if r.split=='train_in_sample_pooled' else alltest
  target=frame.truth
  prediction=frame[r.model]
  mae=float(abs(prediction-target).mean())
  rmse=float(np.sqrt(np.mean((prediction-target)**2)))
  rank_ic=float(prediction.corr(target,method='spearman'))
  assert abs(mae-r.mae)<1e-9 and abs(rmse-r.rmse)<1e-9 and abs(rank_ic-r.rank_ic)<1e-9
 scoreboard=pd.read_csv(OUT/'scoreboard.csv')
 for _,r in scoreboard.iterrows():
  assert abs(float(abs(alltest[r.model]-alltest.truth).mean())-r.mae)<1e-9
 completed=[]
 for p in (OUT/'models').glob('*.joblib'):
  model=joblib.load(p); start=model['issuance']; keys=model['train_keys']
  assert keys.min()>=start-pd.DateOffset(months=3) and (keys+pd.Timedelta(hours=48)<start).all()
  assert not set(model['prep']['columns'])&{'magnitude_log_bps','signed_return_log_bps','direction_up','label_end'}
  assert model['model'].n_estimators==96 and model['model'].random_state==48
  completed.append(p.name)
 ledger=json.loads((OUT/'fit_ledger.json').read_text())
 assert len(completed)==sum(r['status']=='PASS' for r in ledger)
 assert all(r['status'] in ['PASS','INTERRUPTED'] for r in ledger)
 result=dict(status='PASS',no_fits=True,test_rows=len(alltest),unique_test_keys=True,
             completed_models=len(completed),attempts=len(ledger),interrupted=sum(r['status']=='INTERRUPTED' for r in ledger),
             inner_recipe_selection=selection_checks,all_overall_scores_recomputed=True,
             all_baseline_test_mae_recomputed=True,all_saved_model_windows_purged=True,
             repeated_training_rows=len(train),unique_training_timestamps=train.index.nunique())
 (OUT/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__': audit()
