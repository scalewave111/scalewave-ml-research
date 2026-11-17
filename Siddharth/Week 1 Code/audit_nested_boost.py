"""Independent post-run audit; no model fits."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nested_boost as nb
import predict_boost
import numpy as np
import pandas as pd
import joblib
def audit(out):
 c=json.loads((out/'CONTRACT.json').read_text())
 raw,y,dev,report=nb.load_data(c['horizon_hours'],c['price_source'])
 x=nb.features(raw,report);pred=nb.old.read_csv(out/'predictions.csv')
 first=pd.Timestamp(c['outer_months'][0])
 assert pred.index.equals(dev[dev>=first]) and pred.index.is_unique
 np.testing.assert_allclose(pred.truth,y.loc[pred.index,'magnitude_log_bps'],rtol=0,atol=1e-10)
 led=json.loads((out/'fit_ledger.json').read_text())
 for record in led:
  assert record['status']=='PASS'
  a=joblib.load(out/'models'/(record['id']+'.joblib'));s=pd.Timestamp(record['issuance'])
  expected=y.index[y.label_end<s] if record['id'].endswith('__expanding') else nb.train_keys(y,s)
  assert a['train_keys'].equals(expected)
  assert record['train_rows']==len(expected) and y.loc[expected,'label_end'].max()<s
  reconstructed=nb.prepare(x.loc[expected],a['prep']['columns'],standardize='cat_' not in record['id'] and 'lgb_' not in record['id'])
  for key in ['median','center','scale']:
   np.testing.assert_allclose(a['prep'][key],reconstructed[key],rtol=0,atol=1e-12)
 for selection in sorted((out/'outer').glob('*_selection.json')):
  s=json.loads(selection.read_text());outer=pd.Timestamp(s['outer'])
  val=nb.old.read_csv(out/'inner'/f'{outer:%Y-%m}.csv')
  assert (y.loc[val.index,'label_end']<outer).all()
  base=nb.metrics(val.rolling_pair,val.truth)
  if s['selected']!='rolling_pair':
   chosen=nb.metrics(val[s['selected']],val.truth)
   assert chosen['mae']<base['mae'] and nb.rankscore(chosen)>nb.rankscore(base)
 # Verify operational inference without labels against saved final-month predictions.
 temp=out/'inference_check';temp.mkdir(exist_ok=True)
 raw.to_csv(temp/'features.csv',index_label='decision_time')
 result=predict_boost.forecast(out,temp/'features.csv',temp/'replay.csv')
 common=result.index.intersection(pred.index)
 np.testing.assert_allclose(result.loc[common,'predicted_magnitude_log_bps'],pred.loc[common,'selected'],rtol=0,atol=1e-8)
 (temp/'features.csv').unlink()
 scores=pd.read_csv(out/'scores.csv')
 for _,row in scores[scores.split=='test'].iterrows():
  independent_mae=float(np.abs(pred[row.model].to_numpy()-pred.truth.to_numpy()).sum()/len(pred))
  assert abs(independent_mae-row.mae)<1e-9
 nb.dump(out/'INDEPENDENT_CHECK.json',dict(status='PASS',fits_audited=len(led),unique_test_rows=len(pred),
  all_train_populations=True,all_training_transforms=True,strict_label_purges=True,joint_selection_gate=True,
  inference_without_labels_rows=len(common),score_arithmetic=True,new_fits=0))
 print('INDEPENDENT AUDIT PASS',len(led),'fits;',len(common),'inference predictions',flush=True)
if __name__=='__main__':audit(Path(sys.argv[1]))
