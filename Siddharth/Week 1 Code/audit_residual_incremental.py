"""Audit residual correction without fitting predictive models again."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import residual_incremental as ri
import numpy as np
import pandas as pd
def audit(out):
 c=json.loads((out/'CONTRACT.json').read_text())
 raw,y,dev,returns5,source_validation=ri.load_inputs(c['horizon_hours'],c['prices'],c.get('returns_5m'))
 price=None
 if c['prices']:
  q=pd.read_csv(c['prices']);q['decision_time']=pd.to_datetime(q.decision_time,utc=True);price=q.set_index('decision_time').price
 f=ri.build_frame(raw,y,price,returns5);base=ri.fw.nb.old.read_csv(out/'oof_baseline.csv');f['BASE__forecast']=base.baseline
 models={p.stem:ri.fw.joblib.load(p) for p in (out/'models').glob('*.joblib')}
 for name,a in models.items():
  start=a['issuance'];tk=ri.fw.keys_for(y,start,'exp')
  assert a['train_keys'].equals(tk) and y.loc[tk,'label_end'].max()<start
 for startstr,g in base.groupby('baseline_issuance'):
  start=pd.Timestamp(startstr);card=json.loads((out/'baseline'/f'{start:%Y-%m}.json').read_text())
  pp=ri.fw.prediction(models[f'{start:%Y-%m}__pair_exp'],raw,y,g.index)
  pr=ri.fw.prediction(models[f'{start:%Y-%m}__reference_rf'],raw,y,g.index);w=card['rf_weight']
  np.testing.assert_allclose((1-w)*pp+w*pr,g.baseline,rtol=0,atol=1e-8)
  assert (g.index>=start).all() and pd.Timestamp(card['max_inner_label_end'])<start
 np.testing.assert_allclose(base.residual,base.truth-base.baseline,rtol=0,atol=1e-9)
 np.testing.assert_allclose(base.truth,y.loc[base.index,'magnitude_log_bps'],rtol=0,atol=1e-9)
 ledger=json.loads((out/'residual_fit_ledger.json').read_text());artifacts={}
 for rec in ledger:
  a=json.loads((out/'corrections'/(rec['id']+'.json')).read_text())
  start=pd.Timestamp(a['issuance']);tk=ri.residual_keys(y,base,start)
  assert pd.DatetimeIndex(pd.to_datetime(a['train_keys'],utc=True)).as_unit('ns').equals(tk)
  assert y.loc[tk,'label_end'].max()<start
  # Recompute closed-form bin sufficient statistics for verification, not another model search.
  s=f.loc[tk,a['feature']];v=s.fillna(a['median']).to_numpy()
  assert abs(float(s.median())-a['median'])<1e-10
  cuts=np.unique(np.quantile(v,[1/3,2/3]));np.testing.assert_allclose(cuts,a['cuts'],atol=1e-10,rtol=0)
  b=np.searchsorted(cuts,v,side='right');res=base.loc[tk,'residual'].to_numpy();global_med=float(np.median(res))
  levels=[]
  for j in range(len(cuts)+1):
   ix=b==j;n=int(ix.sum());local=float(np.median(res[ix])) if n else global_med
   strength=n/(n+2048.) if n>=256 else 0.
   levels.append(float(np.clip(global_med+strength*(local-global_med),-50.,50.)))
  np.testing.assert_allclose(levels,a['levels'],atol=1e-9,rtol=0)
  artifacts[rec['id']]=a
 pred=ri.fw.nb.old.read_csv(out/'predictions.csv');assert pred.index.equals(dev) and pred.index.is_unique
 for path in sorted((out/'outer').glob('*_selection.json')):
  s=json.loads(path.read_text());outer=pd.Timestamp(s['outer']);earliest=pd.Timestamp(s['earliest_inner'])
  assert pd.Timestamp(s['screen_max_label_end'])<earliest and pd.Timestamp(s['latest_validation_label_end'])<outer
  val=ri.fw.nb.old.read_csv(out/'inner'/f'{outer:%Y-%m}.csv')
  assert (y.loc[val.index,'label_end']<outer).all()
  for start in [outer-pd.DateOffset(months=m) for m in [6,4,2]]:
   vk=val.index[(val.index>=start)&(val.index<start+pd.DateOffset(months=2))]
   for feature in s['candidates']:
    tag=f'{outer:%Y-%m}__{start:%Y-%m}__'+ri.hashlib.sha256(feature.encode()).hexdigest()[:12]
    np.testing.assert_allclose(ri.predict_bin(artifacts[tag],f.loc[vk,feature]),val.loc[vk,feature],rtol=0,atol=1e-9)
  for a in s['admission']:
   pp=np.maximum(0,val.baseline+a['weight']*val[a['feature']])
   g=ri.gate(pp,val.truth,val.baseline,val.index)
   assert g['eligible']==a['eligible'] and abs(g['mae']-a['mae'])<1e-9
  selected=s['selected']
  admitted={a['feature'] for a in s['admission'] if a['eligible']}
  assert set(selected['features']).issubset(admitted) and len(selected['features'])<=3
  keys=pred.index[pred.outer==str(outer)];corrections=[]
  for feature in selected['features']:
   tag=f'{outer:%Y-%m}__outer__'+ri.hashlib.sha256(feature.encode()).hexdigest()[:12]
   corrections.append(ri.predict_bin(artifacts[tag],f.loc[keys,feature]))
  expected=np.maximum(0,base.loc[keys,'baseline']+selected['weight']*(np.mean(corrections,axis=0) if corrections else 0))
  np.testing.assert_allclose(expected,pred.loc[keys,'selected'],rtol=0,atol=1e-8)
  assert np.max(abs(pred.loc[keys,'selected']-base.loc[keys,'baseline']))<=25.+1e-8
 scores=pd.read_csv(out/'scores.csv')
 for split,file in [('train_residual_in_sample','training_predictions.csv'),('validation_selected','validation_predictions.csv'),('test','predictions.csv')]:
  p=ri.fw.nb.old.read_csv(out/file)
  for _,row in scores.query('split==@split').iterrows():
   mae=float(np.sum(np.abs(p[row.model]-p.truth))/len(p));assert abs(mae-row.mae)<1e-9
   assert abs(float(p[row.model].corr(p.truth,method='spearman'))-row.rank_ic)<1e-10
 for path,digest in json.loads((out/'source_hashes.json').read_text()).items():
  source=Path(path)
  if source.name=='residual_incremental.py' and ri.hashlib.sha256(source.read_bytes()).hexdigest()!=digest:source=out/'training_source_snapshot.py'
  assert ri.hashlib.sha256(source.read_bytes()).hexdigest()==digest
 if c.get('returns_5m'):
  assert source_validation['status']=='PASS'
  for path,digest in json.loads((out/'returns_source_hash.json').read_text()).items():assert ri.hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 ri.dump(out/'VERIFICATION.json',dict(status='PASS',baseline_models=len(models),residual_models=len(artifacts),unique_test_rows=len(pred),
  oof_baseline_replay=True,strict_maturation=True,feature_admission=True,bin_statistics=True,inner_and_outer_replay=True,
  correction_bound_bps=25,score_arithmetic=True,source_hashes=True,new_predictive_model_fits=0))
 print('AUDIT PASS',len(artifacts),'residual artifacts;',len(pred),'test rows',flush=True)
if __name__=='__main__':audit(Path(sys.argv[1]))
