"""Replay and audit controlled feature extension, without new model fits."""
import json,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import targeted_feature_extension as m
def main():
 out=m.OUT;complete=json.loads((out/'COMPLETE.json').read_text())
 assert complete['source_sha256']==hashlib.sha256((m.ROOT/'Code/targeted_feature_extension.py').read_bytes()).hexdigest()
 raw,y,dev,_=m.old.load();raw=raw.replace([np.inf,-np.inf],np.nan)
 r=m.crf.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet')
 x=pd.concat([raw,m.price_features(r,raw.index)],axis=1)
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
 assert p.index.equals(dev) and p.index.is_unique
 np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 mapping=json.loads((out/'USER_FEATURE_MAPPING.json').read_text());allowed=list(dict.fromkeys(a['column'] for a in mapping))
 assert len(allowed)==38 and allowed==json.loads((out/'PROTOCOL.json').read_text())['feature_columns']
 selections=json.loads((out/'SELECTIONS.json').read_text());trials=json.loads((out/'VALIDATION_TRIALS.json').read_text());ledger=json.loads((out/'FIT_LEDGER.json').read_text())
 assert len(ledger)<=32
 checked=0;trial_count=0
 for d in selections:
  issue=pd.Timestamp(d['issue']);inner=pd.Timestamp(d['inner']);folder=out/issue.strftime('%Y%m%d')
  assert set(d['engineering_parents']).issubset(allowed)
  assert all(c in allowed or c in m.PAIR or any(c.endswith('__'+a) for a in allowed) for c in d['features'])
  frame=pd.concat([x,m.engineering(x,d['engineering_parents'])],axis=1)
  tk=y.index[y.label_end<inner];ot=y.index[y.label_end<issue]
  vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)]
  te=p.index[p.issue==d['issue']]
  ip=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_pair__expanding.joblib')
  ir=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_rf__expanding.joblib')
  pp=m.predict(ip,frame.loc[vk]);rr=m.predict(ir,frame.loc[vk]);target=y.loc[vk,'magnitude_log_bps']
  bw=min([0.,.1,.25,.5],key=lambda w:(m.metric((1-w)*pp+w*rr,target)['mae'],w))
  assert bw==d['baseline_weight'];baseline=(1-bw)*pp+bw*rr
  bm=m.metric(baseline,target);cache={}
  for rec in [q for q in ledger if q['issue']==d['issue']]:
   a=joblib.load(rec['path']);keys=tk if rec['phase']=='inner' else ot
   assert a['train_keys'].equals(keys)
   assert pd.Timestamp(a['max_label_end'])<pd.Timestamp(a['issuance'])
   expected=m.prep(frame.loc[keys],rec['features'])
   for z in ['median','center','scale']:np.testing.assert_allclose(a['prep'][z],expected[z],rtol=0,atol=0)
   assert a['prep']['columns']==rec['features']
   for k,v in m.PARAMS.items():
    if k!='n_jobs':assert a['model'].get_params()[k]==v
   if rec['phase']=='inner':cache[rec['path']]=m.predict(a,frame.loc[vk])
   checked+=1
  window=[q for q in trials if q['issue']==d['issue']]
  for q in window:
   forecast=(1-q['weight'])*pp+q['weight']*cache[q['artifact']]
   met=m.metric(forecast,target);ab=m.metric((1-q['weight'])*pp+q['weight']*rr,target)
   for k in met:np.testing.assert_allclose(met[k],q[k],rtol=1e-10,atol=1e-8)
   wins=0
   for month in sorted(set(vk.strftime('%Y-%m'))):
    ix=vk.strftime('%Y-%m')==month
    wins+=m.metric(forecast[ix],target[ix])['mae']<m.metric(baseline[ix],target[ix])['mae']-.01
   passed=met['mae']<bm['mae']-.01 and met['rank_ic']>bm['rank_ic']+.0001 and wins>=2 and met['mae']<ab['mae']-.01
   assert passed==q['passed'] and wins==q['monthly_mae_wins'];trial_count+=1
  passing=[q for q in window if q['passed']]
  if passing:
   best=min(passing,key=lambda q:(q['mae'],-q['rank_ic'],len(q['features']),q['pack'],q['weight']))
   assert d['selected']==best['pack'] and d['weight']==best['weight'] and d['features']==best['features']
  else:assert d['selected']=='original_pair_rf' and d['weight']==bw
  op=joblib.load(d['pair_artifact']);a=joblib.load(d['artifact'])
  assert op['train_keys'].equals(ot) and a['train_keys'].equals(ot)
  prediction=(1-d['weight'])*m.predict(op,frame.loc[te])+d['weight']*m.predict(a,frame.loc[te])
  np.testing.assert_allclose(prediction,p.loc[te,'prediction'],rtol=0,atol=1e-7)
  # Training and validation score rows are checked against the selected saved artifacts.
  tp=pd.read_csv(out/'training_predictions.csv',index_col=0,parse_dates=True);tp=tp[tp.issue==d['issue']]
  np.testing.assert_allclose((1-d['weight'])*m.predict(op,frame.loc[ot])+d['weight']*m.predict(a,frame.loc[ot]),tp.prediction,rtol=0,atol=1e-7)
  vp=pd.read_csv(out/'validation_predictions.csv',index_col=0,parse_dates=True);vp=vp[vp.issue==d['issue']]
  vp_expected=baseline if not passing else (1-d['weight'])*pp+d['weight']*cache[best['artifact']]
  np.testing.assert_allclose(vp_expected,vp.prediction,rtol=0,atol=1e-7)
  # Training screens, not future labels, determine ranked feature packs.
  stats=pd.read_csv(folder/'feature_tests.csv',float_precision='round_trip')
  assert set(stats.feature)==set(allowed)
  ranked=stats[stats.quality].sort_values(['score','feature'],ascending=[False,True]).feature.tolist()
  packs=json.loads((folder/'feature_packs.json').read_text())
  for k in [4,8,16,32]:
   cols,_=m.prune(x,tk,ranked,k);assert cols==packs[f'pooled_{k}']
  from predict_targeted_feature_extension import forecast
  replay=forecast(d,raw,te)
  np.testing.assert_allclose(replay.predicted_magnitude_log_bps,p.loc[te,'prediction'],rtol=0,atol=1e-7)
  print('AUDIT',issue.date(),'PASS',flush=True)
 for score in complete['scores']:
  split=score['split'];fname='training_predictions.csv' if split=='training' else 'validation_predictions.csv' if split=='validation' else 'predictions.csv'
  f=pd.read_csv(out/fname,index_col=0,parse_dates=True)
  if split.startswith('last3months'):f=f.loc[f.index>=pd.Timestamp('2024-11-01',tz='UTC')]
  col='baseline' if split.endswith('baseline') else 'prediction'
  met=m.metric(f[col],f.truth)
  for k in met:np.testing.assert_allclose(met[k],score[k],rtol=1e-10,atol=1e-8)
 m.dump(out/'AUDIT.json',dict(status='PASS',saved_artifacts_checked=checked,validation_candidates_recomputed=trial_count,all_outer_predictions_replayed=True,training_and_validation_replayed=True,training_only_preprocessing=True,feature_pack_redundancy_replayed=True,strict_label_purges=True,test_rows=len(p),new_fits=0))
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
