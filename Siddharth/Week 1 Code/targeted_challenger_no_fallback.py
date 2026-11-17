"""Score the selected new-feature challengers without any baseline fallback."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import targeted_feature_extension as m
OUT=m.ROOT/'Experiments/targeted_challenger_no_fallback_v1'
def main():
 if (OUT/'COMPLETE.json').exists():raise RuntimeError('Completed result exists')
 trials=json.loads((m.OUT/'VALIDATION_TRIALS.json').read_text())
 source_decisions=json.loads((m.OUT/'SELECTIONS.json').read_text())
 raw,y,dev,_=m.old.load();raw=raw.replace([np.inf,-np.inf],np.nan)
 ref=pd.read_csv(m.OUT/'predictions.csv',index_col=0,parse_dates=True)
 pieces=[];train=[];validation=[];decisions=[];features=[]
 for original in source_decisions:
  issue=pd.Timestamp(original['issue']);inner=pd.Timestamp(original['inner'])
  choices=[t for t in trials if t['issue']==str(issue)]
  best=min(choices,key=lambda q:(q['mae'],-q['rank_ic'],len(q['features']),q['pack'],q['weight']))
  cols=best['features'];w=best['weight']
  assert len(cols)>2 and w>0
  frame=pd.concat([raw,m.engineering(raw,original['engineering_parents'])],axis=1)
  tk=y.index[y.label_end<issue];vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)]
  te=ref.index[ref.issue==str(issue)]
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  path=folder/'expanded_rf.joblib'
  art=m.fit(frame,y,tk,issue,cols,path)
  pair=joblib.load(original['pair_artifact'])
  assert pair['train_keys'].equals(tk)
  rf=m.predict(art,frame.loc[te]);pp=m.predict(pair,frame.loc[te]);hybrid=(1-w)*pp+w*rf
  pieces.append(pd.DataFrame(dict(expanded_rf=rf,expanded_hybrid=hybrid,baseline=ref.loc[te,'baseline'],truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te))
  train.append(pd.DataFrame(dict(expanded_rf=m.predict(art,frame.loc[tk]),expanded_hybrid=(1-w)*m.predict(pair,frame.loc[tk])+w*m.predict(art,frame.loc[tk]),truth=y.loc[tk,'magnitude_log_bps'],issue=str(issue)),index=tk))
  ia=joblib.load(best['artifact']);ip=joblib.load(m.BASE/'models'/f'{inner:%Y-%m}__report_inner_pair__expanding.joblib')
  vr=m.predict(ia,frame.loc[vk]);vh=(1-w)*m.predict(ip,frame.loc[vk])+w*vr
  vm=m.metric(vh,y.loc[vk,'magnitude_log_bps'])
  np.testing.assert_allclose([vm['mae'],vm['rank_ic']],[best['mae'],best['rank_ic']],rtol=0,atol=1e-8)
  validation.append(pd.DataFrame(dict(expanded_rf=vr,expanded_hybrid=vh,truth=y.loc[vk,'magnitude_log_bps'],issue=str(issue)),index=vk))
  # Reopen each artifact and replay its predictions; validate fitted preprocessing and purge.
  saved=joblib.load(path);assert saved['train_keys'].equals(tk) and pd.Timestamp(saved['max_label_end'])<issue
  expected=m.prep(frame.loc[tk],cols)
  for k in ['median','center','scale']:np.testing.assert_allclose(saved['prep'][k],expected[k],rtol=0,atol=0)
  np.testing.assert_allclose(m.predict(saved,frame.loc[te]),rf,rtol=0,atol=1e-8)
  for col in cols:features.append(dict(issue=str(issue),feature=col,role='control' if col in m.PAIR else 'additional',pack=best['pack']))
  d=dict(issue=str(issue),inner=str(inner),pack=best['pack'],weight=w,features=cols,engineering_parents=original['engineering_parents'],artifact=str(path),pair_artifact=original['pair_artifact'],validation=vm,test_rf=m.metric(rf,y.loc[te,'magnitude_log_bps']),test_hybrid=m.metric(hybrid,y.loc[te,'magnitude_log_bps']),baseline=m.metric(ref.loc[te,'baseline'],y.loc[te,'magnitude_log_bps']),train_rows=len(tk),test_rows=len(te))
  decisions.append(d);m.dump(folder/'selection.json',d);m.dump(OUT/'SELECTIONS.json',decisions)
  print('COMPLETE WINDOW',issue.date(),best['pack'],w,len(cols),flush=True)
 p=pd.concat(pieces);t=pd.concat(train);v=pd.concat(validation)
 assert p.index.equals(dev) and p.index.is_unique
 scores=[]
 for split,f in [('training',t),('validation',v),('test',p),('last3months',p.loc[p.index>=pd.Timestamp('2024-11-01',tz='UTC')])]:
  for model in ['expanded_rf','expanded_hybrid']+(['baseline'] if 'baseline' in f else []):
   scores.append(dict(split=split,model=model,**m.metric(f[model],f.truth),unique_rows=f.index.nunique()))
 for filename,f in [('predictions.csv',p),('training_predictions.csv',t),('validation_predictions.csv',v)]:f.to_csv(OUT/filename,index_label='decision_time')
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False);pd.DataFrame(features).to_csv(OUT/'features.csv',index=False)
 m.dump(OUT/'COMPLETE.json',dict(status='EXECUTED',new_fits=4,scores=scores,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection='Lowest inner-validation MAE among expanded-feature hybrids; ties higherIC,fewercolumns,pack,weight. No baseline fallback; outer outcomes not used.',standalone_note='Expanded RF uses the feature set selected for its hybrid, not an independently optimized standalone RF.',limitation='Retrospective exposed evaluation; diagnostic requested after gated procedure rejected these models.'))
 m.dump(OUT/'VERIFICATION.json',dict(status='PASS',reopened_outer_artifacts=4,all_predictions_replayed=True,training_only_preprocessing=True,strict_label_maturity=True,original_validation_scores_reproduced=True,no_baseline_fallback=True,unique_test_rows=len(p)))
 (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 print(pd.DataFrame(scores).to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
