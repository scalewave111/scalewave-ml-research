"""Audit document-driven experiment without refitting."""
import json,hashlib
import numpy as np
import pandas as pd
import joblib
import tree_document_methods as d
from threadpoolctl import threadpool_limits
from predict_tree_magnitude import predict_frame

def main():
 m=d.m;out=d.OUT
 done=json.loads((out/'COMPLETE.json').read_text())
 assert done['source_sha256']==hashlib.sha256((m.ROOT/'Code/tree_document_methods.py').read_bytes()).hexdigest()
 raw,y,dev,report=m.old.load();raw=raw.replace([np.inf,-np.inf],np.nan)
 parents=list(dict.fromkeys(m.PAIR+[c for c in report if m.family(c)!='other']))
 eng=m.engineered(raw,parents);frame=pd.concat([raw,eng],axis=1)
 decisions=json.loads((out/'selections.json').read_text());checked=0;comparisons=0
 for dec in decisions:
  issue=pd.Timestamp(dec['issue']);folder=out/issue.strftime('%Y%m%d');search=d.Search(frame,y,issue,48,folder)
  log=json.loads((folder/'fits.json').read_text())
  for row in log+[{ 'file':'selected.joblib'}]:
   a=joblib.load(folder/row['file']);start=pd.Timestamp(a['issue']);tr=d.keys(y,start,a['window']);cut=start-pd.Timedelta(days=30)
   sub=tr[y.loc[tr,'label_end'].to_numpy()<cut];ev=tr[tr>=cut]
   assert len(tr)==a['train_rows'] and len(sub)==a['early_stop_train_rows'] and len(ev)==a['early_stop_validation_rows']
   assert y.loc[sub,'label_end'].max()<cut and y.loc[ev,'label_end'].max()<start
   assert a['max_label_end']==str(y.loc[tr,'label_end'].max())
   assert a['early_stop_train_max_label_end']==str(y.loc[sub,'label_end'].max())
   assert 1<=a['selected_iterations']<=300
   assert a['model'].__class__.__name__ in ['CatBoostRegressor','LGBMRegressor']
   actual=a['model'].tree_count_ if a['kind'].startswith('cat') else a['model'].n_estimators_
   # LightGBM can stop adding trees before the requested count when no valid split exists.
   assert 1<=actual<=a['selected_iterations']
   np.testing.assert_allclose(a['medians'],frame.loc[tr,a['columns']].median().fillna(0),rtol=0,atol=0)
   checked+=1
  for kind,cols in {(z['kind'],tuple(sorted(z['columns']))) for z in log}:
   key=(kind,cols);slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];pred=[];arts=[]
   for n,(_,tr,va) in enumerate(search.folds):
    a=joblib.load(folder/f'{slug}_inner{n}.joblib');arts.append(a)
    pred.append(pd.Series(m.predict(a,frame.loc[va]),index=va))
   p=pd.concat(pred);search.cache[key]=dict(p=p,metrics=m.metrics(p,search.truth),arts=arts,cols=list(cols),kind=kind)
  recipes=json.loads((folder/'recipes.json').read_text())
  for rec in recipes:
   result=search.cache[(rec['algorithm']+'_'+rec['window'],tuple(sorted(rec['columns'])))]
   for key in ['mae','rank_ic']:np.testing.assert_allclose(rec[key],result['metrics'][key],rtol=1e-10,atol=1e-9)
  for rec in json.loads((folder/'ablations.json').read_text()):
   before=search.cache[(rec['kind'],tuple(sorted(rec['reference_columns'])))]
   after=search.cache[(rec['kind'],tuple(sorted(c for c in rec['reference_columns'] if c!=rec['feature'])))]
   gate=search.gate(after,before)
   assert gate['pass_gate']==rec['removed']
   importance=search.permutation(before,rec['feature'])
   np.testing.assert_allclose(importance['permutation_repeats'],rec['permutation_repeats'],atol=1e-8)
   comparisons+=1
  for pool in json.loads((folder/'pools.json').read_text()):
   window=pool['window'];pre=d.keys(y,search.folds[0][0],window)
   rc,_=m.screen(raw.loc[pre,parents],y.loc[pre,'magnitude_log_bps'],48,8,'raw')
   ec,_=m.screen(eng.loc[pre],y.loc[pre,'magnitude_log_bps'],48,12,'engineered')
   compact,cl=d.strict_pool(frame.loc[pre],m.PAIR)
   broad,bl=d.strict_pool(frame.loc[pre],list(dict.fromkeys(compact+rc+ec)))
   assert compact==pool['compact'] and broad==pool['broad']
   assert cl==pool['compact_log'] and bl==pool['broad_log']
  a=joblib.load(folder/'selected.joblib')
  assert a['columns']==dec['columns'] and a['kind']==dec['model']
  p=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
  np.testing.assert_allclose(m.predict(a,frame.loc[p.index]),p.prediction,rtol=1e-10,atol=1e-8)
  np.testing.assert_allclose(predict_frame(a,raw).loc[p.index,'magnitude_log_bps'],p.prediction,rtol=1e-10,atol=1e-8)
  v=pd.read_csv(folder/'validation.csv',index_col=0,parse_dates=True)
  result=search.cache[(dec['model'],tuple(sorted(dec['columns'])))]
  np.testing.assert_allclose(result['p'],v.prediction,rtol=1e-10,atol=1e-8)
  t=pd.read_csv(folder/'train.csv',index_col=0,parse_dates=True);tr=d.keys(y,issue,a['window'])
  assert t.index.equals(tr);np.testing.assert_allclose(m.predict(a,frame.loc[tr]),t.prediction,rtol=1e-10,atol=1e-8)
  print(f'Audited document v2 {issue.date()}',flush=True)
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
 assert p.index.is_unique and p.index.equals(dev)
 for score in done['scores']:
  split=score['split'];f=pd.read_csv(out/('train.csv' if split=='train' else 'validation.csv' if split=='validation' else 'predictions.csv'),index_col=0,parse_dates=True)
  actual=m.metrics(f['report_hybrid' if split=='report_hybrid' else 'prediction'],f.truth)
  for key,value in actual.items():
   if value is not None:np.testing.assert_allclose(value,score[key],rtol=1e-8,atol=1e-8)
 m.dump(out/'AUDIT.json',dict(status='PASS',saved_models_checked=checked,ablation_permutation_replayed=comparisons,test_rows=len(p),label_and_earlystop_purges=True,training_imputation=True,pools_recomputed=True,predictions_replayed=True,early_stopping_limit='Stored iteration counts and chronological sample boundaries checked; stopping trajectory not refit.'))
 print('Document v2 audit PASS',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
