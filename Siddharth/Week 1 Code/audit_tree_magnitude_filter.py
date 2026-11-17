"""Read-only audit of tree magnitude filter artifacts; never fits predictive models."""
from pathlib import Path
import json,hashlib,argparse
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import joblib
import tree_magnitude_filter as m
import canonical_return_features as c

def main():
 raw,oy,dev,report=m.old.load()
 r=c.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet')
 sourcehash=hashlib.sha256((m.ROOT/'Code/tree_magnitude_filter.py').read_bytes()).hexdigest()
 parser=argparse.ArgumentParser();parser.add_argument('--horizons',type=int,nargs='+',default=[48,1,4,24,72,120]);args=parser.parse_args()
 results=[]
 for h in args.horizons:
  out=m.OUT/f'{h}h'
  if not (out/'COMPLETE.json').exists():continue
  if (out/'AUDIT.json').exists():
   results.append(json.loads((out/'AUDIT.json').read_text()));print(f'{h}h audit already PASS; reused',flush=True);continue
  done=json.loads((out/'COMPLETE.json').read_text())
  assert done['source_code_sha256']==sourcehash
  y=c.make_labels(r,raw.index,h)
  x=raw.loc[y.index].replace([np.inf,-np.inf],np.nan)
  p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
  assert p.index.equals(dev.intersection(y.index)) and p.index.is_unique
  np.testing.assert_allclose(p.truth,y.loc[p.index,'magnitude_log_bps'],atol=1e-8)
  selected=json.loads((out/'selections.json').read_text());checked=0;gates=0;ties=[]
  for sel in selected:
   issue=pd.Timestamp(sel['issue']);folder=out/issue.strftime('%Y%m%d')
   eng=m.engineered(x,sel['engineering_parents'])
   frame=pd.concat([x,eng],axis=1)
   # Prefix and future-perturbation check for all engineered predictors.
   cutoff=x.index[len(x)//2]
   prefix=m.engineered(x.loc[:cutoff],sel['engineering_parents'])
   pd.testing.assert_frame_equal(prefix,eng.loc[prefix.index])
   modified=x.copy();modified.loc[modified.index>cutoff,sel['engineering_parents']]+=10000
   np.testing.assert_allclose(m.engineered(modified,sel['engineering_parents']).loc[:cutoff],eng.loc[:cutoff],equal_nan=True)
   search=m.FoldSearch(frame,y,issue,h,folder)
   log=json.loads((folder/'fits.json').read_text())
   for item in log:
    art=joblib.load(folder/item['file'])
    start=pd.Timestamp(art['issue']);tr=y.index[y.label_end<start]
    assert y.loc[tr,'label_end'].max()<start
    assert art['train_rows']==len(tr) and art['max_label_end']==str(y.loc[tr,'label_end'].max())
    assert art['model'].__class__.__name__ in ['CatBoostRegressor','LGBMRegressor','RandomForestRegressor']
    expected=frame.loc[tr,art['columns']].median().fillna(0)
    np.testing.assert_allclose(art['medians'],expected,rtol=0,atol=0)
    checked+=1
   # Populate validation cache exclusively from persisted fits, no fitting.
   unique={(a['kind'],tuple(sorted(a['columns']))) for a in log}
   for kind,cols in unique:
    key=(kind,cols);slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    arts=[];preds=[]
    for n,(_,tr,va) in enumerate(search.folds):
     art=joblib.load(folder/f'{slug}_inner{n}.joblib')
     arts.append(art);preds.append(pd.Series(m.predict(art,frame.loc[va]),index=va))
    pred=pd.concat(preds)
    search.cache[key]=dict(p=pred,metrics=m.metrics(pred,search.truth),arts=arts,cols=list(cols),kind=kind)
   contributions=json.loads((folder/'contributions.json').read_text())
   for a in contributions:
    if a['stage']=='drop_ablation':continue
    kind=a['kind'];cols=a['reference_columns'];feature=a['feature']
    ref=search.cache[(kind,tuple(sorted(cols)))]
    trial=search.cache[(kind,tuple(sorted(cols+[feature])))]
    gate=search.gate(trial,ref)
    for k in ['mae_gain','ic_gain','mean_monthly_ic_gain']:
     np.testing.assert_allclose(gate[k],a[k],rtol=1e-10,atol=1e-10)
    if gate['monthly_mae_wins']!=a['monthly_mae_wins']:
     monthly_gains=[]
     for month in sorted(set(search.truth.index.strftime('%Y-%m'))):
      ix=search.truth.index.strftime('%Y-%m')==month
      monthly_gains.append(float(abs(ref['p'][ix]-search.truth[ix]).mean()-abs(trial['p'][ix]-search.truth[ix]).mean()))
     definite=sum(z>1e-9 for z in monthly_gains);ambiguous=sum(abs(z)<=1e-9 for z in monthly_gains)
     assert definite<=gate['monthly_mae_wins']<=definite+ambiguous
     assert definite<=a['monthly_mae_wins']<=definite+ambiguous
     assert not a['pass_gate'], 'An admitted feature with a count discrepancy requires separate investigation'
     ties.append(dict(issue=str(issue),feature=feature,saved_wins=a['monthly_mae_wins'],replayed_wins=gate['monthly_mae_wins'],monthly_mae_gains=monthly_gains,selection_unchanged=True))
    assert gate['pass_gate']==a['pass_gate']
    if a.get('selected'):
     assert a['pass_gate'] and a['permutation_pass']
     importance=search.permutation(trial,feature)
     np.testing.assert_allclose(importance['permutation_repeats'],a['permutation_repeats'],rtol=1e-10,atol=1e-10)
    gates+=1
   art=joblib.load(folder/'selected.joblib')
   assert art['columns']==sel['columns'] and art['kind']==sel['model']
   tr=y.index[y.label_end<issue]
   assert art['train_rows']==len(tr) and y.loc[tr,'label_end'].max()<issue
   test=p[p.issue==str(issue)]
   np.testing.assert_allclose(m.predict(art,frame.loc[test.index]),test.prediction,rtol=1e-10,atol=1e-8)
   t=pd.read_csv(folder/'train.csv',index_col=0,parse_dates=True)
   assert t.index.equals(tr)
   np.testing.assert_allclose(m.predict(art,frame.loc[tr]),t.prediction,rtol=1e-10,atol=1e-8)
   validation=pd.read_csv(folder/'validation.csv',index_col=0,parse_dates=True)
   saved=search.cache[(sel['model'],tuple(sorted(sel['columns'])))]
   np.testing.assert_allclose(saved['p'],validation.prediction,rtol=1e-10,atol=1e-8)
   importance=[dict(feature=col,**search.permutation(saved,col)) for col in sel['columns']]
   m.dump(folder/'selected_feature_permutation.json',importance)
   from predict_tree_magnitude import predict_frame
   inference=predict_frame(art,x)
   np.testing.assert_allclose(inference.loc[test.index,'magnitude_log_bps'],test.prediction,rtol=1e-10,atol=1e-8)
   rf=joblib.load(folder/'rf_baseline.joblib')
   np.testing.assert_allclose(m.predict(rf,x.loc[test.index]),test.rf_baseline,rtol=1e-10,atol=1e-8)
   # Recompute all eight pre-inner diagnostics, ensuring no outer data used.
   pool=[v for v in report if m.family(v)!='other']
   shortlist,diag=m.screen(x.loc[search.keys,pool],y.loc[search.keys,'magnitude_log_bps'],h,6,'raw')
   assert shortlist==sel['raw_shortlist']
   es,ed=m.screen(eng.loc[search.keys],y.loc[search.keys,'magnitude_log_bps'],h,8,'engineered')
   assert es==sel['engineered_shortlist']
   persisted=json.loads((folder/'screen.json').read_text())
   print(f'Auditing {h}h {issue.date()}: prediction replay complete',flush=True)
   for new in diag+ed:
    old=next(z for z in persisted if z['feature']==new['feature'] and z['stage']==new['stage'])
    assert new['reason']==old['reason']
    for field in ['missing_fraction','variance','spearman','vif','mutual_information','ks','lift','stability']:
     if field in new:np.testing.assert_allclose(new[field],old[field],rtol=1e-8,atol=1e-10)
  for score in done['scores']:
   split=score['split']
   f=pd.read_csv(out/('train.csv' if split=='train' else 'validation.csv' if split=='validation' else 'predictions.csv'),index_col=0,parse_dates=True)
   col='rf_baseline' if split=='test_rf_baseline' else 'report_hybrid' if split=='test_report_hybrid' else 'prediction'
   actual=m.metrics(f[col],f.truth)
   for key in actual:
    if actual[key] is not None:np.testing.assert_allclose(actual[key],score[key],rtol=1e-8,atol=1e-8)
  result=dict(horizon_hours=h,status='PASS',inner_models_checked=checked,admission_comparisons_replayed=gates,outer_models_checked=8,feature_causality=True,screen_recomputed=True,unique_test_rows=len(p),numerical_tie_notes=ties)
  m.dump(out/'AUDIT.json',result);results.append(result);print(result,flush=True)
 assert len(results)==len(args.horizons),'Requested horizons must complete'
 allresults=[json.loads(p.read_text()) for p in m.OUT.glob('*h/AUDIT.json')]
 m.dump(m.OUT/'AUDIT.json',dict(status='PASS' if len(allresults)==6 else 'PARTIAL',horizons=allresults,source_sha256=sourcehash))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
