"""Independent saved-model replay audit for the monthly improvement campaign; no fits."""
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
import tree_improvement_campaign as m
from threadpoolctl import threadpool_limits

def check_forest(model,stats,x,y,w,weighted,full):
 assert len(model.estimators_)==len(stats)
 for tree,records in zip(model.estimators_,stats):
  assert sum(z['rows'] for z in records.values())==len(y)
  for node,rec in records.items():
   assert tree.tree_.value[int(node),0,0]==rec['value']
  if full:
   leaves=tree.apply(x)
   for node,rec in records.items():
    mask=leaves==int(node)
    assert int(mask.sum())==rec['rows']
    expected=m.wmedian(y[mask],w[mask]) if weighted else float(np.median(y[mask]))
    np.testing.assert_allclose(expected,rec['value'],rtol=0,atol=1e-10)

def validate_art(art,frame,y,full=False):
 cfg=m.CONFIGS[art['config']];issue=pd.Timestamp(art['issue']);tr=m.keys(y,issue,cfg['window'])
 assert art['train_keys'].equals(tr) and len(tr)==art['train_rows']
 assert y.loc[tr,'label_end'].max()<issue
 assert str(y.loc[tr,'label_end'].max())==art['max_label_end']
 med=frame.loc[tr,art['columns']].median().fillna(0)
 np.testing.assert_allclose(art['medians'],med,rtol=0,atol=0)
 assert art['model'].__class__.__name__ in ['RandomForestRegressor','ExtraTreesRegressor','CatBoostRegressor','LGBMRegressor']
 target=y.loc[tr,'magnitude_log_bps'].to_numpy();x=frame.loc[tr,art['columns']].fillna(med).to_numpy()
 w=np.exp2(-((issue-tr).total_seconds()/86400)/cfg['half_life']) if cfg.get('half_life') else np.ones(len(tr))
 if 'leaf_statistics' in art:check_forest(art['model'],art['leaf_statistics'],x,target,w,bool(cfg.get('half_life')),full)
 if 'target_clip' in art and art['target_clip'] is not None:
  assert art['target_clip']==float(np.quantile(target,cfg['clip']))
 if 'scale_bounds' in art:
  scale=frame.loc[tr,m.RV].to_numpy()*np.sqrt(art['horizon_hours']/168)
  np.testing.assert_allclose(art['scale_bounds'],np.maximum(np.nanquantile(scale,[.01,.99]),1.),rtol=0,atol=0)
  assert art['scale_median']==float(np.nanmedian(scale))
 if 'experts' in art:
  assert art['regime_threshold']==float(frame.loc[tr,m.RV].median())
  states=m.regime_ids(frame.loc[tr],art['regime_threshold'])
  expected={str(g) for g in range(4) if (states==g).sum()>=max(2048,art['horizon_hours']*16)}
  assert set(art['experts'])==expected
  for group,a in art['experts'].items():
   ix=(states==int(group)).to_numpy();assert a['train_rows']==int(ix.sum())
   check_forest(a['model'],a['leaf_statistics'],x[ix],target[ix],w[ix],bool(cfg.get('half_life')),full)
  assert art['fit_count']==1+len(art['experts'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,default=48);args=parser.parse_args();h=args.horizon
 out=m.OUT/f'{h}h'
 complete=json.loads((out/'COMPLETE.json').read_text())
 assert complete['source_sha256']==hashlib.sha256((m.ROOT/'Code/tree_improvement_campaign.py').read_bytes()).hexdigest()
 raw,oy,dev,report=m.old.old.load();r=m.crf.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet');y=m.crf.make_labels(r,raw.index,h)
 parents=list(dict.fromkeys(m.PAIR+[c for c in report if m.old.family(c)!='other']))
 eng=m.engineering(raw,parents);states=m.state_frame(r,raw.index)
 frame=pd.concat([raw,eng,states],axis=1).replace([np.inf,-np.inf],np.nan).astype('float32').loc[y.index]
 # Causal prefix and future perturbation tests cover new trend, volatility and availability features.
 for cutoff in [raw.index[len(raw)//3],raw.index[2*len(raw)//3]]:
  e=m.engineering(raw.loc[:cutoff],parents);pd.testing.assert_frame_equal(e,eng.loc[e.index])
  st=m.state_frame(r.loc[:cutoff],raw.index[raw.index<=cutoff]);pd.testing.assert_frame_equal(st,states.loc[st.index])
  changed=r.copy();changed.loc[changed.index>=cutoff]+=100000
  np.testing.assert_allclose(m.state_frame(changed,pd.DatetimeIndex([cutoff])),m.state_frame(r,pd.DatetimeIndex([cutoff])),rtol=0,atol=0,equal_nan=True)
 ref=pd.read_csv(m.ROOT/'Experiments/residual_incremental_v1/oof_baseline.csv',index_col=0,parse_dates=True) if h==48 else None
 decisions=json.loads((out/'selections.json').read_text());counts=dict(inner_artifacts=0,outer_artifacts=0,admission_tests=0,ensemble_proposals=0);fitcount=0
 # Recomputations must never trigger a new fit.
 def forbidden(*a,**kw):raise AssertionError('Audit attempted to fit a model')
 m.fit=forbidden
 for dec in decisions:
  issue=pd.Timestamp(dec['issue']);folder=out/issue.strftime('%Y%m%d');s=m.Search(frame,y,issue,h,folder)
  logs=json.loads((folder/'inner_fits.json').read_text());outerlogs=json.loads((folder/'outer_fits.json').read_text())
  fitcount+=sum(a['fit_count'] for a in logs+outerlogs)
  for row in logs:
   art=joblib.load(folder/row['file']);validate_art(art,frame,y,full=False);counts['inner_artifacts']+=1
  for cfg,cols in {(a['config'],tuple(sorted(a['columns']))) for a in logs}:
   key=(cfg,cols);slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];arts=[];pp=[]
   for n,(_,va) in enumerate(s.folds):
    art=joblib.load(folder/f'{slug}_inner{n}.joblib');arts.append(art);pp.append(pd.Series(m.predict(art,frame.loc[va]),index=va))
   p=pd.concat(pp);s.cache[key]=dict(config=cfg,cols=arts[0]['columns'],arts=arts,p=p,metrics=m.metric(p,s.truth))
  core,aug,featurelog,screen=m.select_features(s,raw.loc[y.index],eng.loc[y.index],states.loc[y.index],report)
  original=json.loads((folder/'feature_selection.json').read_text())
  assert core==original['core'] and aug==original['augmented']
  assert featurelog['global_correlation_rejections']==original['global_correlation_rejections']
  assert len(featurelog['admissions'])==len(original['admissions'])
  for a,b in zip(featurelog['admissions'],original['admissions']):
   assert a['feature']==b['feature'] and a['passed']==b['passed']
   for key in ['mae_gain','ic_gain','monthly_mae_wins','mean_monthly_ic_gain']:
    np.testing.assert_allclose(a[key],b[key],rtol=1e-8,atol=1e-8)
   if 'permutations' in a:np.testing.assert_allclose(a['permutations'],b['permutations'],rtol=1e-8,atol=1e-8)
  counts['admission_tests']+=len(original['admissions'])
  originalscreen=json.loads((folder/'feature_tests.json').read_text())
  assert len(screen)==len(originalscreen)
  for a,b in zip(screen,originalscreen):
   assert a['feature']==b['feature'] and a['reason']==b['reason']
   for key in ['missing_fraction','variance','vif','spearman','mutual_information','ks','lift','stability']:
    if key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-8,atol=1e-8)
  results={}
  for cfg in m.CONFIGS:
   if cfg=='probe':continue
   for pool,cols in [('core',core),('augmented',aug)]:
    if pool=='augmented' and cols==core:continue
    results[cfg+'__'+pool]=s.cache[(cfg,tuple(sorted(cols)))]
  if h==48:
   rp=ref.loc[s.vindex,'baseline'];reference=dict(p=rp,metrics=m.metric(rp,s.truth))
  else:reference=s.cache[('probe',tuple(sorted(m.PAIR)))]
  chosen,proposals=m.choose_ensemble(s,results,reference)
  assert chosen['weights']==dec['weights'] and chosen['gate']['passed']==dec['validation_gate']['passed']
  counts['ensemble_proposals']+=len(proposals)
  originalproposals=json.loads((folder/'ensemble_candidates.json').read_text())
  assert len(proposals)==len(originalproposals)
  for a,b in zip(proposals,originalproposals):
   assert a['weights']==b['weights'] and a['passed']==b['passed']
   for key in ['mae','rank_ic','mae_gain','ic_gain']:np.testing.assert_allclose(a[key],b[key],rtol=1e-8,atol=1e-8)
  arts={name:joblib.load(folder/(name+'.joblib')) for name in dec['weights']}
  for art in arts.values():validate_art(art,frame,y,full=True);counts['outer_artifacts']+=1
  p=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
  actual=sum(dec['weights'][name]*m.predict(a,frame.loc[p.index]) for name,a in arts.items())
  np.testing.assert_allclose(actual,p.prediction,rtol=1e-10,atol=1e-8)
  assert p.index.min()>=issue
  from predict_tree_campaign import predict_frame
  inference=predict_frame(folder,raw.loc[raw.index<=p.index.max()],r,start=p.index.min())
  np.testing.assert_allclose(inference.loc[p.index,'magnitude_log_bps'],p.prediction,rtol=1e-10,atol=1e-8)
  if h!=48:
   control=joblib.load(folder/'control.joblib');validate_art(control,frame,y,full=True)
   np.testing.assert_allclose(m.predict(control,frame.loc[p.index]),p.reference,rtol=1e-10,atol=1e-8)
  t=pd.read_csv(folder/'train.csv',index_col=0,parse_dates=True)
  tk=next(iter(arts.values()))['train_keys']
  for a in arts.values():tk=tk.intersection(a['train_keys'])
  assert t.index.equals(tk)
  np.testing.assert_allclose(sum(dec['weights'][name]*m.predict(a,frame.loc[tk]) for name,a in arts.items()),t.prediction,rtol=1e-10,atol=1e-8)
  v=pd.read_csv(folder/'validation.csv',index_col=0,parse_dates=True)
  assert v.index.equals(s.vindex);np.testing.assert_allclose(chosen['p'],v.prediction,rtol=1e-10,atol=1e-8)
  print(f'Audit {h}h {issue.date()} PASS',flush=True)
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
 assert p.index.equals(dev.intersection(y.index)) and p.index.is_unique
 np.testing.assert_allclose(p.truth,y.loc[p.index,'magnitude_log_bps'],rtol=0,atol=1e-8)
 if h==48:np.testing.assert_allclose(p.reference,ref.loc[p.index,'baseline'],rtol=0,atol=1e-8)
 for score in complete['scores']:
  split=score['split'];f=pd.read_csv(out/('train.csv' if split=='train' else 'validation.csv' if split=='validation' else 'predictions.csv'),index_col=0,parse_dates=True)
  values=m.metric(f['reference' if split=='test_reference' else 'prediction'],f.truth)
  for key,value in values.items():
   if value is not None:np.testing.assert_allclose(value,score[key],rtol=1e-8,atol=1e-8)
 assert fitcount==complete['fit_count']
 m.dump(out/'AUDIT.json',dict(status='PASS',horizon_hours=h,**counts,fit_count_checked=fitcount,outer_test_rows=len(p),label_purges=True,median_leaf_estimates=True,causal_engineering=True,selection_replayed=True,no_new_model_fits=True))
 print('Campaign audit PASS',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
