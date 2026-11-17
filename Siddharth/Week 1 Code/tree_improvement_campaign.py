"""Monthly nested tree-only campaign. Explicit target transforms, median-leaf forests,
causal regime features and validation-selected ensembles. No fits on import."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
import joblib
import tree_magnitude_filter as old
import canonical_return_features as crf
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
ROOT=old.ROOT;OUT=ROOT/'Experiments/tree_improvement_campaign_v3'
PAIR=old.PAIR
RV='STATE__volatility__realized_vol_168h'
TREND='STATE__trend__net_return_168h'
CONFIGS={
'probe':dict(model='extra',target='log',depth=3,leaf=384,trees=64,window='exp'),
'rf_log':dict(model='rf',target='log',depth=3,leaf=384,trees=96,window='exp'),
'rf_sqrt_clip':dict(model='rf',target='sqrt',depth=3,leaf=384,trees=96,window='exp',clip=.995),
'extra_log':dict(model='extra',target='log',depth=4,leaf=384,trees=96,window='exp'),
'extra_12':dict(model='extra',target='log',depth=3,leaf=512,trees=96,window='12'),
'cat_raw':dict(model='cat',target='raw',depth=2,leaf=0,trees=120,window='exp'),
'lgb_raw':dict(model='lgb',target='raw',depth=2,leaf=512,trees=120,window='exp'),
'lgb_12':dict(model='lgb',target='raw',depth=3,leaf=768,trees=160,window='12'),
'lgb_normal':dict(model='lgb',target='normal',depth=2,leaf=768,trees=100,window='exp'),
'extra_weight':dict(model='extra',target='log',depth=3,leaf=512,trees=96,window='exp',half_life=180),
'extra_regime':dict(model='extra',target='log',depth=3,leaf=512,trees=96,window='exp',regime=True)}
def dump(p,obj):old.dump(p,obj)
def metric(p,y):
 result=old.metrics(p,y)
 result['rank_ic']=old.ic(np.round(np.asarray(p),10),np.asarray(y))
 return result
def keys(y,start,window):
 mask=y.label_end<start
 if window!='exp':mask&=y.index>=start-pd.DateOffset(months=int(window))
 return y.index[mask]
def state_frame(r,index):
 f=crf.features(r,index)
 f.columns=[c.replace('PRICE__','STATE__volatility__') for c in f]
 f[TREND]=r.rolling('168h',closed='left',min_periods=168*12).sum().reindex(index)
 return f
def engineering(raw,parents):
 f=old.engineered(raw,parents)
 extra={}
 for c in parents:
  extra['COVERAGE24__'+c]=raw[c].notna().astype(float).rolling('24h',closed='left',min_periods=24).mean()
 # Materialize cross-family products so any train-selected parent combination has a stable schema.
 for fa,fb in [('liquidation','flow'),('liquidation','volatility'),('flow','volatility')]:
  for a in [c for c in parents if old.family(c)==fa]:
   for b in [c for c in parents if old.family(c)==fb]:
    name='PRODUCT__'+a+'___'+b
    if name not in f:extra[name]=np.tanh(raw[a]/3)*np.tanh(raw[b]/3)
 return pd.concat([f,pd.DataFrame(extra,index=raw.index)],axis=1).astype('float32')
def wmedian(a,w):
 order=np.argsort(a,kind='stable');a=a[order];w=w[order]
 return float(a[np.searchsorted(np.cumsum(w),.5*np.sum(w),side='left')])
def forest_fit(x,y,w,cfg,hours):
 klass=RandomForestRegressor if cfg['model']=='rf' else ExtraTreesRegressor
 kwargs=dict(n_estimators=cfg['trees'],criterion='squared_error',max_depth=cfg['depth'],min_samples_leaf=max(cfg['leaf'],hours*8),max_features=.75 if x.shape[1]>2 else 1.,random_state=48,n_jobs=2)
 if cfg['model']=='rf':kwargs.update(bootstrap=True,max_samples=min(8192,len(y)))
 model=klass(**kwargs)
 cap=float(np.quantile(y,cfg['clip'])) if 'clip' in cfg else None
 fit_y=np.minimum(y,cap) if cap is not None else y
 transformed=np.log1p(fit_y) if cfg['target']=='log' else np.sqrt(fit_y) if cfg['target']=='sqrt' else fit_y
 model.fit(x,transformed,sample_weight=w)
 # Transform controls split topology; raw conditional median leaves prevent inverse-mean bias.
 summaries=[]
 for tree in model.estimators_:
  leaf=tree.apply(x);values={}
  for node in np.unique(leaf):
   ix=leaf==node;value=wmedian(y[ix],w[ix]) if cfg.get('half_life') else float(np.median(y[ix]))
   tree.tree_.value[node,0,0]=value;values[str(int(node))]=dict(rows=int(ix.sum()),value=value)
  summaries.append(values)
 return model,dict(leaf_statistics=summaries,target_clip=cap,leaf_estimation='original-target conditional median; transformed target used only for splits')
def regime_ids(frame,threshold):
 vol=frame[RV];trend=frame[TREND]
 result=(vol>threshold).astype(int)*2+(trend>0).astype(int)
 result.loc[vol.isna()|trend.isna()]=-1
 return result
def fit(frame,y,start,cfgid,cols,hours):
 cfg=CONFIGS[cfgid];tr=keys(y,start,cfg['window'])
 med=frame.loc[tr,cols].median().fillna(0);x=frame.loc[tr,cols].fillna(med).to_numpy()
 target=y.loc[tr,'magnitude_log_bps'].to_numpy()
 weight=np.exp2(-((start-tr).total_seconds()/86400)/cfg['half_life']) if cfg.get('half_life') else np.ones(len(tr))
 art=dict(config=cfgid,columns=cols,medians=med,issue=str(start),horizon_hours=hours,train_keys=tr,train_rows=len(tr),max_label_end=str(y.loc[tr,'label_end'].max()),fit_count=1)
 assert y.loc[tr,'label_end'].max()<start and np.isfinite(target).all()
 if cfg['model'] in ['rf','extra']:
  model,meta=forest_fit(x,target,weight,cfg,hours);art.update(model=model,**meta)
 elif cfg['model']=='cat':
  model=CatBoostRegressor(loss_function='MAE',iterations=cfg['trees'],depth=cfg['depth'],learning_rate=.025,l2_leaf_reg=100,random_seed=48,thread_count=2,verbose=False,allow_writing_files=False,bootstrap_type='No')
  model.fit(x,target);art['model']=model
 else:
  yy=target
  if cfg['target']=='normal':
   scale=frame.loc[tr,RV].to_numpy()*np.sqrt(hours/168)
   smed=float(np.nanmedian(scale));bounds=np.nanquantile(scale,[.01,.99])
   bounds=np.maximum(bounds,1.)
   scale=np.clip(np.nan_to_num(scale,nan=smed),*bounds)
   yy=target/scale;art.update(scale_median=smed,scale_bounds=bounds)
  model=LGBMRegressor(objective='regression_l1',n_estimators=cfg['trees'],max_depth=cfg['depth'],num_leaves=2**cfg['depth'],min_child_samples=max(cfg['leaf'],hours*8),learning_rate=.025,reg_lambda=100,reg_alpha=5,max_bin=31,random_state=48,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
  model.fit(x,yy);art['model']=model
 if cfg.get('regime'):
  threshold=float(frame.loc[tr,RV].median());state=regime_ids(frame.loc[tr],threshold)
  experts={}
  for group in range(4):
   ix=(state==group).to_numpy()
   if ix.sum()<max(2048,hours*16):continue
   local=dict(cfg,depth=2,trees=64);model,meta=forest_fit(x[ix],target[ix],weight[ix],local,hours)
   experts[str(group)]=dict(model=model,train_rows=int(ix.sum()),**meta);art['fit_count']+=1
  art.update(regime_threshold=threshold,experts=experts,expert_weight=.5)
 return art
def predict(art,frame):
 x=frame[art['columns']].fillna(art['medians']).to_numpy()
 # Fixed single-worker RF summation for deterministic tie handling.
 if isinstance(art['model'],(RandomForestRegressor,ExtraTreesRegressor)):art['model'].n_jobs=1
 p=np.asarray(art['model'].predict(x))
 if 'scale_bounds' in art:
  scale=frame[RV].to_numpy()*np.sqrt(art['horizon_hours']/168)
  p*=np.clip(np.nan_to_num(scale,nan=art['scale_median']),*art['scale_bounds'])
 if 'experts' in art:
  state=regime_ids(frame,art['regime_threshold'])
  for group,a in art['experts'].items():
   ix=(state==int(group)).to_numpy()
   if not ix.any():continue
   a['model'].n_jobs=1
   p[ix]=(1-art['expert_weight'])*p[ix]+art['expert_weight']*a['model'].predict(x[ix])
 return np.maximum(p,0.)
class Search:
 def __init__(self,frame,y,issue,hours,folder):
  self.frame=frame;self.y=y;self.issue=issue;self.hours=hours;self.folder=folder;self.cache={};self.log=[]
  starts=[issue-pd.DateOffset(months=6),issue-pd.DateOffset(months=3)]
  self.folds=[(s,y.index[(y.index>=s)&(y.index<e)&(y.label_end<issue)]) for s,e in zip(starts,starts[1:]+[issue])]
  self.vindex=self.folds[0][1].append(self.folds[1][1]);self.truth=y.loc[self.vindex,'magnitude_log_bps']
  self.pre=keys(y,starts[0],'exp')
 def evaluate(self,cfgid,cols):
  key=(cfgid,tuple(sorted(cols)))
  if key in self.cache:return self.cache[key]
  slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];pieces=[];arts=[]
  for n,(start,va) in enumerate(self.folds):
   file=self.folder/f'{slug}_inner{n}.joblib'
   if file.exists():art=joblib.load(file)
   else:
    art=fit(self.frame,self.y,start,cfgid,cols,self.hours);joblib.dump(art,file,compress=3)
   arts.append(art);pieces.append(pd.Series(predict(art,self.frame.loc[va]),index=va))
   self.log.append(dict(file=file.name,config=cfgid,columns=cols,issue=str(start),fit_count=art['fit_count']))
  p=pd.concat(pieces);result=dict(config=cfgid,cols=cols,arts=arts,p=p,metrics=metric(p,self.truth))
  self.cache[key]=result;return result
 def gate(self,a,b):
  maeg=b['metrics']['mae']-a['metrics']['mae'];icg=a['metrics']['rank_ic']-b['metrics']['rank_ic']
  months=self.vindex.strftime('%Y-%m');wins=[];igs=[]
  for month in sorted(set(months)):
   ix=months==month
   wins.append(float(abs(b['p'][ix]-self.truth[ix]).mean()-abs(a['p'][ix]-self.truth[ix]).mean())>.01)
   igs.append(metric(a['p'][ix],self.truth[ix])['rank_ic']-metric(b['p'][ix],self.truth[ix])['rank_ic'])
  return dict(mae_gain=maeg,ic_gain=icg,monthly_mae_wins=int(sum(wins)),mean_monthly_ic_gain=float(np.mean(igs)),passed=bool(maeg>.01 and icg>.0001 and sum(wins)>=4 and np.mean(igs)>0))
 def permutation(self,result,column):
  vals=[]
  for seed in [48,49,50]:
   pieces=[]
   for art,(_,va) in zip(result['arts'],self.folds):
    z=self.frame.loc[va].copy();ids=((va-va[0]).total_seconds()//(7*86400)).astype(int)
    blocks=[np.flatnonzero(ids==b) for b in np.unique(ids)]
    order=np.random.default_rng(seed).permutation(len(blocks));ix=np.concatenate([blocks[b] for b in order])
    z[column]=z[column].to_numpy()[ix]
    pieces.append(pd.Series(predict(art,z),index=va))
   s=metric(pd.concat(pieces),self.truth);vals.append([s['mae']-result['metrics']['mae'],result['metrics']['rank_ic']-s['rank_ic']])
  med=np.median(vals,axis=0)
  return dict(mae_importance=float(med[0]),ic_importance=float(med[1]),permutation_pass=bool(med[0]>0 and med[1]>0),permutations=vals)

def select_features(search,raw,eng,states,report):
 target=search.y.loc[search.pre,'magnitude_log_bps']
 candidates=[c for c in report if old.family(c)!='other']
 rawlist,rawdiag=old.screen(raw.loc[search.pre,candidates],target,search.hours,6,'raw')
 # All23 candidates' availability histories remain testable even if raw coverage is poor.
 permitted=list(old.engineered(raw,list(dict.fromkeys(PAIR+rawlist))).columns)+[c for c in eng if c.startswith('COVERAGE24__')]+list(states)
 transformed=search.frame.loc[search.pre,permitted]
 englist,engdiag=old.screen(transformed,target,search.hours,8,'engineered')
 controls,controldiag=old.screen(raw.loc[search.pre,PAIR],target,search.hours,2,'controls')
 base_trials=[search.evaluate('probe',cols) for cols in [PAIR,[PAIR[0]],[PAIR[1]]]]
 core=min(base_trials,key=lambda z:(z['metrics']['mae'],-z['metrics']['rank_ic']))
 # Global correlation test against retained controls and accepted candidate representatives.
 pool=[];redundant=[]
 for col in rawlist+englist:
  if col in core['cols']:continue
  representative=next((c for c in core['cols']+pool if abs(search.frame.loc[search.pre,[col,c]].corr(method='spearman').iloc[0,1])>=.95),None)
  if representative:redundant.append(dict(feature=col,representative=representative));continue
  pool.append(col)
 admissions=[];passed=[]
 for col in pool:
  trial=search.evaluate('probe',core['cols']+[col]);gate=search.gate(trial,core)
  record=dict(feature=col,**gate,**trial['metrics'])
  if gate['passed']:
   record.update(search.permutation(trial,col))
   if record['permutation_pass']:passed.append((col,trial))
  admissions.append(record)
 passed.sort(key=lambda a:(a[1]['metrics']['mae'],-a[1]['metrics']['rank_ic']))
 selected=core
 if passed:
  combined=core['cols']+[c for c,_ in passed[:2]]
  proposed=search.evaluate('probe',combined)
  if search.gate(proposed,core)['passed']:selected=proposed
  else:selected=passed[0][1]
 return core['cols'],selected['cols'],dict(raw_shortlist=rawlist,engineered_shortlist=englist,global_correlation_rejections=redundant,admissions=admissions,core=core['cols'],augmented=selected['cols']),rawdiag+engdiag+controldiag

def choose_ensemble(search,results,reference):
 proposals=[]
 for name,r in results.items():proposals.append(dict(weights={name:1.},p=r['p'],metrics=r['metrics']))
 top=sorted(results,key=lambda name:(results[name]['metrics']['mae'],-results[name]['metrics']['rank_ic']))[:3]
 if len(top)>1:
  p=sum(results[name]['p'] for name in top)/len(top)
  proposals.append(dict(weights={name:1/len(top) for name in top},p=p,metrics=metric(p,search.truth)))
  for i,a in enumerate(top):
   for b in top[i+1:]:
    for w in [.25,.5,.75]:
     p=w*results[a]['p']+(1-w)*results[b]['p']
     proposals.append(dict(weights={a:w,b:1-w},p=p,metrics=metric(p,search.truth)))
 for p in proposals:p['gate']=search.gate(p,reference)
 passing=[p for p in proposals if p['gate']['passed']]
 chosen=min(passing or proposals,key=lambda p:(p['metrics']['mae'],-p['metrics']['rank_ic']))
 return chosen,[dict(weights=p['weights'],**p['metrics'],**p['gate']) for p in proposals]

def run(hours,raw,original_y,dev,report,r):
 out=OUT/f'{hours}h';out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():print(f'{hours}h complete; reused',flush=True);return
 y=crf.make_labels(r,raw.index,hours)
 if hours==48:np.testing.assert_allclose(y.magnitude_log_bps,original_y.magnitude_log_bps,rtol=0,atol=1e-8)
 parents=list(dict.fromkeys(PAIR+[c for c in report if old.family(c)!='other']))
 eng=engineering(raw,parents);states=state_frame(r,raw.index)
 frame=pd.concat([raw,eng,states],axis=1).replace([np.inf,-np.inf],np.nan).astype('float32')
 frame=frame.loc[y.index];raw=raw.loc[y.index];eng=eng.loc[y.index];states=states.loc[y.index]
 dev=dev.intersection(y.index)
 oldref=pd.read_csv(ROOT/'Experiments/residual_incremental_v1/oof_baseline.csv',index_col=0,parse_dates=True) if hours==48 else None
 issues=pd.date_range('2024-01-01','2025-01-01',freq='MS',tz='UTC')
 allp=[];allt=[];allv=[];decisions=[];totalfits=0
 for n,issue in enumerate(issues):
  folder=out/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  if (folder/'DONE.json').exists():
   dec=json.loads((folder/'DONE.json').read_text());decisions.append(dec);totalfits+=dec['fit_count']
   for name,collection in [('predictions',allp),('train',allt),('validation',allv)]:collection.append(pd.read_csv(folder/f'{name}.csv',index_col=0,parse_dates=True))
   continue
  search=Search(frame,y,issue,hours,folder)
  core,aug,feature_log,screen=select_features(search,raw,eng,states,report)
  results={};definitions={}
  for cfgid in CONFIGS:
   if cfgid=='probe':continue
   for pool,cols in [('core',core),('augmented',aug)]:
    if pool=='augmented' and cols==core:continue
    name=cfgid+'__'+pool
    results[name]=search.evaluate(cfgid,cols);definitions[name]=dict(config=cfgid,columns=cols)
  if hours==48:
   refp=oldref.loc[search.vindex,'baseline']
   np.testing.assert_allclose(oldref.loc[search.vindex,'truth'],search.truth,atol=1e-8,rtol=0)
   assert (pd.to_datetime(oldref.loc[search.vindex,'baseline_issuance'],utc=True)<=search.vindex).all()
   reference=dict(p=refp,metrics=metric(refp,search.truth))
  else:reference=search.evaluate('probe',PAIR)
  chosen,ensemble_log=choose_ensemble(search,results,reference)
  artifacts={};outerlog=[];train_union=pd.DatetimeIndex([],tz='UTC')
  for name,w in chosen['weights'].items():
   definition=definitions[name];art=fit(frame,y,issue,definition['config'],definition['columns'],hours)
   art.update(engineering_parents=parents)
   joblib.dump(art,folder/(name+'.joblib'),compress=3);artifacts[name]=art
   train_union=train_union.union(art['train_keys'])
   outerlog.append(dict(file=name+'.joblib',config=definition['config'],columns=definition['columns'],fit_count=art['fit_count']))
  # In-sample ensemble reporting uses intersection: every component was fitted on these rows.
  train_keys=next(iter(artifacts.values()))['train_keys']
  for art in artifacts.values():train_keys=train_keys.intersection(art['train_keys'])
  end=issues[n+1] if n+1<len(issues) else dev.max()+pd.Timedelta(nanoseconds=1)
  te=dev[(dev>=issue)&(dev<end)]
  def combined(idx):return sum(chosen['weights'][name]*predict(a,frame.loc[idx]) for name,a in artifacts.items())
  p=pd.DataFrame({'truth':y.loc[te,'magnitude_log_bps'],'prediction':combined(te),'issue':str(issue)})
  if hours==48:p['reference']=oldref.loc[te,'baseline']
  else:
   control=fit(frame,y,issue,'probe',PAIR,hours);joblib.dump(control,folder/'control.joblib',compress=3)
   p['reference']=predict(control,frame.loc[te]);outerlog.append(dict(file='control.joblib',fit_count=control['fit_count']))
  t=pd.DataFrame({'truth':y.loc[train_keys,'magnitude_log_bps'],'prediction':combined(train_keys),'issue':str(issue)})
  v=pd.DataFrame({'truth':search.truth,'prediction':chosen['p'],'reference':reference['p'],'issue':str(issue)})
  for name,a in [('predictions',p),('train',t),('validation',v)]:a.to_csv(folder/f'{name}.csv')
  dump(folder/'feature_selection.json',feature_log);dump(folder/'feature_tests.json',screen);dump(folder/'ensemble_candidates.json',ensemble_log)
  dump(folder/'inner_fits.json',search.log);dump(folder/'outer_fits.json',outerlog)
  dec=dict(issue=str(issue),horizon_hours=hours,weights=chosen['weights'],definitions={name:definitions[name] for name in chosen['weights']},validation_gate=chosen['gate'],validation_metrics=chosen['metrics'],fit_count=sum(z['fit_count'] for z in search.log+outerlog),training_rows=len(train_keys),trained_union_rows=len(train_union))
  dump(folder/'DONE.json',dec);decisions.append(dec);totalfits+=dec['fit_count'];allp.append(p);allt.append(t);allv.append(v)
  print(f'{hours}h {issue.date()} complete: {dec["fit_count"]} fits; {len(aug)} augmented inputs; validation gate {chosen["gate"]["passed"]}; weights {chosen["weights"]}',flush=True)
 p=pd.concat(allp);t=pd.concat(allt);v=pd.concat(allv)
 assert p.index.equals(dev) and p.index.is_unique
 if hours==48:
  previous=pd.read_csv(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',index_col=0,parse_dates=True)
  np.testing.assert_allclose(p.reference,previous.loc[p.index,'report_hybrid'],rtol=0,atol=1e-8)
 scores=[dict(split=s,unique_timestamps=f.index.nunique(),**metric(f[col],f.truth)) for s,f,col in [('train',t,'prediction'),('validation',v,'prediction'),('test',p,'prediction'),('test_reference',p,'reference')]]
 for name,a in [('predictions',p),('train',t),('validation',v)]:a.to_csv(out/f'{name}.csv')
 pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);dump(out/'selections.json',decisions)
 dump(out/'COMPLETE.json',dict(status='EXECUTED',horizon_hours=hours,scores=scores,fit_count=totalfits,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(scores),flush=True)
def preflight():
 rng=np.random.default_rng(48);x=rng.normal(size=(1600,3));y=np.exp(1+.7*x[:,0]+rng.normal(size=1600))
 model,meta=forest_fit(x,y,np.ones(len(y)),CONFIGS['probe'],48)
 for tree,stats in zip(model.estimators_,meta['leaf_statistics']):
  leaf=tree.apply(x)
  for node,rec in stats.items():
   assert rec['rows']==int(np.sum(leaf==int(node)))
   assert rec['value']==float(np.median(y[leaf==int(node)]))
   assert tree.tree_.value[int(node),0,0]==rec['value']
 model.n_jobs=1
 np.testing.assert_allclose(model.predict(x),np.mean([t.predict(x) for t in model.estimators_],axis=0),rtol=0,atol=1e-12)
 assert np.isfinite(model.predict(x)).all()
 dump(OUT/'PREFLIGHT.json',dict(status='PASS',synthetic_forest_fits=1,median_leaf_estimates_checked=True,ensemble_prediction_checked=True))

def main():
 p=argparse.ArgumentParser();p.add_argument('--horizons',type=int,nargs='+',default=[48]);args=p.parse_args()
 OUT.mkdir(exist_ok=True)
 if not (OUT/'PREFLIGHT.json').exists():preflight()
 raw,y,dev,report=old.old.load();r=crf.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet')
 dump(OUT/'CANONICAL_VERIFICATION.json',crf.validate(r,y))
 outliers=dict(rows=len(raw),duplicate_times=int(raw.index.duplicated().sum()),nonfinite_targets=int((~np.isfinite(y.magnitude_log_bps)).sum()),negative_magnitudes=int((y.magnitude_log_bps<0).sum()),target_quantiles={str(q):float(y.magnitude_log_bps.quantile(q)) for q in [.5,.9,.99,.995,1.]},test_policy='No valid testtarget excluded, clipped or winsorized; no extreme market event classified as an error merely because of size')
 dump(OUT/'DATA_OUTLIER_AUDIT.json',outliers)
 for h in args.horizons:
  if h not in [1,4,24,48,72,120]:raise ValueError('Use verified horizons1/4/24/48/72/120')
  run(h,raw,y,dev,report,r)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
