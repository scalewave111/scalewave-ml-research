"""Document-driven tree-only follow-up: lookback selection, early stopping, VIF tracks."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
import tree_magnitude_filter as m
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor,early_stopping
from threadpoolctl import threadpool_limits
OUT=m.ROOT/'Experiments/tree_document_methods_v2'

def keys(y,start,window):
 ok=y.label_end<start
 if window!='exp':ok&=y.index>=start-pd.DateOffset(months=int(window))
 return y.index[ok]
def strict_pool(frame,ordered,max_features=8):
 chosen=[];log=[]
 for col in ordered:
  if col in chosen:continue
  dup=next((other for other in chosen if abs(frame[[col,other]].corr(method='spearman').iloc[0,1])>=.95),None)
  if dup:log.append(dict(feature=col,action='correlation_redundant',representative=dup));continue
  chosen.append(col)
 # Iterative VIF among current feature set, always retaining a liquidation input.
 while len(chosen)>1:
  z=frame[chosen].iloc[::max(1,len(frame)//3000)].copy().fillna(frame[chosen].median()).fillna(0)
  z=(z-z.mean())/z.std().replace(0,1);vifs={}
  for col in chosen:
   a=z[col].to_numpy();design=np.column_stack([np.ones(len(a)),z.drop(columns=col).to_numpy()])
   residual=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
   vifs[col]=min(1e12,float(np.sum((a-a.mean())**2)/max(np.sum(residual**2),1e-12)))
  removable=[col for col in chosen if any(m.family(v)=='liquidation' for v in chosen if v!=col)]
  worst=max(removable,key=lambda col:(vifs[col],chosen.index(col))) if removable else None
  if worst is None or vifs[worst]<=3.3:break
  chosen.remove(worst);log.append(dict(feature=worst,action='VIF_removed',vif=vifs[worst]))
 if len(chosen)>max_features:
  keep=chosen[:max_features]
  if not any(m.family(v)=='liquidation' for v in keep):
   keep[-1]=next(v for v in chosen if m.family(v)=='liquidation')
  log.extend(dict(feature=c,action='search_cap') for c in chosen if c not in keep);chosen=keep
 if not any(m.family(v)=='liquidation' for v in chosen):raise ValueError('Pool lacks liquidation')
 log.extend(dict(feature=c,action='retained') for c in chosen)
 return chosen,log
def model(algorithm,n):
 if algorithm=='cat':
  return CatBoostRegressor(loss_function='MAE',iterations=n,depth=3,learning_rate=.025,l2_leaf_reg=100,random_seed=48,thread_count=2,verbose=False,allow_writing_files=False,bootstrap_type='No')
 return LGBMRegressor(objective='regression_l1',n_estimators=n,max_depth=3,num_leaves=8,min_child_samples=512,learning_rate=.025,reg_lambda=100,reg_alpha=5,max_bin=31,random_state=48,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
def fit(x,y,start,kind):
 algorithm,window=kind.split('_')
 tr=keys(y,start,window);cut=start-pd.Timedelta(days=30)
 sub=tr[y.loc[tr,'label_end'].to_numpy()<cut]
 ev=tr[tr>=cut]
 assert len(sub)>1000 and len(ev)>500 and y.loc[sub,'label_end'].max()<cut and y.loc[ev,'label_end'].max()<start
 med=x.loc[sub].median().fillna(0)
 probe=model(algorithm,300)
 xx=x.loc[sub].fillna(med).to_numpy();vv=x.loc[ev].fillna(med).to_numpy()
 if algorithm=='cat':
  probe.fit(xx,y.loc[sub,'magnitude_log_bps'],eval_set=(vv,y.loc[ev,'magnitude_log_bps']),early_stopping_rounds=30)
  n=max(1,int(probe.get_best_iteration())+1)
 else:
  probe.fit(xx,y.loc[sub,'magnitude_log_bps'],eval_set=[(vv,y.loc[ev,'magnitude_log_bps'])],callbacks=[early_stopping(30,verbose=False)])
  n=max(1,int(probe.best_iteration_))
 med=x.loc[tr].median().fillna(0);final=model(algorithm,n)
 final.fit(x.loc[tr].fillna(med).to_numpy(),y.loc[tr,'magnitude_log_bps'])
 return dict(model=final,medians=med,columns=list(x),kind=kind,window=window,issue=str(start),horizon_hours=48,
  train_rows=len(tr),train_start=str(tr.min()),train_end=str(tr.max()),max_label_end=str(y.loc[tr,'label_end'].max()),
  selected_iterations=n,early_stop_train_rows=len(sub),early_stop_cut=str(cut),early_stop_train_max_label_end=str(y.loc[sub,'label_end'].max()),early_stop_validation_rows=len(ev),early_stop_validation_max_label_end=str(y.loc[ev,'label_end'].max()),predictive_fit_count=2)
class Search(m.FoldSearch):
 def evaluate(self,kind,cols):
  key=(kind,tuple(sorted(cols)))
  if key in self.cache:return self.cache[key]
  slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];arts=[];preds=[]
  for i,(start,tr,va) in enumerate(self.folds):
   path=self.out/f'{slug}_inner{i}.joblib'
   if path.exists():art=joblib.load(path)
   else:
    art=fit(self.x[cols],self.y,start,kind);joblib.dump(art,path,compress=3)
   arts.append(art);preds.append(pd.Series(m.predict(art,self.x.loc[va]),index=va))
   self.fitlog.append(dict(file=path.name,kind=kind,columns=cols,issue=str(start),selected_iterations=art['selected_iterations'],predictive_fit_count=2))
  p=pd.concat(preds);result=dict(p=p,metrics=m.metrics(p,self.truth),arts=arts,cols=cols,kind=kind)
  self.cache[key]=result;return result

def main():
 OUT.mkdir(exist_ok=True)
 raw,y,dev,report=m.old.load();raw=raw.replace([np.inf,-np.inf],np.nan)
 parents=list(dict.fromkeys(m.PAIR+[c for c in report if m.family(c)!='other']))
 eng=m.engineered(raw,parents);frame=pd.concat([raw,eng],axis=1)
 issues=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 ps=[];ts=[];vs=[];decisions=[];fitcount=0
 for i,issue in enumerate(issues):
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  if (folder/'DONE.json').exists():
   decisions.append(json.loads((folder/'DONE.json').read_text()))
   for name,collect in [('predictions',ps),('train',ts),('validation',vs)]:collect.append(pd.read_csv(folder/f'{name}.csv',index_col=0,parse_dates=True))
   fitcount+=decisions[-1]['fit_count'];continue
  search=Search(frame,y,issue,48,folder);recipes=[];poollog=[];screens=[]
  for window in ['6','12','exp']:
   pre=keys(y,search.folds[0][0],window)
   rawcols,rd=m.screen(raw.loc[pre,parents],y.loc[pre,'magnitude_log_bps'],48,8,'raw')
   ec,ed=m.screen(eng.loc[pre],y.loc[pre,'magnitude_log_bps'],48,12,'engineered')
   compact,cl=strict_pool(frame.loc[pre],m.PAIR)
   broad,bl=strict_pool(frame.loc[pre],list(dict.fromkeys(compact+rawcols+ec)))
   for d in rd+ed:d['window']=window
   screens+=rd+ed
   poollog.append(dict(window=window,compact=compact,broad=broad,compact_log=cl,broad_log=bl,screen_start=str(pre.min()),screen_end=str(pre.max())))
   for algorithm in ['cat','lgb']:
    for track,cols in [('compact',compact),('broad',broad)]:
     result=search.evaluate(algorithm+'_'+window,cols)
     recipes.append(dict(track=track,window=window,algorithm=algorithm,result=result))
  compact=min([z for z in recipes if z['track']=='compact'],key=lambda z:(z['result']['metrics']['mae'],-z['result']['metrics']['rank_ic']))['result']
  candidates=[z['result'] for z in recipes if z['track']=='broad' and search.gate(z['result'],compact)['pass_gate']]
  current=min(candidates,key=lambda z:(z['metrics']['mae'],-z['metrics']['rank_ic'])) if candidates else compact
  ablations=[]
  for col in list(current['cols']):
   rest=[c for c in current['cols'] if c!=col]
   if not rest or not any(m.family(c)=='liquidation' for c in rest):continue
   trial=search.evaluate(current['kind'],rest);gate=search.gate(trial,current)
   importance=search.permutation(current,col)
   ablations.append(dict(feature=col,reference_columns=current['cols'],kind=current['kind'],**gate,**importance,removed=gate['pass_gate']))
   if gate['pass_gate']:current=trial
  art=fit(frame[current['cols']],y,issue,current['kind']);art['engineering_parents']=parents
  joblib.dump(art,folder/'selected.joblib',compress=3)
  tr=keys(y,issue,art['window']);end=issues[i+1] if i+1<len(issues) else dev.max()+pd.Timedelta(nanoseconds=1)
  te=dev[(dev>=issue)&(dev<end)]
  p=pd.DataFrame({'truth':y.loc[te,'magnitude_log_bps'],'prediction':m.predict(art,frame.loc[te]),'issue':str(issue)})
  t=pd.DataFrame({'truth':y.loc[tr,'magnitude_log_bps'],'prediction':m.predict(art,frame.loc[tr]),'issue':str(issue)})
  v=pd.DataFrame({'truth':search.truth,'prediction':current['p'],'issue':str(issue)})
  for name,obj in [('predictions',p),('train',t),('validation',v)]:obj.to_csv(folder/f'{name}.csv')
  rec=[dict(track=z['track'],window=z['window'],algorithm=z['algorithm'],columns=z['result']['cols'],**z['result']['metrics'],**search.gate(z['result'],compact)) for z in recipes]
  m.dump(folder/'recipes.json',rec);m.dump(folder/'screens.json',screens);m.dump(folder/'pools.json',poollog);m.dump(folder/'ablations.json',ablations);m.dump(folder/'fits.json',search.fitlog)
  d=dict(issue=str(issue),model=current['kind'],columns=current['cols'],selected_iterations=art['selected_iterations'],validation_metrics=current['metrics'],fit_count=2*len(search.fitlog)+2)
  m.dump(folder/'DONE.json',d);decisions.append(d);ps.append(p);ts.append(t);vs.append(v);fitcount+=d['fit_count']
  print(f'Document v2 {issue.date()}: {current["kind"]}, {len(current["cols"])}features, {art["selected_iterations"]}trees; {d["fit_count"]}fits',flush=True)
 p=pd.concat(ps);t=pd.concat(ts);v=pd.concat(vs)
 assert p.index.equals(dev)
 ref=pd.read_csv(m.ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',index_col=0,parse_dates=True)
 p['report_hybrid']=ref.loc[p.index,'report_hybrid'];np.testing.assert_allclose(p.truth,ref.loc[p.index,'truth'],atol=1e-8)
 scores=[dict(split=s,unique_timestamps=f.index.nunique(),**m.metrics(f[col],f.truth)) for s,f,col in [('train',t,'prediction'),('validation',v,'prediction'),('test',p,'prediction'),('report_hybrid',p,'report_hybrid')]]
 for name,f in [('predictions',p),('train',t),('validation',v)]:f.to_csv(OUT/f'{name}.csv')
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 m.dump(OUT/'selections.json',decisions);m.dump(OUT/'COMPLETE.json',dict(status='EXECUTED',scores=scores,predictive_fit_count=fitcount,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(scores),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
