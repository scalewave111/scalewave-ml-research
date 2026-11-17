"""Tree-only nested walk-forward experiment with eight documented feature tests.
No training on import. All selection uses pre-issuance data.
"""
from pathlib import Path
import sys,json,argparse,hashlib,time,warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.boost_deps'))
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr,ks_2samp
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
import nested_ic_rf as old
import canonical_return_features as canonical
warnings.filterwarnings('ignore',message='X does not have valid feature names')
PAIR=[f'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__{w}__value' for w in ['15m','5m']]
CONFIGS=['lgb_stump','lgb_depth2','cat_depth2','rf_depth3']
OUT=ROOT/'Experiments/tree_magnitude_filter_v1'

def dump(path,obj):
 path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def ic(p,y):
 return float(spearmanr(p,y).statistic) if len(p)>2 and np.ptp(p)>0 and np.ptp(y)>0 else 0.
def metrics(p,y):
 p=np.asarray(p);y=np.asarray(y)
 assert len(p) and np.isfinite(p).all() and np.isfinite(y).all()
 return dict(mae=float(abs(p-y).mean()),rank_ic=ic(p,y),pearson_ic=float(np.corrcoef(p,y)[0,1]) if np.ptp(p)>0 else None,rmse=float(np.sqrt(np.mean((p-y)**2))),rows=len(p))
def family(c):
 for f in ['liquidation','flow','volatility']:
  if '__'+f+'__' in c:return f
 return 'other'
def effect(a,y):
 good=np.isfinite(a)&np.isfinite(y);a=np.asarray(a)[good];y=np.asarray(y)[good]
 if len(a)<80 or np.unique(a).size<3:return 0.,0.,1.
 lo,hi=np.quantile(y,[.25,.75])
 ks=float(ks_2samp(a[y<=lo],a[y>=hi],method='asymp').statistic)
 bins=np.searchsorted(np.unique(np.quantile(a,[.2,.4,.6,.8])),a)
 meds=[np.median(y[bins==b]) for b in np.unique(bins) if np.sum(bins==b)>=20]
 return ic(a,y),ks,float(max(meds)/max(min(meds),1e-6)) if len(meds)>1 else 1.
def screen(x,y,hours,cap,stage):
 """Every candidate gets quality, association, MI-null, VIF and redundancy records."""
 rec=[];clean=x.replace([np.inf,-np.inf],np.nan)
 quality=[]
 for c in clean:
  a=clean[c].dropna(); counts=a.value_counts(normalize=True)
  spread=float(a.quantile(.99)-a.quantile(.01)) if len(a) else 0.
  ok=len(a)>=1000 and len(a)/len(x)>=.7 and len(counts)>=3 and float(counts.iloc[0])<.995 and spread>1e-10
  r=dict(feature=c,family=family(c),stage=stage,missing_fraction=float(1-len(a)/len(x)),finite_rows=len(a),unique=len(counts),max_point_mass=float(counts.iloc[0]) if len(a) else 1.,variance=float(a.var()) if len(a)>1 else 0.,spread_01_99=spread,quality_pass=bool(ok),reason='quality_failed' if not ok else '')
  rec.append(r)
  if ok:quality.append(c)
 # VIF uses standardized, median-imputed training values, no target.
 z=clean[quality].iloc[::max(1,len(x)//3000)].copy()
 z=z.fillna(clean[quality].median()); z=(z-z.mean())/z.std().replace(0,1)
 vif={}
 for c in quality:
  others=z.drop(columns=c).to_numpy();a=z[c].to_numpy()
  if others.shape[1]==0:vif[c]=1.;continue
  design=np.column_stack([np.ones(len(a)),others])
  residual=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
  ratio=float(np.dot(residual,residual)/max(np.dot(a-a.mean(),a-a.mean()),1e-15))
  vif[c]=min(1e12,1/max(ratio,1e-12))
 quarters=x.index.strftime('%Y')+'Q'+x.index.quarter.astype(str)
 recent=sorted(set(quarters))[-4:]
 # Spacing >= target horizon reduces overlapping labels in MI diagnostic.
 sample=x.index[::max(1,hours*2)]
 for r in rec:
  c=r['feature']
  if c not in quality:continue
  rho,ks,lift=effect(clean[c].to_numpy(),y.to_numpy())
  effects=[effect(clean.loc[quarters==q,c].to_numpy(),y.loc[quarters==q].to_numpy()) for q in recent]
  stable=float(np.mean([(abs(a)>=.02 and np.sign(a)==np.sign(rho)) or (b>=.10 and d>=1.10) for a,b,d in effects]))
  a=clean.loc[sample,c].to_numpy();b=y.loc[sample].to_numpy()
  ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok];mi=0.;null=0.
  if len(a)>=60 and np.unique(a).size>=3:
   mi=float(mutual_info_regression(a[:,None],b,random_state=48,n_neighbors=3)[0])
   null=max(float(mutual_info_regression(a[:,None],np.roll(b,max(1,int(len(b)*f))),random_state=48,n_neighbors=3)[0]) for f in [.23,.47,.71])
  signal=abs(rho)>=.02 or (ks>=.10 and lift>=1.10) or (mi>.01 and mi>null)
  r.update(spearman=rho,ks=ks,lift=lift,mutual_information=mi,mi_shift_null_max=null,mi_rows=len(a),stability=stable,vif=vif[c],vif_high=vif[c]>10,signal_pass=bool(signal),score=abs(rho)+ks+max(0,mi-null),reason='signal_failed' if not signal else ('unstable' if stable<.5 else 'eligible'))
 ranked=sorted([r for r in rec if r['reason']=='eligible'],key=lambda r:(-r['score'],r['feature']))
 selected=[];counts={}
 for r in ranked:
  c=r['feature'];redundant=None
  for keep in selected:
   correlation=clean[[c,keep]].corr(method='spearman').iloc[0,1]
   if np.isfinite(correlation) and abs(correlation)>=.95:redundant=keep;break
  if redundant:r.update(reason='redundant',correlation_representative=redundant);continue
  f=r['family']
  if stage=='raw' and counts.get(f,0)>=2:r['reason']='family_search_cap';continue
  if len(selected)>=cap:r['reason']='search_cap';continue
  selected.append(c);counts[f]=counts.get(f,0)+1;r['reason']='shortlisted'
 return selected,rec

def engineered(raw,parents):
 result={}
 for c in parents:
  s=raw[c]
  mean=s.rolling('24h',closed='left',min_periods=24).mean()
  sd=s.rolling('24h',closed='left',min_periods=24).std()
  result['MEAN24__'+c]=mean
  result['DELTA1__'+c]=s-s.shift(freq='1h').reindex(s.index)
  result['Z24__'+c]=((s-mean)/sd.clip(lower=1e-4)).clip(-20,20)
  result['SLOG__'+c]=np.sign(s)*np.log1p(abs(s))
 # Bounded cross-family interactions. Parents ordered by train-only evidence.
 first={}
 for c in parents:first.setdefault(family(c),c)
 for a,b in [('liquidation','flow'),('liquidation','volatility'),('flow','volatility')]:
  if a in first and b in first:
   ca,cb=first[a],first[b]
   result['PRODUCT__'+ca+'___'+cb]=np.tanh(raw[ca]/3)*np.tanh(raw[cb]/3)
 return pd.DataFrame(result,index=raw.index).replace([np.inf,-np.inf],np.nan).astype('float32')

def make_model(kind):
 if kind.startswith('lgb'):
  depth=1 if kind=='lgb_stump' else 2
  return LGBMRegressor(objective='regression_l1',n_estimators=48 if depth==1 else 64,max_depth=depth,num_leaves=2**depth,min_child_samples=512,learning_rate=.03,reg_lambda=50,reg_alpha=5,max_bin=31,random_state=48,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
 if kind=='cat_depth2':
  return CatBoostRegressor(loss_function='MAE',iterations=80,depth=2,learning_rate=.03,l2_leaf_reg=50,random_seed=48,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='No')
 return RandomForestRegressor(n_estimators=48,criterion='absolute_error',max_depth=3,min_samples_leaf=128,max_features=1.,bootstrap=True,max_samples=4096,random_state=48,n_jobs=4)
def train(x,y,keys,kind):
 med=x.loc[keys].median().fillna(0)
 model=make_model(kind)
 model.fit(x.loc[keys].fillna(med).to_numpy(),y.loc[keys].to_numpy())
 return dict(model=model,medians=med,columns=list(x),kind=kind,train_start=str(keys.min()),train_end=str(keys.max()),train_rows=len(keys))
def predict(artifact,x):
 return np.maximum(0,artifact['model'].predict(x[artifact['columns']].fillna(artifact['medians']).to_numpy()))

class FoldSearch:
 def __init__(self,x,y,issue,hours,out):
  self.x=x;self.y=y;self.issue=issue;self.hours=hours;self.out=out
  self.cache={};self.records=[];self.fitlog=[]
  starts=[issue-pd.DateOffset(months=6),issue-pd.DateOffset(months=3)]
  self.folds=[]
  for start,end in zip(starts,starts[1:]+[issue]):
   tr=y.index[y.label_end<start]
   va=y.index[(y.index>=start)&(y.index<end)&(y.label_end<issue)]
   assert len(tr)>4096 and len(va)>100
   self.folds.append((start,tr,va))
  self.keys=self.folds[0][1]
  self.vindex=self.folds[0][2].append(self.folds[1][2])
  self.truth=y.loc[self.vindex,'magnitude_log_bps']
 def evaluate(self,kind,cols):
  key=(kind,tuple(sorted(cols)))
  if key in self.cache:return self.cache[key]
  slug=hashlib.sha256(repr(key).encode()).hexdigest()[:16];preds=[];arts=[]
  for n,(start,tr,va) in enumerate(self.folds):
   path=self.out/f'{slug}_inner{n}.joblib'
   if path.exists():art=joblib.load(path)
   else:
    art=train(self.x[cols],self.y.magnitude_log_bps,tr,kind)
    art.update(issue=str(start),max_label_end=str(self.y.loc[tr,'label_end'].max()),horizon_hours=self.hours)
    assert self.y.loc[tr,'label_end'].max()<start
    joblib.dump(art,path,compress=3)
   self.fitlog.append(dict(file=path.name,kind=kind,columns=cols,issue=str(start),train_rows=len(tr),validation_rows=len(va)))
   preds.append(pd.Series(predict(art,self.x.loc[va]),index=va));arts.append(art)
  p=pd.concat(preds);result=dict(p=p,metrics=metrics(p,self.truth),arts=arts,cols=cols,kind=kind)
  self.cache[key]=result
  return result
 def gate(self,candidate,reference):
  p=candidate['p'];b=reference['p'];y=self.truth
  wins=0;icg=[]
  for month in sorted(set(y.index.strftime('%Y-%m'))):
   ix=y.index.strftime('%Y-%m')==month
   wins+=float(abs(p[ix]-y[ix]).mean())<float(abs(b[ix]-y[ix]).mean())
   icg.append(ic(p[ix],y[ix])-ic(b[ix],y[ix]))
  gain=reference['metrics']['mae']-candidate['metrics']['mae']
  ig=candidate['metrics']['rank_ic']-reference['metrics']['rank_ic']
  ok=gain>.01 and ig>.0001 and wins>=4 and np.mean(icg)>0
  return dict(pass_gate=bool(ok),mae_gain=gain,ic_gain=ig,monthly_mae_wins=int(wins),mean_monthly_ic_gain=float(np.mean(icg)))
 def permutation(self,result,column):
  impacts=[]
  for seed in [48,49,50]:
   ps=[]
   for art,(_,tr,va) in zip(result['arts'],self.folds):
    z=self.x.loc[va,result['cols']].copy()
    # Permute contiguous seven-day blocks; >= maximum tested120h horizon.
    blocks=[np.flatnonzero(((va-va[0]).total_seconds()//(7*86400)).astype(int)==b) for b in np.unique(((va-va[0]).total_seconds()//(7*86400)).astype(int))]
    rng=np.random.default_rng(seed);order=rng.permutation(len(blocks))
    idx=np.concatenate([blocks[i] for i in order])
    z[column]=z[column].to_numpy()[idx]
    ps.append(pd.Series(predict(art,z),index=va))
   m=metrics(pd.concat(ps),self.truth)
   impacts.append([m['mae']-result['metrics']['mae'],result['metrics']['rank_ic']-m['rank_ic']])
  med=np.median(impacts,axis=0)
  return dict(permutation_mae_increase=float(med[0]),permutation_ic_decrease=float(med[1]),permutation_pass=bool(med[0]>0 and med[1]>0),permutation_repeats=impacts)
 def forward(self,current,candidates,stage):
  for step in range(2):
   proposals=[]
   for c in candidates:
    if c in current['cols']:continue
    trial=self.evaluate(current['kind'],current['cols']+[c])
    record=dict(stage=stage,step=step,feature=c,kind=current['kind'],reference_columns=current['cols'],**trial['metrics'],**self.gate(trial,current))
    if record['pass_gate']:
     record.update(self.permutation(trial,c))
     if record['permutation_pass']:proposals.append((trial,record))
    self.records.append(record)
   if not proposals:break
   trial,record=min(proposals,key=lambda t:(t[0]['metrics']['mae'],-t[0]['metrics']['rank_ic']))
   record['selected']=True;current=trial
  return current

def run_horizon(raw,original_y,dev,report,r,hours):
 out=OUT/f'{hours}h';out.mkdir(exist_ok=True)
 if (out/'COMPLETE.json').exists():
  print(f'{hours}h already complete; reused',flush=True);return
 y=canonical.make_labels(r,raw.index,hours)
 if hours==48:
  assert y.index.equals(original_y.index)
  np.testing.assert_allclose(y.magnitude_log_bps,original_y.magnitude_log_bps,atol=1e-8,rtol=0)
 x=raw.loc[y.index].replace([np.inf,-np.inf],np.nan)
 pool=[c for c in report if family(c)!='other']
 issues=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 allp=[];allt=[];allv=[];selections=[];screens=[];logs=[];contributions=[]
 for n,issue in enumerate(issues):
  folder=out/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  if (folder/'DONE.json').exists():
   saved=json.loads((folder/'DONE.json').read_text())
   selections.append(saved)
   for file,collection in [('predictions.csv',allp),('train.csv',allt),('validation.csv',allv)]:
    q=pd.read_csv(folder/file,index_col=0,parse_dates=True);collection.append(q)
   screens+=json.loads((folder/'screen.json').read_text());contributions+=json.loads((folder/'contributions.json').read_text());logs+=json.loads((folder/'fits.json').read_text())
   continue
  search=FoldSearch(x,y,issue,hours,folder)
  shortlist,diagnostics=screen(x.loc[search.keys,pool],y.loc[search.keys,'magnitude_log_bps'],hours,6,'raw')
  _,controls=screen(x.loc[search.keys,PAIR],y.loc[search.keys,'magnitude_log_bps'],hours,2,'controls')
  for d in controls:d['role']='baseline_control_subject_to_final_drop_ablation'
  diagnostics+=controls
  # Baseline controls remain available; additions come only from magnitude tables.
  shortlist=[c for c in shortlist if c not in PAIR]
  for d in diagnostics:d.update(issue=str(issue),horizon_hours=hours)
  bases=[search.evaluate(kind,PAIR) for kind in CONFIGS]
  current=min(bases,key=lambda a:(a['metrics']['mae'],-a['metrics']['rank_ic']))
  base_kind=current['kind']
  raw_selected=search.forward(current,shortlist,'raw')
  parents=list(dict.fromkeys(raw_selected['cols']+shortlist))
  eng=engineered(x,parents)
  search.x=pd.concat([x,eng],axis=1)
  engshort,engdiag=screen(eng.loc[search.keys],y.loc[search.keys,'magnitude_log_bps'],hours,8,'engineered')
  for d in engdiag:d.update(issue=str(issue),horizon_hours=hours)
  current=search.forward(raw_selected,engshort,'engineered')
  # Genuine drop-column refits. Keep at least one liquidation-derived feature.
  for c in list(current['cols']):
   rest=[v for v in current['cols'] if v!=c]
   if not rest or not any(family(v)=='liquidation' for v in rest):continue
   trial=search.evaluate(current['kind'],rest)
   g=search.gate(trial,current)
   search.records.append(dict(stage='drop_ablation',feature=c,**g,selected=g['pass_gate']))
   if g['pass_gate']:current=trial
  finalcols=current['cols']
  tr=y.index[y.label_end<issue]
  end=issues[n+1] if n+1<len(issues) else dev.max()+pd.Timedelta(nanoseconds=1)
  te=dev.intersection(y.index);te=te[(te>=issue)&(te<end)]
  art=train(search.x[finalcols],y.magnitude_log_bps,tr,current['kind'])
  art.update(issue=str(issue),max_label_end=str(y.loc[tr,'label_end'].max()),horizon_hours=hours,engineering_parents=parents,source_sha256=canonical.EXPECTED_SHA256)
  joblib.dump(art,folder/'selected.joblib',compress=3)
  # Tree-only RF comparator trained on identical horizon/issuance.
  baseline=train(x[PAIR],y.magnitude_log_bps,tr,'rf_depth3')
  baseline.update(issue=str(issue),max_label_end=str(y.loc[tr,'label_end'].max()),horizon_hours=hours)
  joblib.dump(baseline,folder/'rf_baseline.joblib',compress=3)
  p=pd.DataFrame({'truth':y.loc[te,'magnitude_log_bps'],'prediction':predict(art,search.x.loc[te]),'rf_baseline':predict(baseline,x.loc[te]),'issue':str(issue)})
  t=pd.DataFrame({'truth':y.loc[tr,'magnitude_log_bps'],'prediction':predict(art,search.x.loc[tr]),'issue':str(issue)})
  v=pd.DataFrame({'truth':search.truth,'prediction':current['p'],'issue':str(issue)})
  p.to_csv(folder/'predictions.csv');t.to_csv(folder/'train.csv');v.to_csv(folder/'validation.csv')
  selection=dict(issue=str(issue),horizon_hours=hours,model=current['kind'],base_model=base_kind,columns=finalcols,raw_shortlist=shortlist,engineering_parents=parents,engineered_shortlist=engshort,validation_metrics=current['metrics'],train_metrics=metrics(t.prediction,t.truth),test_metrics=metrics(p.prediction,p.truth))
  # Scores never fed back into selection.
  dump(folder/'screen.json',diagnostics+engdiag);dump(folder/'contributions.json',search.records);dump(folder/'fits.json',search.fitlog);dump(folder/'DONE.json',selection)
  allp.append(p);allt.append(t);allv.append(v);selections.append(selection);screens+=diagnostics+engdiag;contributions+=search.records;logs+=search.fitlog
  print(f'{hours}h {issue.date()} complete; {len(search.fitlog)+2} fits, selected {current["kind"]}, {len(finalcols)} features',flush=True)
 p=pd.concat(allp);t=pd.concat(allt);v=pd.concat(allv)
 assert p.index.is_unique and p.index.equals(dev.intersection(y.index))
 if hours==48:
  ref=pd.read_csv(ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',index_col=0,parse_dates=True)
  np.testing.assert_allclose(p.truth,ref.loc[p.index,'truth'],atol=1e-8,rtol=0)
  p['report_hybrid']=ref.loc[p.index,'report_hybrid']
 scores=[]
 for split,frame,col in [('train',t,'prediction'),('validation',v,'prediction'),('test',p,'prediction'),('test_rf_baseline',p,'rf_baseline')]+([('test_report_hybrid',p,'report_hybrid')] if hours==48 else []):
  scores.append(dict(horizon_hours=hours,split=split,unique_timestamps=frame.index.nunique(),**metrics(frame[col],frame.truth)))
 monthly=[]
 for month,g in p.groupby(p.index.strftime('%Y-%m')):
  for col in [c for c in ['prediction','rf_baseline','report_hybrid'] if c in p]:
   monthly.append(dict(horizon_hours=hours,month=month,model=col,**metrics(g[col],g.truth)))
 p.to_csv(out/'predictions.csv');t.to_csv(out/'train.csv');v.to_csv(out/'validation.csv')
 pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False)
 pd.DataFrame(screens).to_csv(out/'feature_tests.csv',index=False);dump(out/'selections.json',selections);dump(out/'contributions.json',contributions);dump(out/'fits.json',logs)
 dump(out/'COMPLETE.json',dict(status='EXECUTED',horizon_hours=hours,total_fits=len(logs)+8,scores=scores,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(scores),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--horizons',type=int,nargs='+',default=[48,1,4,24,72,120]);args=p.parse_args()
 OUT.mkdir(exist_ok=True)
 raw,original_y,dev,report=old.load()
 r=canonical.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet')
 dump(OUT/'CANONICAL_VERIFICATION.json',canonical.validate(r,original_y))
 for hours in args.horizons:
  if hours not in [1,4,24,48,72,120]:raise ValueError('This registered run supports six horizons only')
  with threadpool_limits(limits=4):run_horizon(raw,original_y,dev,report,r,hours)
if __name__=='__main__':main()
