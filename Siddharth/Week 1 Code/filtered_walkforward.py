"""Filtered, horizon-aware full-period nested walk-forward research."""
from pathlib import Path
import sys,json,argparse,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nested_boost as nb
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,spearmanr
from threadpoolctl import threadpool_limits
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
ROOT=nb.ROOT
PREV=ROOT/'Experiments/nested_boost_48h_v1'
FAMILIES=['flow','volatility','funding']
CONFIGS=[dict(id=f'pair_{w}',kind='pair',window=w,pool='pair',transform='raw') for w in [3,6,12,'exp']]
for name,window,pool,transform,kind in [
 ('cat_pair_exp','exp','pair','raw','cat'),
 ('cat_flow_exp','exp','flow','raw','cat'),
 ('cat_volatility_exp','exp','volatility','raw','cat'),
 ('cat_funding_exp','exp','funding','raw','cat'),
 ('cat_core_exp','exp','core','raw','cat'),
 ('cat_core_12',12,'core','raw','cat'),
 ('cat_core_6',6,'core','raw','cat'),
 ('cat_robust_exp','exp','core','robust','cat'),
 ('cat_engineered_exp','exp','engineered','robust','cat'),
 ('cat_state_exp','exp','state','robust','cat'),
 ('cat_normalized_exp','exp','state','robust','normalized'),
 ('lgb_core_exp','exp','core','raw','lgb'),
 ('cat_no_flow_exp','exp','no_flow','raw','cat'),
 ('cat_no_volatility_exp','exp','no_volatility','raw','cat'),
 ('cat_no_funding_exp','exp','no_funding','raw','cat')]:
 CONFIGS.append(dict(id=name,window=window,pool=pool,transform=transform,kind=kind))
def keys_for(y,start,window):
 ok=y.label_end<start
 if window!='exp':ok&=y.index>=start-pd.DateOffset(months=int(window))
 return y.index[ok]
def ic(a,b):
 return float(spearmanr(a,b).statistic) if len(a)>20 and np.ptp(a)>0 and np.ptp(b)>0 else 0.
def effect(a,b):
 ok=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[ok];b=np.asarray(b)[ok]
 if len(a)<100 or np.unique(a).size<3:return 0.,0.,1.
 rho=ic(a,b);lo,hi=np.quantile(b,[.25,.75])
 ks=float(ks_2samp(a[b<=lo],a[b>=hi],method='asymp').statistic)
 cuts=np.unique(np.quantile(a,[.2,.4,.6,.8]));bucket=np.searchsorted(cuts,a)
 med=[float(np.median(b[bucket==j])) for j in range(len(cuts)+1) if (bucket==j).sum()>=20]
 lift=max(med)/max(min(med),1.) if len(med)>1 else 1.
 return rho,ks,lift
def screen(frame,target):
 records=[];months=frame.index.strftime('%Y-%m')
 recent=sorted(set(months))[-6:]
 for c in frame:
  a=frame[c].to_numpy();coverage=float(np.isfinite(a).mean())
  if np.isfinite(a).sum()<1000 or coverage<.7 or frame[c].nunique()<3:continue
  rho,ks,lift=effect(a,target.to_numpy());by=[]
  for m in recent:
   ix=months==m;by.append(effect(a[ix],target.to_numpy()[ix]))
  stable=np.mean([(np.sign(r)==np.sign(rho) and abs(r)>=.02) or (k>=.12 and l>=1.1) for r,k,l in by])
  passed=bool((abs(rho)>=.02 or (ks>=.12 and lift>=1.1)) and stable>=.6)
  records.append(dict(feature=c,coverage=coverage,rank_ic=rho,ks_stat=ks,magnitude_lift=lift,
   stable_fraction=float(stable),passed=passed,score=abs(rho)*stable+.25*ks+.1*np.log(max(lift,1.))))
 return pd.DataFrame(records,columns=['feature','coverage','rank_ic','ks_stat','magnitude_lift','stable_fraction','passed','score']).sort_values(['score','feature'],ascending=[False,True])
def choose_columns(train,ranking,limit):
 columns=[]
 for c in ranking.loc[ranking.passed,'feature']:
  if columns:
   cor=train[columns].corrwith(train[c],method='spearman').abs()
   if cor.max()>=.9:continue
  columns.append(c)
  if len(columns)>=limit:break
 return columns
def engineered(raw,parents):
 extra={}
 for c in parents:
  s=raw[c];roll=s.rolling('24h',closed='left',min_periods=24)
  extra['ABS__'+c]=s.abs()
  extra['DELTA1H__'+c]=s-s.shift(freq='1h').reindex(s.index)
  extra['MEAN24H__'+c]=roll.mean()
  extra['INNOV24H__'+c]=s-roll.mean()
 return pd.DataFrame(extra,index=raw.index)
def matrix(frame,prep):
 z=frame[prep['columns']];miss=z.isna().to_numpy(dtype=float)
 v=z.fillna(prep['median'])
 if prep['transform']=='robust':
  v=v.clip(prep['lo'],prep['hi'],axis=1)
  v=(v-prep['median'])/prep['iqr']
  v=np.sign(v)*np.log1p(abs(v))
 else:v=(v-prep['center'])/prep['scale']
 return np.column_stack([v.to_numpy(dtype=float),miss])
def prediction(a,raw,y,keys):
 if a.get('reuse_nb'):return nb.predict(a,raw.loc[keys])
 frame=raw
 if a['parents']:frame=frame.join(engineered(raw,a['parents']))
 if a['history']:frame=frame.join(nb.old.histories(y,raw.index))
 p=a['model'].predict(matrix(frame.loc[keys],a['prep']))
 if a['normalized']:
  p=p*np.maximum(20.,frame.loc[keys,'HIST__median_30d'].fillna(a['scale_fill']).to_numpy())
 return np.maximum(0,p)
class Runner:
 def __init__(self,raw,y,out,hours,prices):
  self.raw=raw;self.y=y;self.out=out;self.hours=hours;self.prices=prices
  self.cache={};self.ledger=[];self.filters={};self.baselines={}
 def get_filter(self,start,window):
  key=(str(start),str(window))
  if key in self.filters:return self.filters[key]
  tk=keys_for(self.y,start,window);z=self.raw.loc[tk];target=self.y.loc[tk,'magnitude_log_bps']
  families={};reports=[]
  for fam in FAMILIES:
   cols=[c for c in z if c.startswith('F4Z__'+fam+'__')]
   ranking=screen(z[cols],target);ranking['family']=fam;reports.append(ranking)
   families[fam]=choose_columns(z,ranking,3)
  ranking=pd.concat(reports);ranking.to_csv(self.out/'filters'/f'{start:%Y-%m}__{window}.csv',index=False)
  self.filters[key]=families
  return families
 def fit(self,cfg,start):
  name=cfg['id'];tag=f'{start:%Y-%m}__{name}'
  if tag in self.cache:return self.cache[tag]
  tk=keys_for(self.y,start,cfg['window'])
  assert len(tk)>=1000 and self.y.loc[tk,'label_end'].max()<start
  # Reuse only exactly matching immutable baseline fits from the previous run.
  if self.hours==48 and not self.prices and name in ['pair_exp','reference_rf']:
   kind='pair' if name=='pair_exp' else 'rf'
   paths=list((PREV/'models').glob(f'{start:%Y-%m}__report_*_{kind}__expanding.joblib'))
   for path in paths:
    a=joblib.load(path)
    if a['train_keys'].equals(tk):
     a['reuse_nb']=True
     joblib.dump(a,self.out/'models'/f'{tag}.joblib',compress=3)
     self.cache[tag]=a
     self.ledger.append(dict(id=tag,status='REUSED',source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_rows=len(tk)))
     nb.dump(self.out/'fit_ledger.json',self.ledger)
     return a
  cols=list(nb.PAIR);parents=[];history=False;frame=self.raw
  if cfg['pool']!='pair':
   families=self.get_filter(start,cfg['window'])
   for fam in FAMILIES:
    if cfg['pool'] in [fam,'core','engineered','state'] or (cfg['pool'].startswith('no_') and cfg['pool']!='no_'+fam):
     cols+=families[fam]
   if cfg['pool']=='engineered':
    parents=cols[len(nb.PAIR):];eng=engineered(self.raw,parents)
    rank=screen(eng.loc[tk],self.y.loc[tk,'magnitude_log_bps'])
    rank.to_csv(self.out/'filters'/f'{start:%Y-%m}__engineered.csv',index=False)
    additions=choose_columns(eng.loc[tk],rank,6)
    frame=frame.join(eng);cols+=additions
   if cfg['pool']=='state':
    history=True;frame=frame.join(nb.old.histories(self.y,self.raw.index))
    hist=['HIST__median_7d','HIST__median_30d','HIST__mean_30d']
    rank=screen(frame.loc[tk,hist],self.y.loc[tk,'magnitude_log_bps'])
    rank.to_csv(self.out/'filters'/f'{start:%Y-%m}__history.csv',index=False)
    cols+=choose_columns(frame.loc[tk],rank,2)
  z=frame.loc[tk,cols];prep=nb.prepare(z,cols,standardize=cfg['kind'] in ['pair','rf'])
  prep.update(transform=cfg['transform'],lo=z.quantile(.01),hi=z.quantile(.99),iqr=(z.quantile(.75)-z.quantile(.25)).clip(lower=1e-4))
  if cfg['kind']=='pair':model=QuantileRegressor(quantile=.5,alpha=0,solver='highs')
  elif cfg['kind']=='rf':model=RandomForestRegressor(n_estimators=128,criterion='absolute_error',max_depth=3,min_samples_leaf=128,max_features=1.,bootstrap=True,max_samples=4096,n_jobs=4,random_state=48)
  elif cfg['kind']=='lgb':
   model=LGBMRegressor(objective='regression_l1',n_estimators=160,learning_rate=.025,num_leaves=4,max_depth=2,
    min_child_samples=1024,reg_alpha=10.,reg_lambda=50.,colsample_bytree=1.,random_state=48,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
  else:
   model=CatBoostRegressor(loss_function='MAE',iterations=160,depth=2,learning_rate=.025,l2_leaf_reg=50,
    random_seed=48,thread_count=4,verbose=False,allow_writing_files=False,has_time=True)
  response=self.y.loc[tk,'magnitude_log_bps'];normalized=cfg['kind']=='normalized';scale_fill=None;weights=None
  if normalized:
   scale_fill=float(frame.loc[tk,'HIST__median_30d'].median())
   scale=np.maximum(20.,frame.loc[tk,'HIST__median_30d'].fillna(scale_fill).to_numpy())
   response=response/scale;weights=scale/np.mean(scale)
  a=dict(model=model,prep=prep,train_keys=tk,issuance=start,config=cfg,parents=parents,history=history,normalized=normalized,scale_fill=scale_fill)
  record=dict(id=tag,status='STARTED',train_rows=len(tk),train_start=str(tk.min()),train_end=str(tk.max()),max_label_end=str(self.y.loc[tk,'label_end'].max()),columns=cols,config=cfg,parameters=model.get_params())
  if sum(r['status']!='REUSED' for r in self.ledger)>=140:raise RuntimeError('Fit cap 140 exhausted')
  self.ledger.append(record);nb.dump(self.out/'fit_ledger.json',self.ledger)
  print('FIT',len(self.ledger),tag,len(tk),len(cols),flush=True);tick=time.monotonic()
  try:
   with threadpool_limits(limits=4):
    if weights is None:model.fit(matrix(frame.loc[tk],prep),response)
    else:model.fit(matrix(frame.loc[tk],prep),response,sample_weight=weights)
   record.update(status='PASS',seconds=time.monotonic()-tick);joblib.dump(a,self.out/'models'/f'{tag}.joblib',compress=3)
   self.cache[tag]=a;nb.dump(self.out/'fit_ledger.json',self.ledger);return a
  except BaseException as e:
   record.update(status='FAILED',error=repr(e));nb.dump(self.out/'fit_ledger.json',self.ledger);raise
 def base(self,start,keys):
  if str(start) not in self.baselines:
   inner=start-pd.DateOffset(months=3)
   vk=self.y.index[(self.y.index>=inner)&(self.y.index<start)&(self.y.label_end<start)]
   cfgpair=CONFIGS[3];cfgrf=dict(id='reference_rf',kind='rf',window='exp',pool='pair',transform='raw')
   ip=self.fit(cfgpair,inner);ir=self.fit(cfgrf,inner)
   pp=prediction(ip,self.raw,self.y,vk);pr=prediction(ir,self.raw,self.y,vk);truth=self.y.loc[vk,'magnitude_log_bps']
   weight=min([0.,.1,.25,.5],key=lambda w:(float(np.mean(abs((1-w)*pp+w*pr-truth))),w))
   self.baselines[str(start)]=(self.fit(cfgpair,start),self.fit(cfgrf,start),weight)
   nb.dump(self.out/'baseline'/f'{start:%Y-%m}.json',dict(start=str(start),inner_start=str(inner),inner_rows=len(vk),
    max_inner_label_end=str(self.y.loc[vk,'label_end'].max()),rf_weight=weight))
  ap,ar,w=self.baselines[str(start)]
  return (1-w)*prediction(ap,self.raw,self.y,keys)+w*prediction(ar,self.raw,self.y,keys)

def select(val):
 target=val.truth;baseline=nb.metrics(val.baseline,target)
 monthly=val.index.strftime('%Y-%m');candidates={};records=[]
 for cfg in CONFIGS:
  name=cfg['id']
  for w in [.1,.25,.5,1.]:
   tag=name+f'@{w}'
   p=(1-w)*val.baseline+w*val[name];m=nb.metrics(p,target)
   wins=0;ic_changes=[]
   for month in sorted(set(monthly)):
    ix=monthly==month;a=nb.metrics(p[ix],target[ix]);b=nb.metrics(val.baseline[ix],target[ix])
    wins+=a['mae']<b['mae']
    ic_changes.append(nb.rankscore(a)-nb.rankscore(b))
   eligible=bool(m['mae']<baseline['mae'] and nb.rankscore(m)>nb.rankscore(baseline) and wins>=4 and np.mean(ic_changes)>0)
   record=dict(recipe=tag,model=name,weight=w,eligible=eligible,monthly_mae_wins=int(wins),
    monthly_mean_ic_gain=float(np.mean(ic_changes)),**m)
   records.append(record);candidates[tag]=record
 good=[v for v in records if v['eligible']]
 best=min(good,key=lambda z:(z['mae'],-nb.rankscore(z),z['weight'],z['recipe'])) if good else dict(recipe='baseline',model=None,weight=0.)
 standalone=min([v for v in records if v['weight']==1.],key=lambda z:(z['mae'],-nb.rankscore(z),z['recipe']))
 return dict(selected=best,best_standalone=standalone,baseline_metrics=baseline,candidates=records)
def contribution(val):
 rows=[]
 comparisons=[('cat_pair_exp','cat_'+f+'_exp','add_'+f) for f in FAMILIES]
 comparisons += [('cat_no_'+f+'_exp','cat_core_exp','incremental_'+f) for f in FAMILIES]
 comparisons += [('cat_core_exp','cat_robust_exp','robust_transform'),('cat_robust_exp','cat_engineered_exp','causal_engineering'),
  ('cat_robust_exp','cat_state_exp','matured_history'),('cat_state_exp','cat_normalized_exp','target_scale_normalization')]
 for before,after,label in comparisons:
  a=nb.metrics(val[before],val.truth);b=nb.metrics(val[after],val.truth)
  rows.append(dict(change=label,before=before,after=after,mae_improvement=a['mae']-b['mae'],
   rank_ic_improvement=nb.rankscore(b)-nb.rankscore(a),improves_both=b['mae']<a['mae'] and nb.rankscore(b)>nb.rankscore(a)))
 return rows
def preflight(raw,y,months,out,hours):
 cut=months[0];parents=[c for c in raw if c.split('__')[1] in FAMILIES][:9]
 pd.testing.assert_frame_equal(engineered(raw,parents).loc[:cut],engineered(raw.loc[:cut],parents))
 changed=y.copy();changed.loc[changed.label_end>=cut,'magnitude_log_bps']+=100000.
 ix=raw.index[raw.index<=cut]
 np.testing.assert_allclose(nb.old.histories(y,ix),nb.old.histories(changed,ix),rtol=0,atol=0,equal_nan=True)
 idx=pd.date_range('2020-01-01',periods=300,freq='30min',tz='UTC');price=pd.Series(np.exp(np.arange(300)*.0001),index=idx)
 for h in [1,4,24,48,72,120]:
  lab=nb.make_labels(price,idx,h);np.testing.assert_allclose(lab.magnitude_log_bps,2*h,atol=1e-9)
 assert (y.label_end==y.index+pd.Timedelta(hours=hours)).all()
 nb.dump(out/'PREFLIGHT.json',dict(status='PASS',causal_engineering=True,future_label_perturbation=True,horizon_math=[1,4,24,48,72,120]))
def run(hours,prices,out):
 raw,y,dev,_=nb.load_data(hours,prices)
 starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 if dev.min()!=starts[0]:raise ValueError('This registered full-period study requires evaluation starting January 2024.')
 out.mkdir(parents=True,exist_ok=False)
 for d in ['models','filters','inner','outer','baseline']: (out/d).mkdir()
 contract=dict(created_utc=str(pd.Timestamp.now(tz='UTC')),horizon_hours=hours,prices=str(prices) if prices else None,
  configs=CONFIGS,fit_cap=140,outer_starts=list(map(str,starts)),test_rows=len(dev),
  protocol='Quarterly outer windows Jan/Apr/Jul/Oct2024; final window ends at last supplied origin. Two preceding quarterly inner folds, six months total, labels mature strictly before outer issuance.',
  history='3/6/12 month and expanding; authorized change from prior three-month-only rule.',
  screening='Within each training population: >=1000 finite, >=70% coverage, >=3 unique; |IC|>=.02 OR KS>=.12 and quintile magnitude lift>=1.1; effect stability in >=60% of latest six months. KS is descriptive effect size, not an IID significance test.',
  feature_cap='3 per flow/volatility/funding, absolute Spearman redundancy <.90; 6 extra engineered features; 2 admitted matured-history features. Original two liquidation inputs retained.',
  transformation='Training-only 1/99% clipping, median/IQR scaling, signed log1p. Causal abs, 1h delta, previous24h mean and innovation.',
  normalized_loss='Predict target / known matured30d median, sample weight proportional to scale; equivalent raw-unit MAE weighting.',
  selector='Require pooled MAE and IC improvement over reference, >=4/6 monthly MAE wins, positive mean monthly IC gain; then minimum MAE. Else retain reference.',
  weights=[.1,.25,.5,1.],baseline='Reconstructed report policy, quarterly expanding pair/RF and preceding3mo weight selection; cached fits reused only on exact populations.',
  stop='One registered bounded run, no outer-score-driven changes. Exposed historical research, not untouched confirmation.',
  fullperiod_mae_target=181.5238678328609,other_horizons='Require canonical price CSV; each horizon gets separate labels and model fits.')
 nb.dump(out/'CONTRACT.json',contract)
 (out/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 nb.dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'Code/nested_boost.py',ROOT/'Code/nested_ic_rf.py',nb.old.PACKAGE/'MANIFEST.json']})
 nb.dump(out/'environment.json',dict(python=sys.version,catboost=__import__('catboost').__version__,lightgbm=__import__('lightgbm').__version__,numpy=np.__version__,pandas=pd.__version__))
 preflight(raw,y,starts,out,hours);r=Runner(raw,y,out,hours,prices)
 tests=[];trains=[];validations=[];receipts=[];contributions=[]
 for outer_no,outer in enumerate(starts):
  inner=[]
  for start in [outer-pd.DateOffset(months=6),outer-pd.DateOffset(months=3)]:
   keys=y.index[(y.index>=start)&(y.index<start+pd.DateOffset(months=3))&(y.label_end<outer)]
   frame=pd.DataFrame(dict(truth=y.loc[keys,'magnitude_log_bps']),index=keys)
   frame['baseline']=r.base(start,keys)
   for cfg in CONFIGS:frame[cfg['id']]=prediction(r.fit(cfg,start),raw,y,keys)
   inner.append(frame)
  val=pd.concat(inner).sort_index();val.to_csv(out/'inner'/f'{outer:%Y-%m}.csv',index_label='decision_time')
  choice=select(val);choice.update(outer=str(outer),inner_rows=len(val),latest_inner_label_end=str(y.loc[val.index,'label_end'].max()))
  nb.dump(out/'outer'/f'{outer:%Y-%m}_selection.json',choice)
  contributions += [dict(outer=str(outer),**v) for v in contribution(val)]
  print('SELECTED',str(outer),choice['selected']['recipe'],flush=True)
  end=starts[outer_no+1] if outer_no+1<len(starts) else dev.max()+pd.Timedelta(minutes=30)
  keys=dev[(dev>=outer)&(dev<end)]
  test=pd.DataFrame(dict(truth=y.loc[keys,'magnitude_log_bps']),index=keys);test['baseline']=r.base(outer,keys)
  for cfg in CONFIGS:test[cfg['id']]=prediction(r.fit(cfg,outer),raw,y,keys)
  for frame in [test,val]:
   chosen=choice['selected'];frame['selected']=frame.baseline if chosen['model'] is None else (1-chosen['weight'])*frame.baseline+chosen['weight']*frame[chosen['model']]
   frame['standalone_selected']=frame[choice['best_standalone']['model']]
   frame['outer']=str(outer)
  # Training scores for the selected forecast use the intersection of its component training populations.
  selected=choice['selected'];cfg=next((c for c in CONFIGS if c['id']==selected['model']),CONFIGS[3])
  tk=keys_for(y,outer,cfg['window'])
  train=pd.DataFrame(dict(truth=y.loc[tk,'magnitude_log_bps']),index=tk);train['baseline']=r.base(outer,tk)
  train['selected']=train.baseline if selected['model'] is None else (1-selected['weight'])*train.baseline+selected['weight']*prediction(r.fit(cfg,outer),raw,y,tk)
  train['outer']=str(outer)
  test.to_csv(out/'outer'/f'{outer:%Y-%m}_predictions.csv',index_label='decision_time')
  tests.append(test);trains.append(train);validations.append(val)
 p=pd.concat(tests).sort_index();training=pd.concat(trains);validation=pd.concat(validations)
 assert p.index.equals(dev) and p.index.is_unique and p.select_dtypes('number').notna().all().all()
 p.to_csv(out/'predictions.csv',index_label='decision_time');training.to_csv(out/'training_predictions.csv',index_label='decision_time')
 validation.to_csv(out/'validation_predictions.csv',index_label='decision_time')
 pd.DataFrame(contributions).to_csv(out/'validation_contributions.csv',index=False)
 scores=[];monthly=[]
 for split,frame in [('train',training),('validation_selected',validation),('test',p)]:
  for name in [c for c in frame.select_dtypes('number') if c!='truth']:
   scores.append(dict(split=split,model=name,**nb.metrics(frame[name],frame.truth)))
 for month,f in p.groupby(p.index.strftime('%Y-%m')):
  for name in ['selected','baseline','standalone_selected']:
   monthly.append(dict(month=month,model=name,**nb.metrics(f[name],f.truth)))
 pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False)
 uncertainty=nb.uncertainty(p.rename(columns={'baseline':'report_hybrid'}))
 nb.dump(out/'uncertainty.json',uncertainty)
 result=dict(status='EXECUTED',scores=scores,test_rows=len(p),new_fits=sum(v['status']=='PASS' for v in r.ledger),reused_fits=sum(v['status']=='REUSED' for v in r.ledger),
  training_model_rows=len(training),training_unique_rows=training.index.nunique(),validation_model_rows=len(validation),validation_unique_rows=validation.index.nunique(),
  beats_mae_number=nb.metrics(p.selected,p.truth)['mae']<181.5238678328609,
  beats_reconstructed_both=nb.metrics(p.selected,p.truth)['mae']<nb.metrics(p.baseline,p.truth)['mae'] and nb.metrics(p.selected,p.truth)['rank_ic']>nb.metrics(p.baseline,p.truth)['rank_ic'],
  interpretation='Primary procedure selected inside nested validation. Individual fixed candidate scores are exploratory diagnostics, not a retrospectively selected winner. Other horizons untrained.')
 nb.dump(out/'RESULT.json',result)
 print('FINAL TEST\n'+pd.DataFrame(scores).query("split=='test'").sort_values('mae').to_string(index=False),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--horizon-hours',type=int,default=48);ap.add_argument('--prices',type=Path)
 ap.add_argument('--out',type=Path,default=ROOT/'Experiments/filtered_walkforward_v1')
 args=ap.parse_args();run(args.horizon_hours,args.prices,args.out)
