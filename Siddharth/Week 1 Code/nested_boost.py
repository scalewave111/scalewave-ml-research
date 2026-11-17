"""Horizon-aware nested walk-forward magnitude experiment; no fits on import."""
from pathlib import Path
import argparse, hashlib, json, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.boost_deps'))
sys.path.insert(0,str(ROOT/'Code'))
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import nested_ic_rf as old
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
PAIR=[f'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__{w}__value' for w in ['15m','5m']]
CONFIGS=[dict(id=a+'_'+pool,algorithm=a,pool=pool,iterations=350,depth=3 if pool=='pair' else 4)
         for a in ['cat','lgb'] for pool in ['pair','report','ic24']]
def dump(path,obj):
 path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metrics(p,y):
 p=np.asarray(p); y=np.asarray(y)
 assert len(p)==len(y) and np.isfinite(p).all() and np.isfinite(y).all()
 return dict(mae=float(abs(p-y).mean()),rmse=float(np.sqrt(np.mean((p-y)**2))),
  rank_ic=float(spearmanr(p,y).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else None,
  pearson_ic=float(pearsonr(p,y).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else None,rows=len(y))
def rankscore(m): return m['rank_ic'] if m['rank_ic'] is not None else -2.
def make_labels(price,index,hours):
 if hours<=0: raise ValueError('Horizon must be positive')
 price=price.sort_index()
 assert price.index.is_unique and (price.dropna()>0).all()
 end=index+pd.Timedelta(hours=hours)
 p0=price.reindex(index).to_numpy(); p1=price.reindex(end).to_numpy()
 signed=10000*np.log(p1/p0)
 return pd.DataFrame(dict(label_end=end,signed_return_log_bps=signed,magnitude_log_bps=abs(signed)),index=index).dropna()
def load_data(hours,prices):
 x,y,dev,report=old.load()
 if hours!=48 or prices:
  if not prices: raise ValueError('Non-48h runs require --prices CSV with decision_time and price columns; never reuse 48h labels.')
  f=pd.read_csv(prices); f['decision_time']=pd.to_datetime(f.decision_time,utc=True)
  y=make_labels(f.set_index('decision_time').price,x.index,hours)
  x=x.loc[y.index]; dev=dev.intersection(y.index)
 return x,y,dev,report
def features(raw,report):
 extra={}
 for c in report:
  for hours in [1,24]:
   extra[f'DELTA{hours}H__'+c]=raw[c]-raw[c].shift(freq=pd.Timedelta(hours=hours)).reindex(raw.index)
 return pd.concat([raw,pd.DataFrame(extra,index=raw.index)],axis=1).replace([np.inf,-np.inf],np.nan)
def train_keys(y,start):
 return y.index[(y.index>=start-pd.DateOffset(months=3))&(y.label_end<start)]
def design(x,prep):
 a=x[prep['columns']]; v=(a.fillna(prep['median'])-prep['center'])/prep['scale']
 return np.column_stack([v.to_numpy(dtype=float),a.isna().to_numpy(dtype=float)])
def prepare(x,columns,standardize=False):
 z=x[columns]; med=z.median(); filled=z.fillna(med)
 return dict(columns=columns,median=med,center=filled.mean() if standardize else pd.Series(0.,index=columns),
  scale=filled.std(ddof=0).replace(0,1.) if standardize else pd.Series(1.,index=columns))
def predict(a,x):
 return np.maximum(0,a['model'].predict(design(x,a['prep'])))
class Runner:
 def __init__(self,x,y,report,out):
  self.x=x;self.y=y;self.report=report;self.out=out;self.ledger=[];self.cache={};self.rankings={}
 def fit(self,name,start,keys=None,expanding=False,report_rf=False):
  tk=keys if keys is not None else train_keys(self.y,start)
  tag=f'{start:%Y-%m}__{name}'+('__expanding' if expanding else '')
  if tag in self.cache:return self.cache[tag]
  assert len(tk)>1000 and self.y.loc[tk,'label_end'].max()<start
  cfg=next((c for c in CONFIGS if c['id']==name),None)
  if cfg is None or cfg['pool']=='pair': columns=PAIR
  elif cfg['pool']=='report':
   columns=[c for c in self.report if self.x.loc[tk,c].notna().sum()>=1000 and self.x.loc[tk,c].nunique()>1]
  else:
   if str(start) not in self.rankings:
    r=old.screen(self.x.loc[tk],self.y.loc[tk,'magnitude_log_bps'])
    self.rankings[str(start)]=r
    r.to_csv(self.out/'ic'/f'{start:%Y-%m}.csv',index=False)
   ranked=list(self.rankings[str(start)].feature); columns=list(PAIR)
   pool=ranked[:96]; corr=self.x.loc[tk,list(dict.fromkeys(PAIR+pool))].rank().corr().abs()
   for c in pool:
    if c not in columns and corr.loc[c,columns].max()<.97: columns.append(c)
    if len(columns)>=24:break
  prep=prepare(self.x.loc[tk],columns,standardize=cfg is None)
  if report_rf:
   model=RandomForestRegressor(n_estimators=128,criterion='absolute_error',max_depth=3,min_samples_leaf=128,
     max_features=1.,bootstrap=True,max_samples=4096,n_jobs=4,random_state=48)
  elif cfg is None:model=QuantileRegressor(quantile=.5,alpha=0,solver='highs')
  elif cfg['algorithm']=='cat':
   model=CatBoostRegressor(loss_function='MAE',iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=.03,
    l2_leaf_reg=20,random_seed=48,thread_count=4,verbose=False,allow_writing_files=False,has_time=True)
  else:
   model=LGBMRegressor(objective='regression_l1',n_estimators=cfg['iterations'],learning_rate=.03,
    num_leaves=7 if cfg['pool']=='pair' else 12,max_depth=cfg['depth'],min_child_samples=192,
    reg_alpha=5.,reg_lambda=20.,colsample_bytree=1. if cfg['pool']=='pair' else .8,
    random_state=48,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
  rec=dict(id=tag,status='STARTED',train_rows=len(tk),train_start=str(tk.min()),train_end=str(tk.max()),
   max_label_end=str(self.y.loc[tk,'label_end'].max()),issuance=str(start),columns=columns,parameters=model.get_params())
  if len(self.ledger)>=70:raise RuntimeError('70 fit cap reached')
  self.ledger.append(rec);dump(self.out/'fit_ledger.json',self.ledger)
  print('FIT',len(self.ledger),tag,len(tk),len(columns),flush=True); tick=time.monotonic()
  try:
   with threadpool_limits(limits=4):model.fit(design(self.x.loc[tk],prep),self.y.loc[tk,'magnitude_log_bps'])
   a=dict(model=model,prep=prep,train_keys=tk,issuance=start,tag=tag)
   joblib.dump(a,self.out/'models'/f'{tag}.joblib',compress=3)
   rec.update(status='PASS',seconds=time.monotonic()-tick);dump(self.out/'fit_ledger.json',self.ledger)
   self.cache[tag]=a
   return a
  except BaseException as e:
   rec.update(status='FAILED',error=repr(e));dump(self.out/'fit_ledger.json',self.ledger);raise
def report_baseline(r,dev):
 pieces=[];cards=[]
 starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 for i,s in enumerate(starts):
  inner=s-pd.DateOffset(months=3)
  it=r.y.index[r.y.label_end<inner]
  vk=r.y.index[(r.y.index>=inner)&(r.y.index<s)&(r.y.label_end<s)]
  ap=r.fit('report_inner_pair',inner,it,True)
  ar=r.fit('report_inner_rf',inner,it,True,True)
  pp=predict(ap,r.x.loc[vk]);pr=predict(ar,r.x.loc[vk]);truth=r.y.loc[vk,'magnitude_log_bps']
  w=min([0.,.1,.25,.5],key=lambda w:(float(np.mean(abs((1-w)*pp+w*pr-truth))),w))
  tk=r.y.index[r.y.label_end<s];end=starts[i+1] if i+1<len(starts) else dev.max()+pd.Timedelta(minutes=30)
  keys=dev[(dev>=s)&(dev<end)]
  ap=r.fit('report_outer_pair',s,tk,True)
  ar=r.fit('report_outer_rf',s,tk,True,True)
  pp=predict(ap,r.x.loc[keys]);pr=predict(ar,r.x.loc[keys])
  pieces.append(pd.DataFrame(dict(report_pair=pp,report_rf=pr,report_hybrid=(1-w)*pp+w*pr),index=keys))
  cards.append(dict(start=str(s),weight=w,train_rows=len(tk),inner_train_rows=len(it),inner_validation_rows=len(vk),test_rows=len(keys)))
 p=pd.concat(pieces);p['truth']=r.y.loc[p.index,'magnitude_log_bps']
 scores={c:metrics(p[c],p.truth) for c in ['report_pair','report_rf','report_hybrid']}
 delta=abs(scores['report_hybrid']['mae']-181.5238678328609)
 dump(r.out/'report_baseline_reproduction.json',dict(cards=cards,scores=scores,reference_mae=181.5238678328609,
   mae_difference=delta,aggregate_matches_report=delta<.001,
   limitation='Recipe reconstructed from PDF; original prediction hashes unavailable; not exact artifact replay.'))
 p.to_csv(r.out/'report_baseline_predictions.csv',index_label='decision_time')
 return p
def select(val):
 scores={c:metrics(val[c],val.truth) for c in val if c!='truth'}
 base=scores['rolling_pair']
 eligible=[c for c in scores if c!='rolling_pair' and scores[c]['mae']<base['mae'] and rankscore(scores[c])>rankscore(base)]
 selected=min(eligible,key=lambda c:(scores[c]['mae'],-rankscore(scores[c]),c)) if eligible else 'rolling_pair'
 per_algo={}
 for algo in ['cat','lgb']:
  names=[c['id'] for c in CONFIGS if c['algorithm']==algo]
  good=[n for n in names if n in eligible]
  per_algo[algo]=min(good or names,key=lambda c:(scores[c]['mae'],-rankscore(scores[c]),c))
 return dict(selected=selected,eligible=eligible,best_cat=per_algo['cat'],best_lgb=per_algo['lgb'],scores=scores)
def blends(frame):
 for cfg in CONFIGS:
  for w in [.25,.5]:
   frame[cfg['id']+f'_blend{w}']=(1-w)*frame.rolling_pair+w*frame[cfg['id']]
 return frame

def verify(out,x,y,dev):
 saved=old.read_csv(out/'predictions.csv'); errors=[]
 assert saved.index.is_unique
 for path in sorted((out/'outer').glob('*_selection.json')):
  s=json.loads(path.read_text());start=pd.Timestamp(s['outer'])
  keys=old.month_keys(saved.index,start);tk=train_keys(y,start)
  val=old.read_csv(out/'inner'/f'{start:%Y-%m}.csv')
  assert (y.loc[val.index,'label_end']<start).all()
  chosen=select(val)
  assert all(chosen[k]==s[k] for k in ['selected','best_cat','best_lgb','eligible'])
  raw=pd.DataFrame(index=keys)
  for name in ['rolling_pair']+[c['id'] for c in CONFIGS]:
   a=joblib.load(out/'models'/f'{start:%Y-%m}__{name}.joblib')
   assert a['train_keys'].equals(tk) and y.loc[tk,'label_end'].max()<start
   np.testing.assert_allclose(a['prep']['median'],x.loc[tk,a['prep']['columns']].median(),rtol=0,atol=0)
   raw[name]=predict(a,x.loc[keys])
  raw=blends(raw)
  for target,name in [('selected',s['selected']),('catboost',s['best_cat']),('lightgbm',s['best_lgb']),('rolling_pair','rolling_pair')]:
   err=float(np.max(abs(raw[name]-saved.loc[keys,target])));assert err<1e-8;errors.append(err)
 # Independent arithmetic, saved scores and unique coverage.
 for row in pd.read_csv(out/'scores.csv').to_dict('records'):
  if row['split']=='test':
   m=metrics(saved[row['model']],saved.truth)
   assert abs(m['mae']-row['mae'])<1e-8
 dump(out/'VERIFICATION.json',dict(status='PASS',model_replay=True,purges=True,training_medians=True,
  selection_recomputed=True,independent_score_arithmetic=True,max_replay_error=max(errors),new_fits=0))
def uncertainty(pred):
 records=[]
 n=len(pred);days=14
 groups=((pred.index-pred.index.min()).total_seconds()//(days*86400)).astype(int)
 blocks=[np.flatnonzero(groups==g) for g in sorted(set(groups))]
 rng=np.random.default_rng(62)
 for baseline in [c for c in ['report_hybrid','previous_rf','rolling_pair'] if c in pred]:
  dm=[];di=[]
  for _ in range(1000):
   ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
   a=metrics(pred.selected.iloc[ix],pred.truth.iloc[ix]);b=metrics(pred[baseline].iloc[ix],pred.truth.iloc[ix])
   dm.append(a['mae']-b['mae'])
   if a['rank_ic'] is not None and b['rank_ic'] is not None:di.append(a['rank_ic']-b['rank_ic'])
  records.append(dict(baseline=baseline,block_days=days,mae_difference_95=list(np.quantile(dm,[.025,.975])),
   rank_ic_difference_95=list(np.quantile(di,[.025,.975])) if di else None))
 return records
def run(hours,prices,out):
 raw,y,dev,report=load_data(hours,prices);x=features(raw,report)
 if len(dev)==0:raise ValueError('No evaluation labels available')
 last=dev.max().normalize().replace(day=1);months=pd.date_range(last-pd.DateOffset(months=2),last,freq='MS')
 out.mkdir(parents=True,exist_ok=False)
 for d in ['models','inner','outer','ic']: (out/d).mkdir()
 dump(out/'CONTRACT.json',dict(horizon_hours=hours,configs=CONFIGS,fit_cap=70,outer_months=list(map(str,months)),
  training='Trailing 3 calendar months, label_end strictly before issuance; monthly refits.',
  validation='Three preceding monthly walk-forward folds, each same training rule; validation labels mature before outer issuance.',
  selection='Candidates require lower pooled MAE AND higher pooled Spearman than rolling pair. Lowest MAE eligible wins; otherwise rolling pair. Per-family standalone finalists are reported even if not eligible.',
  blends=[.25,.5],calibration='None',search='Six fixed recipes, 350 iterations, no test-driven retuning or early stopping.',
  exposure='Previously exposed development; exploratory, not fresh confirmation.',
  report_comparator='Separately reconstructed expanding-history quarterly hybrid, never used for tuning.',
  multi_horizon='Separate fitted model and purges per horizon; non48h requires canonical price CSV.',
  validation_reporting='Selected inner validation is selection-biased and overlapping across outer decisions.',
  training_reporting='In-sample model-row observations, repeated timestamps across overlapping windows.',
  price_source=str(prices) if prices else None))
 (out/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 dump(out/'environment.json',dict(python=sys.version,catboost=__import__('catboost').__version__,
  lightgbm=__import__('lightgbm').__version__,numpy=np.__version__,pandas=pd.__version__))
 dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
  [Path(__file__),ROOT/'Code/nested_ic_rf.py',old.PACKAGE/'MANIFEST.json']+([Path(prices)] if prices else [])})
 # Verify exact timestamp target construction and causal feature prefix before fitting.
 toyidx=pd.date_range('2020-01-01',periods=300,freq='30min',tz='UTC')
 toy=pd.Series(np.exp(np.arange(300)*.0001),index=toyidx)
 for h in [1,4,24,48,72,120]:
  lab=make_labels(toy,toyidx,h)
  np.testing.assert_allclose(lab.magnitude_log_bps,h*2,atol=1e-9)
  assert (lab.label_end==lab.index+pd.Timedelta(hours=h)).all()
 cut=months[0]
 pd.testing.assert_frame_equal(x.loc[:cut],features(raw.loc[:cut],report))
 dump(out/'PREFLIGHT.json',dict(status='PASS',horizon_target_tests=[1,4,24,48,72,120],causal_prefix=True,rows=len(y)))
 r=Runner(x,y,report,out);tests=[];trains=[];validations=[];monthly=[]
 for outer in months:
  inner=[]
  for start in pd.date_range(outer-pd.DateOffset(months=3),outer-pd.DateOffset(months=1),freq='MS'):
   keys=old.month_keys(x.index,start);keys=keys[y.loc[keys,'label_end']<outer]
   block=pd.DataFrame({'truth':y.loc[keys,'magnitude_log_bps']},index=keys)
   for name in ['rolling_pair']+[c['id'] for c in CONFIGS]:
    block[name]=predict(r.fit(name,start),x.loc[keys])
   inner.append(blends(block))
  val=pd.concat(inner).sort_index();s=select(val);s['outer']=str(outer)
  dump(out/'outer'/f'{outer:%Y-%m}_selection.json',s)
  val.to_csv(out/'inner'/f'{outer:%Y-%m}.csv',index_label='decision_time')
  print('SELECT',outer,s['selected'],s['best_cat'],s['best_lgb'],flush=True)
  keys=old.month_keys(dev,outer);tk=train_keys(y,outer)
  test=pd.DataFrame(index=keys);train=pd.DataFrame(index=tk)
  for name in ['rolling_pair']+[c['id'] for c in CONFIGS]:
   a=r.fit(name,outer);test[name]=predict(a,x.loc[keys]);train[name]=predict(a,x.loc[tk])
  test=blends(test);train=blends(train)
  for frame,split,target in [(test,'test',y.loc[keys,'magnitude_log_bps']),(train,'train',y.loc[tk,'magnitude_log_bps']),(val,'validation',val.truth)]:
   result=pd.DataFrame(index=frame.index)
   for targetname,name in [('selected',s['selected']),('catboost',s['best_cat']),('lightgbm',s['best_lgb']),('rolling_pair','rolling_pair')]:
    result[targetname]=frame[name]
    monthly.append(dict(month=f'{outer:%Y-%m}',split=split,model=targetname,**metrics(frame[name],target)))
   result['truth']=target;result['outer_month']=f'{outer:%Y-%m}'
   (tests if split=='test' else trains if split=='train' else validations).append(result)
 p=pd.concat(tests).sort_index();training=pd.concat(trains);validation=pd.concat(validations)
 assert p.index.equals(dev[dev>=months[0]]) and p.index.is_unique
 if hours==48 and not prices:
  reference=report_baseline(r,dev)
  p=p.join(reference[['report_hybrid','report_pair','report_rf']])
  previous=old.read_csv(old.prior.OUT/'predictions.csv')
  p['previous_rf']=previous.loc[p.index,'magnitude_log_bps']
  p['previous_hybrid']=old.read_csv(old.PACKAGE/'submissions/member_02/rf_walk_forward_v2/predictions.csv').loc[p.index,'magnitude_log_bps']
  p['failed_nested_rf']=old.read_csv(ROOT/'Experiments/nested_ic_rf_v3/predictions.csv').loc[p.index,'nested_selected']
  p['frozen_pair']=old.prior.frozen_predictions(raw.loc[p.index]).magnitude_log_bps
 for frame in [p,training,validation]:assert frame.select_dtypes('number').notna().all().all()
 p.to_csv(out/'predictions.csv',index_label='decision_time')
 training.to_csv(out/'training_predictions.csv',index_label='decision_time')
 validation.to_csv(out/'validation_predictions.csv',index_label='decision_time')
 scores=[]
 for split,frame in [('train',training),('validation',validation),('test',p)]:
  for name in frame.select_dtypes('number'):
   if name!='truth':scores.append(dict(split=split,model=name,**metrics(frame[name],frame.truth)))
 pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
 # Include baseline monthly scores as well.
 for month,frame in p.groupby('outer_month'):
  for name in [c for c in p.select_dtypes('number') if c not in ['truth','selected','catboost','lightgbm','rolling_pair']]:
   monthly.append(dict(month=month,split='test',model=name,**metrics(frame[name],frame.truth)))
 pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False)
 dump(out/'uncertainty.json',uncertainty(p))
 dump(out/'RESULT.json',dict(status='EXECUTED',horizon_hours=hours,test_rows=len(p),scores=scores,
  fits=len(r.ledger),training_unique_timestamps=training.index.nunique(),training_model_rows=len(training),
  validation_unique_timestamps=validation.index.nunique(),validation_model_rows=len(validation),
  other_horizons='Pipeline implemented and target math tested; not trained without price source.'))
 verify(out,x,y,dev)
 html='<html><head><meta charset="utf-8"><title>Nested boosting results</title></head><body><h1>48h nested walk-forward boosting</h1><p>Exploratory exposed data. Training and validation include repeated timestamps across outer decisions. Validation was used for selection. Each test timestamp appears once.</p>'+pd.DataFrame(scores).to_html(index=False,float_format=lambda v:f'{v:.6f}')+'<h2>Monthly scores</h2>'+pd.DataFrame(monthly).to_html(index=False,float_format=lambda v:f'{v:.6f}')+'</body></html>'
 html=html.replace('48h nested walk-forward boosting',str(hours)+'h nested walk-forward boosting')
 (out/'RESULTS.html').write_text(html,encoding='utf-8')
 print('FINAL\n'+pd.DataFrame(scores).to_string(index=False),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--horizon-hours',type=int,default=48)
 parser.add_argument('--prices',type=Path)
 parser.add_argument('--out',type=Path,default=ROOT/'Experiments/nested_boost_48h_v1')
 args=parser.parse_args();run(args.horizon_hours,args.prices,args.out)
