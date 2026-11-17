"""Controlled extension of the reconstructed pair/RF hybrid. No fits on import."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np,pandas as pd,joblib
from scipy.stats import spearmanr,pearsonr,ks_2samp
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import nested_ic_rf as old
import canonical_return_features as crf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/targeted_feature_extension_v1'
BASE=ROOT/'Experiments/nested_boost_48h_v1'
PAIR=[f'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__{w}__value' for w in ['15m','5m']]
FAMILIES=['liquidation','flow','volatility','basis','funding','price']
PARAMS=dict(n_estimators=128,criterion='absolute_error',max_depth=3,min_samples_leaf=128,max_features=1.,bootstrap=True,max_samples=4096,n_jobs=8,random_state=48)
def dump(path,obj):
 path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def metric(p,y):
 p=np.asarray(p);y=np.asarray(y)
 assert len(p)==len(y) and np.isfinite(p).all() and np.isfinite(y).all()
 return dict(mae=float(abs(p-y).mean()),rank_ic=float(spearmanr(np.round(p,10),y).statistic) if np.ptp(p)>0 else 0.,pearson_ic=float(pearsonr(p,y).statistic) if np.ptp(p)>0 else 0.,rows=len(p))
def family(c):
 if c.startswith('PRICE__'):return 'price'
 return next((f for f in FAMILIES if '__'+f+'__' in c),'other')
def price_features(r,index):
 d={};cum=r.fillna(0).cumsum()
 for h in [1,4,24,72,168]:
  roll=r.rolling(f'{h}h',closed='left',min_periods=h*12)
  net=roll.sum();vol=np.sqrt((r*r).rolling(f'{h}h',closed='left',min_periods=h*12).sum())
  d[f'PRICE__net_return_{h}h']=net.reindex(index)
  d[f'PRICE__realized_vol_{h}h']=vol.reindex(index)
  d[f'PRICE__mean_abs_return_{h}h']=r.abs().rolling(f'{h}h',closed='left',min_periods=h*12).mean().reindex(index)
  d[f'PRICE__trend_strength_{h}h']=(net/vol.clip(lower=1e-8)).reindex(index)
  current=cum.shift(1)
  top=cum.rolling(f'{h}h',closed='left',min_periods=h*12).max()
  bottom=cum.rolling(f'{h}h',closed='left',min_periods=h*12).min()
  d[f'PRICE__drawdown_{h}h']=(current-top).where(net.notna()).reindex(index)
  d[f'PRICE__range_{h}h']=(top-bottom).where(net.notna()).reindex(index)
 return pd.DataFrame(d,index=index).astype('float32')
def engineering(x,parents):
 d={}
 for c in parents:
  s=x[c]
  for h in [1,24]:
   d[f'DELTA{h}H__{c}']=s-s.shift(freq=pd.Timedelta(hours=h)).reindex(x.index)
  mean=s.rolling('24h',closed='left',min_periods=24).mean()
  sd=s.rolling('24h',closed='left',min_periods=24).std()
  d[f'Z24H__{c}']=((s-mean)/sd.where(sd>1e-8)).clip(-10,10)
  d[f'ABS__{c}']=s.abs()
  d[f'PAIR_PRODUCT__{c}']=s*x[PAIR[0]]
 return pd.DataFrame(d,index=x.index).replace([np.inf,-np.inf],np.nan).astype('float32')
def prep(x,cols):
 a=x[cols];med=a.median().fillna(0);filled=a.fillna(med)
 return dict(columns=cols,median=med,center=filled.mean(),scale=filled.std(ddof=0).replace(0,1).fillna(1))
def design(x,p):
 a=x[p['columns']];v=(a.fillna(p['median'])-p['center'])/p['scale']
 return np.column_stack([v.to_numpy(dtype=float),a.isna().to_numpy(dtype=float)])
def predict(a,x):
 model=a['model']
 if isinstance(model,RandomForestRegressor):model.n_jobs=1
 return np.maximum(0,model.predict(design(x,a['prep'])))
def fit(x,y,keys,issue,cols,path):
 assert y.loc[keys,'label_end'].max()<issue
 if path.exists():
  try:
   saved=joblib.load(path)
   assert saved['train_keys'].equals(keys) and saved['prep']['columns']==cols and pd.Timestamp(saved['issuance'])==issue
   for k,v in PARAMS.items():
    if k!='n_jobs':assert saved['model'].get_params()[k]==v
   print('REUSE',path.name,flush=True)
   return saved
  except Exception:
   path.rename(path.with_suffix('.interrupted_'+str(int(time.time()))))
 p=prep(x.loc[keys],cols);model=RandomForestRegressor(**PARAMS)
 begin=time.monotonic();model.fit(design(x.loc[keys],p),y.loc[keys,'magnitude_log_bps'])
 a=dict(model=model,prep=p,train_keys=keys,issuance=issue,max_label_end=str(y.loc[keys,'label_end'].max()))
 joblib.dump(a,path,compress=3)
 print('FIT',path.name,len(keys),len(cols),round(time.monotonic()-begin,1),flush=True)
 return a
def screen(x,y,keys,columns):
 yy=y.loc[keys,'magnitude_log_bps'];threshold=yy.quantile(.9);rows=[]
 for c in columns:
  a=x.loc[keys,c];good=a.notna();n=int(good.sum());count=a.nunique()
  if n<1000 or good.mean()<.5 or count<3:
   rows.append(dict(feature=c,family=family(c),finite=n,missing_fraction=float(1-good.mean()),unique=int(count),quality=False));continue
  v=a[good];t=yy[good];s=float(spearmanr(v,t).statistic)
  lo=v[t<threshold];hi=v[t>=threshold]
  ks=float(ks_2samp(lo,hi).statistic) if len(lo)>0 and len(hi)>0 else 0.
  low,high=v.quantile([.1,.9]);event=t>=threshold;rate=float(event.mean())
  lift=max(float(event[v<=low].mean()),float(event[v>=high].mean()))/max(rate,1e-12)
  # MI estimation sample is taken exclusively from the earlier training population.
  rng=np.random.default_rng(48);take=np.sort(rng.choice(len(v),min(4096,len(v)),replace=False))
  mi=float(mutual_info_regression(v.to_numpy()[take,None],t.to_numpy()[take],random_state=48,n_neighbors=5)[0])
  rows.append(dict(feature=c,family=family(c),finite=n,missing_fraction=float(1-good.mean()),unique=int(count),variance=float(v.var()),spearman=s,mutual_information=mi,ks=ks,lift=lift,quality=True))
 stats=pd.DataFrame(rows)
 good=stats[stats.quality].copy()
 good['score']=sum(good[k].abs().rank(pct=True) if k=='spearman' else good[k].rank(pct=True) for k in ['spearman','mutual_information','ks','lift'])/4
 good=good.sort_values(['score','feature'],ascending=[False,True])
 stats=stats.merge(good[['feature','score']],on='feature',how='left')
 return list(good.feature),stats
def prune(x,keys,ranked,k):
 selected=list(PAIR);rejected=[]
 for c in ranked:
  if c in selected:continue
  z=x.loc[keys,selected+[c]].rank().corr().abs()
  if z.loc[c,selected].max()>.98:rejected.append(dict(feature=c,reason='rank_correlation_over_0.98'));continue
  selected.append(c)
  if len(selected)>=k+2:break
 # VIF is a conservative redundancy filter; protected original pair is never removed.
 while len(selected)>2:
  a=x.loc[keys,selected];a=a.fillna(a.median()).fillna(0)
  a=(a-a.mean())/a.std(ddof=0).replace(0,1)
  arr=a.to_numpy();vifs=[]
  for j in range(2,len(selected)):
   target=arr[:,j];other=np.delete(arr,j,axis=1)
   resid=target-other@np.linalg.lstsq(other,target,rcond=None)[0]
   vif=float((target@target)/max(float(resid@resid),1e-12));vifs.append((vif,j))
  v,j=max(vifs)
  if v<=20:break
  rejected.append(dict(feature=selected[j],reason='VIF_over_20',vif=v));selected.pop(j)
 return selected,rejected
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'COMPLETE.json').exists():raise RuntimeError('Completed run must not be overwritten')
 raw,y,dev,report=old.load();raw=raw.replace([np.inf,-np.inf],np.nan)
 r=crf.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet')
 prices=price_features(r,raw.index)
 # Causal feature checks before fitting.
 cutoff=raw.index[len(raw)//2]
 pd.testing.assert_frame_equal(price_features(r.loc[:cutoff],raw.index[raw.index<=cutoff]),prices.loc[:cutoff])
 changed=r.copy();changed.loc[changed.index>=cutoff]+=100000
 np.testing.assert_allclose(price_features(changed,pd.DatetimeIndex([cutoff])),prices.loc[[cutoff]],rtol=0,atol=0,equal_nan=True)
 x=pd.concat([raw,prices],axis=1);candidate=list(dict.fromkeys(a['column'] for a in json.loads((OUT/'USER_FEATURE_MAPPING.json').read_text())))
 assert len(candidate)==38 and set(candidate).issubset(raw.columns)
 dump(OUT/'DATA.json',dict(rows=len(x),test_rows=len(dev),candidate_counts={f:sum(family(c)==f for c in candidate) for f in FAMILIES},price_source='canonical past 5m returns, no raw dollar price',causality='PASS'))
 reference=pd.read_csv(BASE/'report_baseline_predictions.csv',index_col=0,parse_dates=True)
 selections=[];trials=[];fits=[];preds=[];valpred=[];trainpred=[];importance=[]
 starts=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
 for issue in starts:
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  inner=issue-pd.DateOffset(months=3)
  tk=y.index[y.label_end<inner]
  vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)]
  ot=y.index[y.label_end<issue]
  end=issue+pd.DateOffset(months=3) if issue!=starts[-1] else dev.max()+pd.Timedelta(minutes=30)
  te=dev[(dev>=issue)&(dev<end)]
  ranked,stats=screen(x,y,tk,candidate);stats.to_csv(folder/'feature_tests.csv',index=False)
  packs={};pruning=[]
  for k in [4,8,16,32]:
   cols,reject=prune(x,tk,ranked,k);packs[f'pooled_{k}']=cols;pruning.extend(reject)
  # Family-balanced pool prevents a larger source family monopolizing the shortlist.
  balanced=[]
  for f in FAMILIES:balanced += [c for c in ranked if family(c)==f][:3]
  packs['family_balanced'],reject=prune(x,tk,balanced,18);pruning.extend(reject)
  parents=list(dict.fromkeys(packs['pooled_8'][2:]+packs['family_balanced'][2:]))
  eng=engineering(x,parents)
  pd.testing.assert_frame_equal(engineering(x.loc[:cutoff],parents),eng.loc[:cutoff])
  frame=pd.concat([x,eng],axis=1)
  erank,estats=screen(frame,y,tk,list(eng));estats.to_csv(folder/'engineered_feature_tests.csv',index=False)
  packs['engineered_8'],reject=prune(frame,tk,erank,8);pruning.extend(reject)
  packs['raw_plus_engineered'],reject=prune(frame,tk,list(dict.fromkeys(packs['pooled_8'][2:]+erank)),16);pruning.extend(reject)
  dump(folder/'pruning.json',pruning);dump(folder/'feature_packs.json',packs)
  inner_pair=joblib.load(BASE/'models'/f'{inner:%Y-%m}__report_inner_pair__expanding.joblib')
  inner_rf=joblib.load(BASE/'models'/f'{inner:%Y-%m}__report_inner_rf__expanding.joblib')
  assert inner_pair['train_keys'].equals(tk) and inner_rf['train_keys'].equals(tk)
  pp=predict(inner_pair,frame.loc[vk]);rr=predict(inner_rf,frame.loc[vk]);truth=y.loc[vk,'magnitude_log_bps']
  bw=min([0.,.1,.25,.5],key=lambda w:(metric((1-w)*pp+w*rr,truth)['mae'],w))
  baseline=(1-bw)*pp+bw*rr;bm=metric(baseline,truth)
  cache={};best=None;validation_candidates=[]
  for name,cols in packs.items():
   if len(cols)<=2:continue
   key=tuple(cols)
   if key not in cache:
    path=folder/(name+'_inner.joblib');a=fit(frame,y,tk,inner,cols,path)
    rp=predict(a,frame.loc[vk]);cache[key]=(a,rp,path)
    fits.append(dict(path=str(path),phase='inner',issue=str(issue),features=cols,train_rows=len(tk)))
   a,rp,path=cache[key]
   for w in [.1,.25,.5]:
    pred=(1-w)*pp+w*rp;met=metric(pred,truth)
    ablation=metric((1-w)*pp+w*rr,truth)
    wins=sum(metric(pred[vk.strftime('%Y-%m')==month],truth[vk.strftime('%Y-%m')==month])['mae']<metric(baseline[vk.strftime('%Y-%m')==month],truth[vk.strftime('%Y-%m')==month])['mae']-.01 for month in sorted(set(vk.strftime('%Y-%m'))))
    passed=met['mae']<bm['mae']-.01 and met['rank_ic']>bm['rank_ic']+.0001 and wins>=2 and met['mae']<ablation['mae']-.01
    rec=dict(issue=str(issue),pack=name,weight=w,features=cols,artifact=str(path),**met,baseline=bm,group_ablation=ablation,monthly_mae_wins=wins,passed=passed)
    trials.append(rec);validation_candidates.append(rec)
    if passed and (best is None or (met['mae'],-met['rank_ic'],len(cols),name,w)<(best['mae'],-best['rank_ic'],len(best['features']),best['pack'],best['weight'])):best=rec
  # Block-permutation is diagnostic; no tuning on outer data.
  if best:
   a,rp,path=cache[tuple(best['features'])]
   A=frame.loc[vk].copy();rng=np.random.default_rng(48)
   for col in best['features'][2:]:
    losses=[]
    for shift in [336,672,1008]:
     z=A.copy();z[col]=np.roll(z[col].to_numpy(),shift)
     pm=(1-best['weight'])*pp+best['weight']*predict(a,z)
     losses.append(metric(pm,truth)['mae']-best['mae'])
    importance.append(dict(issue=str(issue),feature=col,validation_mae_increase=float(np.mean(losses)),method='circular block shifts 7/14/21 days; diagnostic'))
  outer_pair=joblib.load(BASE/'models'/f'{issue:%Y-%m}__report_outer_pair__expanding.joblib')
  outer_rf=joblib.load(BASE/'models'/f'{issue:%Y-%m}__report_outer_rf__expanding.joblib')
  assert outer_pair['train_keys'].equals(ot) and outer_rf['train_keys'].equals(ot)
  if best:
   cols=best['features'];w=best['weight'];name=best['pack']
   path=folder/'selected_outer.joblib';oa=fit(frame,y,ot,issue,cols,path)
   fits.append(dict(path=str(path),phase='outer',issue=str(issue),features=cols,train_rows=len(ot)))
   va=(1-w)*pp+w*cache[tuple(cols)][1]
  else:
   cols=PAIR;w=bw;name='original_pair_rf';oa=outer_rf;va=baseline;path=BASE/'models'/f'{issue:%Y-%m}__report_outer_rf__expanding.joblib'
  base_test=(1-bw)*predict(outer_pair,frame.loc[te])+bw*predict(outer_rf,frame.loc[te])
  np.testing.assert_allclose(base_test,reference.loc[te,'report_hybrid'],rtol=0,atol=1e-7)
  pred=(1-w)*predict(outer_pair,frame.loc[te])+w*predict(oa,frame.loc[te])
  trpred=(1-w)*predict(outer_pair,frame.loc[ot])+w*predict(oa,frame.loc[ot])
  preds.append(pd.DataFrame(dict(prediction=pred,baseline=base_test,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te))
  valpred.append(pd.DataFrame(dict(prediction=va,baseline=baseline,truth=truth,issue=str(issue)),index=vk))
  trainpred.append(pd.DataFrame(dict(prediction=trpred,truth=y.loc[ot,'magnitude_log_bps'],issue=str(issue)),index=ot))
  rec=dict(issue=str(issue),inner=str(inner),selected=name,weight=w,baseline_weight=bw,features=cols,engineering_parents=parents,artifact=str(path),pair_artifact=str(BASE/'models'/f'{issue:%Y-%m}__report_outer_pair__expanding.joblib'),train_rows=len(ot),inner_train_rows=len(tk),validation_rows=len(vk),test_rows=len(te),test=metric(pred,y.loc[te,'magnitude_log_bps']),baseline_test=metric(base_test,y.loc[te,'magnitude_log_bps']))
  selections.append(rec);dump(folder/'selection.json',rec);dump(OUT/'FIT_LEDGER.json',fits)
  print('WINDOW COMPLETE',issue.date(),name,'selected extra features',len(cols)-2,flush=True)
 p=pd.concat(preds);v=pd.concat(valpred);t=pd.concat(trainpred)
 assert p.index.equals(dev)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time');t.to_csv(OUT/'training_predictions.csv',index_label='decision_time')
 dump(OUT/'SELECTIONS.json',selections);dump(OUT/'VALIDATION_TRIALS.json',trials);dump(OUT/'PERMUTATION.json',importance)
 scores=[dict(split='training',**metric(t.prediction,t.truth),unique_rows=t.index.nunique()),dict(split='validation',**metric(v.prediction,v.truth),unique_rows=v.index.nunique()),dict(split='test',**metric(p.prediction,p.truth)),dict(split='test_baseline',**metric(p.baseline,p.truth))]
 last=p.loc[p.index>=pd.Timestamp('2024-11-01',tz='UTC')]
 scores += [dict(split='last3months',**metric(last.prediction,last.truth)),dict(split='last3months_baseline',**metric(last.baseline,last.truth))]
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 dump(OUT/'COMPLETE.json',dict(status='EXECUTED',scores=scores,new_fits=len(fits),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),meets_under180_and_ic=scores[2]['mae']<180 and scores[2]['rank_ic']>scores[3]['rank_ic'],beats_both=scores[2]['mae']<181.5238678328609 and scores[2]['rank_ic']>scores[3]['rank_ic'],limitation='Retrospective exposed historical evaluation; no fresh holdout. Training rows repeat across expanding fits.'))
 (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 print(pd.DataFrame(scores).to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
