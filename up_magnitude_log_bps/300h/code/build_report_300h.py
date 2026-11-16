"""Audited 300h PDF report from saved predictions and model metadata; no inference."""
from pathlib import Path
import json, hashlib, re, shutil, zipfile, joblib, ast
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'Code/experiments'
OUT=ROOT/'Report'
SUP=OUT/'300h_report_support_2026-11-14'
PDF=OUT/'BTC_300h_Models_Features_Rankings_2026-11-14.pdf'
RUNS={
 'A':('Original F4Z search','excursion_pipeline_300h_v1/20261113T115754520122Z/search/20261113T115801555917Z'),
 'B':('Price-block follow-up','excursion_pipeline_300h_v1/20261113T115754520122Z/price_followup/20261113T121655587383Z'),
 'C':('Long-window feature packs','excursion_long_features_300h_v1/20261113T123508595159Z'),
 'D':('Compact long-feature search','compact_features_300h_v1/20261113T125036278413Z'),
 'E':('Fast joint MAE/RMSE search','magnitude_models_300h_fast_v1/20261114T162602551164Z'),
 'F':('UP MAE-only refinement','mae_up_300h_v1/20261114T163703375349Z'),
 'G':('Both-target MAE refinement','mae_both_300h_v1/20261114T165116421962Z')}
PRIMARY=dict(A='prediction',B='selected',C='selected',D='new_features',E='selected',F='selected',G='selected')
SUP.mkdir(parents=True,exist_ok=True)
hashes={};data={};cards={};contracts={};audit=[];metrics_rows=[];policies=[];rank_inputs=[];partial=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p):hashes[str(p.relative_to(ROOT))]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(track(p))
def side(t):return t.split('_')[0].upper()
def metrics(y,p):
 e=np.asarray(y)-np.asarray(p)
 return dict(MAE=float(np.abs(e).mean()),MedianAE=float(np.median(np.abs(e))),RMSE=float(np.sqrt(np.square(e).mean())),
             IC=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if np.ptp(p)>0 and np.ptp(y)>0 else float('nan'),Bias=float(-e.mean()))
for label,(name,rel) in RUNS.items():
 path=EXP/rel;done=js(path/'COMPLETE.json')
 assert done['status']=='COMPLETE' and sha(path/'predictions.csv')==done['predictions_sha256']
 f=csv(path/'predictions.csv').sort_values(['target','decision_time']).reset_index(drop=True)
 assert not f.duplicated(['target','decision_time']).any() and f.groupby('target').size().eq(17473).all()
 if data:
  base=data['A'][data['A'].target.isin(f.target.unique())].reset_index(drop=True)
  assert f[['target','decision_time','issue']].equals(base[['target','decision_time','issue']])
  for c in ['actual','training_median','training_mean']:np.testing.assert_allclose(f[c],base[c],rtol=0,atol=1e-8)
 data[label]=f;cards[label]=js(path/'selections.json');contracts[label]=js(path/'CONTRACT.json')
 for fn in ['COMPLETE.json','CONTRACT.json','selections.json']:shutil.copyfile(path/fn,SUP/(label+'_'+fn))
 for p in path.glob('*.py'):track(p)
 savedpath=path/('metrics_all_variants.csv' if label=='B' else 'metrics.csv')
 saved=csv(savedpath)
 for target,g in f.groupby('target'):
  cols=[c for c in g.select_dtypes('number').columns if c!='actual']
  for col in cols:
   for period,part in [('ALL',g)]+list(g.groupby('period')):
    sub=part[np.isfinite(part[col])]
    if not len(sub):continue
    values=metrics(sub.actual.to_numpy(),sub[col].to_numpy())
    metrics_rows.append(dict(run=label,target=side(target),forecast=col,period=period,rows=len(sub),full_coverage=len(sub)==len(part),**values))
    sr=saved[(saved.target==target)&(saved.period==period)&(saved.model==col)]
    if len(sr)==1:
     for low,up in [('mae','MAE'),('rmse','RMSE'),('ic','IC')]:
      if low in sr:np.testing.assert_allclose(sr.iloc[0][low],values[up],rtol=1e-8,atol=1e-8,equal_nan=True)
   ispolicy=col==PRIMARY[label] or (label in 'EFG' and col=='candidate')
   if ispolicy:
    role='diagnostic candidate' if col=='candidate' else 'selected policy'
    policies.append(dict(run=label,target=side(target),forecast=col,role=role,**metrics(g.actual.to_numpy(),g[col].to_numpy())))
   if ispolicy or label=='B' or (label=='A' and col in ['training_mean','training_median']):
    if col.startswith('training_') and label!='A':continue
    role='baseline' if col.startswith('training_') else ('diagnostic candidate' if col=='candidate' else 'selected policy' if ispolicy else 'fixed diagnostic')
    if not np.isfinite(g[col]).all():partial.append(dict(run=label,target=side(target),forecast=col,rows=int(np.isfinite(g[col]).sum())));continue
    rank_inputs.append((label,side(target),col,role,g[col].to_numpy(),g.actual.to_numpy()))
 audit.append(dict(run=label,name=name,path=rel,predictions_sha256=done['predictions_sha256'],rows=len(f),replay_reported=done.get('replay_passed',False)))
for target,g in data['A'].groupby('target'):
 for col in ['training_mean','training_median']:policies.append(dict(run='A',target=side(target),forecast=col,role='baseline',**metrics(g.actual.to_numpy(),g[col].to_numpy())))
pol=pd.DataFrame(policies).sort_values(['target','MAE'])
for t in ['UP','DOWN']:pol.loc[pol.target.eq(t),'MAE_rank']=pol.loc[pol.target.eq(t),'MAE'].rank(method='min')
allm=pd.DataFrame(metrics_rows)
allm.to_csv(SUP/'all_forecasts_period_metrics.csv',index=False);pol.to_csv(SUP/'policy_rankings.csv',index=False)
full=[]
for t in ['UP','DOWN']:
 seen=[]
 for label,target,col,role,v,y in rank_inputs:
  if target!=t:continue
  match=next((row for old,row in seen if np.allclose(old,v,atol=1e-7,rtol=0)),None)
  if match is not None:match['aliases']+='; '+label+'.'+col;continue
  row=dict(run=label,target=t,forecast=col,role=role,aliases=label+'.'+col,**metrics(y,v))
  full.append(row);seen.append((v,row))
full=pd.DataFrame(full).sort_values(['target','MAE'])
for t in ['UP','DOWN']:
 ix=full.target.eq(t)
 for m,asc in [('MAE',True),('MedianAE',True),('RMSE',True),('IC',False)]:full.loc[ix,m+'_rank']=full.loc[ix,m].rank(method='min',ascending=asc)
full.to_csv(SUP/'all_full_coverage_rankings.csv',index=False);pd.DataFrame(partial).to_csv(SUP/'partial_variants.csv',index=False)
assert full[full.target.eq('UP')].iloc[0]['aliases'].startswith('G.candidate')
assert full[full.target.eq('DOWN')].iloc[0]['aliases'].startswith('G.selected')

# Read saved models as metadata only; do not fit or call predict.
model_hashes=[];models=[];packs={};pack_features={};best=[]
def pack(cols):
 key=tuple(cols)
 if key not in packs:packs[key]=f'P{len(packs)+1:03d}';pack_features[packs[key]]=list(cols)
 return packs[key]
def load_model(p,digest):
 assert sha(p)==digest
 model_hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=digest));track(p)
 return joblib.load(p)
def describe(m,cols,spec,label,target,issue,branch):
 cfg={};params=0
 if 'constant' in m:kind='Median constant';cfg=dict(constant=m['constant']);cols=[]
 elif 'state' in m:
  cfg=dict(m['config']);kind=cfg['kind'].upper()+str(cfg['hidden'])
  params=sum(np.asarray(v).size for v in m['state'].values())
  cfg.update(loss=m.get('loss','absolute_error'),epochs=contracts[label]['epochs'],batch=512,seed=48,gradient_clip=1,
             estimator_rows=m['estimator_training_rows'],cap=contracts[label]['estimator_row_caps']['neural'],parameters=params,
             raw_per_step=len(cols)//4 if cfg['kind']=='gru' else len(cols),channels_per_step=len(cols)//2 if cfg['kind']=='gru' else 2*len(cols))
 else:
  kind=type(m['model']).__name__;cfg=m['model'].get_params()
  cfg.update(estimator_rows=m.get('estimator_training_rows','all mature training rows'))
 row=dict(run=label,target=side(target),issue=issue[:7],forecast=branch,architecture=kind,pack=pack(cols),features=cols,
          specification=spec,hyperparameters=cfg,parameters=params)
 models.append(row)
 return row
for label,(_,rel) in RUNS.items():
 path=EXP/rel
 for s in cards[label]:
  target,issue=s['target'],s['issue'];folder=path/(target+'/'+issue[:7] if label in 'EFG' else issue[:7]+'_'+target)
  if label=='B':
   card=js(folder/'selection.json')
   for variant,spec in card['variants'].items():
    if not spec.get('artifact'):continue
    m=load_model(folder/spec['artifact'],spec['artifact_sha256'])
    describe(m['model'],m['features'],dict(spec,selected=variant==card['selected_variant']),label,target,issue,variant)
  else:
   m=load_model(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256'])
   if label in 'AD':describe(m['model'],m['features'],s,label,target,issue,PRIMARY[label])
   elif label=='C':
    spec=m['selection']
    describe(m['candidate'] if spec is not None else dict(constant=m['median']),spec['features'] if spec else [],spec or {},label,target,issue,'selected')
   else:
    for branch in ['selected','candidate']:
     row=describe(m[branch],m[branch]['features'],m['plan'][branch],label,target,issue,branch)
     if label=='G' and ((target.startswith('up') and branch=='candidate') or (target.startswith('down') and branch=='selected')):best.append(row)

base_names=sorted(set(re.sub(r'^SEQ\d+__','',c) for cols in pack_features.values() for c in cols))
fid={c:f'F{i+1:03d}' for i,c in enumerate(base_names)}
(SUP/'model_details.json').write_text(json.dumps(models,indent=2,default=str))
(SUP/'feature_packs.json').write_text(json.dumps(pack_features,indent=2))
pd.DataFrame([dict(id=fid[c],feature=c) for c in base_names]).to_csv(SUP/'feature_ids.csv',index=False)
trials=[]
for label,(_,rel) in RUNS.items():
 path=EXP/rel
 for q in path.rglob('selection.json'):
  dest=SUP/'cards'/label/q.relative_to(path);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(track(q),dest)
 for q in path.rglob('validation_trials.json'):
  a=js(q)
  if isinstance(a,list):
   a=pd.DataFrame(a);a.insert(0,'run',label);a.insert(1,'folder',str(q.parent.relative_to(path)));trials.append(a)
 for q in path.rglob('validation_trials.csv'):
  a=csv(q);a.insert(0,'run',label);a.insert(1,'folder',str(q.parent.relative_to(path)));trials.append(a)
 for q in path.glob('*dictionary*.csv'):shutil.copyfile(track(q),SUP/(label+'_'+q.name))
if trials:pd.concat(trials,ignore_index=True).to_csv(SUP/'validation_trials.csv',index=False)
definitions=csv(EXP/RUNS['G'][1]/'feature_dictionary.csv').fillna('')
def formula(c):
 a=definitions[definitions.feature.eq(c)]
 if len(a):return str(a.iloc[0].get('formula',''))
 if c.startswith('F4Z__'):return 'Inherited masked source measurement; exact source family/measurement/timeframe in name. Upstream percentile or robust-Z construction not independently reconstructed.'
 if c.startswith('LONG300__'):
  field=c.split('__')[1]
  fm=dict(net='N=sum past 5m returns',trend_to_vol='N/max(V,1e-6)',efficiency='N/max(sum abs(r),1e-6)',range_position='(C-L)/max(H-L,1e-6)',drawdown_from_high='H-C',rebound_from_low='C-L',range_to_vol='(H-L)/max(V,1e-6)',rv='V=sqrt(sum r^2)',up_rv='sqrt(sum max(r,0)^2)',down_rv='sqrt(sum min(r,0)^2)',downside_share='down variance/max(total variance,1e-12)',jump_concentration='max(r^2)/max(sum r^2,1e-12)',vol_rate_ratio='(V_short/sqrt(short))/max(V_long/sqrt(long),1e-6)',trend_rate_change='N_short/short - N_long/long')
  return fm[field]+'; complete strictly past 5m window(s) given by suffix.'
 if c.startswith('NEWPRICE__'):
  field=c.split('__')[1]
  return dict(rv='sqrt(sum r^2)',up_rv='sqrt(sum max(r,0)^2)',down_rv='sqrt(sum min(r,0)^2)',net='sum r',past_range='H-L',distance_from_high='H-C',distance_from_low='C-L',max_abs_return='max abs(r)',rv_ratio='(RV_short/sqrt(short))/max(RV_long/sqrt(long),1e-6)')[field]+'; complete strictly past 5m window(s) in suffix.'
 raise ValueError('Missing feature formula: '+c)
feature_formulas=[dict(id=fid[c],feature=c,formula=formula(c)) for c in base_names]
pd.DataFrame(feature_formulas).to_csv(SUP/'used_feature_formulas.csv',index=False)

W,H=landscape(A4);WIDTH=W-72;NAVY=colors.HexColor('#17354a');TEAL=colors.HexColor('#087f8c')
styles={n:ParagraphStyle(n,fontName=f,fontSize=s,leading=l,textColor=c,spaceAfter=a,splitLongWords=True) for n,f,s,l,c,a in [
 ('title','Helvetica-Bold',23,27,NAVY,12),('head','Helvetica-Bold',13,17,TEAL,8),('body','Helvetica',10,14,NAVY,8),
 ('small','Helvetica',8.5,11,NAVY,6),('cell','Helvetica',8.3,11,NAVY,0),('th','Helvetica-Bold',8.3,11,colors.white,0),('code','Courier',7.4,10,NAVY,0)]}
story=[]
def para(s,style='body'):return Paragraph(escape(str(s)),styles[style])
def text(s,style='body'):story.append(para(s,style))
def page(title,subtitle=None):
 if story:story.append(PageBreak())
 text(title,'title')
 if subtitle:text(subtitle,'small')
def table(head,rows,widths=None,codecol=-1):
 if not rows:return
 t=Table([[para(h,'th') for h in head]]+[[para(v,'code' if i==codecol else 'cell') for i,v in enumerate(row)] for row in rows],colWidths=widths or [WIDTH/len(head)]*len(head),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f6'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.extend([t,Spacer(1,8)])
def fmt(v):return '--' if pd.isna(v) else f'{v:.3f}'
def chunks(a,n):return [a[i:i+n] for i in range(0,len(a),n)]
def score(t,col):return allm[(allm.run=='G')&(allm.target==t)&(allm.forecast==col)&(allm.period=='ALL')].iloc[0]

page('BTC | 300-hour magnitude models','14 November 2026 | Seven completed stages | 17,473 matched origins per target | Errors in log basis points')
text('Best-MAE models, architectures, features, formulas and rankings','head')
text('Chosen by lowest pooled MAE among the completed saved forecasts audited here. UP is a diagnostic quarterly HGB/GRU candidate; DOWN is a selected GRU/median policy. These different validation statuses matter even though both have lower observed MAE than both baselines.')
rows=[]
for t,col,name in [('UP','candidate','HGB + GRU16 (diagnostic)'),('DOWN','selected','GRU16 + median fallback')]:
 r=score(t,col);rows.append([t,name,fmt(r.MAE),fmt(r.MedianAE),fmt(r.RMSE),fmt(r.IC)])
table(['Target','Best-MAE workflow','MAE','Median abs. error','RMSE','Spearman IC'],rows,[45,245,90,110,90,WIDTH-580])
table(['Target','Baseline','MAE','Median abs. error','RMSE'],[[t,c.replace('training_','').title(),fmt(score(t,c).MAE),fmt(score(t,c).MedianAE),fmt(score(t,c).RMSE)] for t in ['UP','DOWN'] for c in ['training_mean','training_median']],[60,180,160,180,WIDTH-580])
text('UP best MAE: 454.893284. DOWN best MAE: 455.151240. Neither beats the mean baseline on RMSE. UP failed all four earlier MAE acceptance gates; its selected forecast remains the median. DOWN accepted January/April GRU updates and uses the median in July/October.')
text('This report audits saved predictions and inspects saved estimator metadata. No model training or new inference was performed. Historical evaluation has been examined repeatedly; these results are not fresh independent confirmation.','small')

page('Best UP | architecture, features and forecast formula','Chosen by MAE: G.candidate, run 20261114T165116421962Z. Diagnostic, not gate-approved.')
text('Let m_q and mu_q be the median and mean of all strictly matured training labels at issue q. Let H_q be the nonnegative absolute-loss histogram tree forecast. Let R_q be the GRU16 forecast defined on the model-formula page. Equations below apply when at least 80% of required inputs are finite; otherwise the constant anchor is used.')
table(['Issue','Architecture / inputs','Exact blend on usable rows','Status'],[
 ['Jan 2024','HGB; flow/liquidation 30 features -> 60 standardized channels','0.95*m_q + 0.05*H_q','Diagnostic'],
 ['Apr 2024','HGB; same 30-feature pack, separately fitted','0.90*m_q + 0.10*H_q','Diagnostic'],
 ['Jul 2024','HGB; same 30-feature pack, separately fitted','0.90*m_q + 0.10*H_q','Diagnostic'],
 ['Oct 2024 / Jan 2025','GRU16; leverage 38 features at four lags -> 4 x 76 channels','0.95*mu_q + 0.05*R_q','Diagnostic']],[100,285,245,WIDTH-630])
text('HGB settings: absolute_error, 60 iterations, learning rate 0.05, depth 2, 4 leaves, minimum leaf 256, L2=10, max_bins=63, random_state=48, no early stopping. At most 3,000 evenly spaced estimator rows; training preprocessing uses all mature rows.')
text('GRU16: one recurrent layer, hidden size 16, linear scalar head, bounded tanh residual. Leverage input shape 4 x 76; 4,529 trainable parameters. AdamW lr=0.001, weight decay=0.03; 12 epochs, batch 512, gradient clip 1, seed 48; at most 2,048 estimator rows.')
text('Scores: MAE 454.893284; median absolute error 349.220927; RMSE 634.760294; Spearman IC 0.015343. The larger GRU is used only in October; this is not one standalone GRU for the entire period.')

page('Best DOWN | architecture, features and forecast formula','Chosen by MAE: G.selected. Accepted GRU updates in January/April; later quarters use a constant.')
table(['Issue','Architecture / inputs','Exact blend on usable rows','Status'],[
 ['Jan 2024','GRU16; leverage 38 -> 4 x 76; 4,529 parameters','0.25*m_q + 0.75*R_q','Accepted'],
 ['Apr 2024','GRU16; flow/liquidation 30 -> 4 x 60; 3,761 parameters','0.25*m_q + 0.75*R_q','Accepted'],
 ['Jul 2024','No feature inputs; constant training median','m_q','Fallback'],
 ['Oct 2024 / Jan 2025','No feature inputs; constant training median','m_q','Fallback']],[110,280,230,WIDTH-620])
text('Both GRUs: single recurrent layer, hidden size 16, scalar linear head; AdamW lr=0.001, weight decay=0.03, 12 epochs, batch 512, gradient clip 1, deterministic seed 48; 2,048 estimator-row cap. Exact snapshots at t-90h, t-60h, t-30h and t. Full-training median, imputation and standardization.')
text('R_q = max(0, m_q + 0.75*max(m_q,1)*tanh(w_h*h_last+b_h)). The final model blend adds another factor 0.75 to the GRU residual. If less than 80% of required values are finite, predict m_q.')
text('Scores: MAE 455.151240; median absolute error 334.608250; RMSE 653.278562; Spearman IC 0.079913. Earlier DOWN selected MAE was 459.144369, so mean absolute error improved by 0.870%. Its median absolute error slightly worsened from 334.101530.')
text('The earlier E.candidate has lower median absolute error (333.315411), but worse MAE (463.067842). This report selects by MAE, as requested; it does not claim one forecast minimizes every metric.')

page('Evaluation, score definitions and ranking scope')
table(['Item','Definition / audit'],[
 ['Targets','BTCUSDT maximum nonnegative UP/DOWN path excursions over 18,000 minutes (300h). UP=max(0,max_s 10000*ln(P(t+s)/P(t))); DOWN=max(0,-min_s 10000*ln(P(t+s)/P(t))), 0<s<=300h. Canonical valid path labels, including zeros; not signed endpoint return.'],
 ['Population',f"{data['A'].decision_time.min()} to {data['A'].decision_time.max()}; 17,473 origins per target. 47,088 eligible training/evaluation rows. Development exclusions: 504 INVALID_PRICE_PATH and 504 OUTSIDE_RELEASE."],
 ['Timing','Jan/Apr/Jul/Oct 2024 expanding training; October retained in January 2025. Strict origin+300h<boundary in training and earlier validation. Original 48h input gap inherited; it does not replace the 300h label purge.'],
 ['MAE / median absolute error','MAE=mean(abs(y-p)); MedianAE=median(abs(y-p)). Both are error summaries of the same predictions. These are different from mean/median BASELINE forecasts, which predict a training constant. Lower is better.'],
 ['RMSE / IC / bias','RMSE=sqrt(mean((y-p)^2)); Spearman IC=rank correlation(y,p); bias=mean(p-y), negative means underprediction. Constant predictions may have undefined IC.'],
 ['Rank scope','Seven completed stages A-G. Policy ranking includes selected and diagnostic candidate policies plus baselines. Full-coverage ranking also includes B fixed variants, with identical predictions consolidated. Partially available forecasts excluded from full-period rankings.'],
 ['Verification','Prediction SHA256, unique keys, full coverage, matched timestamps/issues/truths/baselines; pooled and period score recomputation. Saved model hashes checked; estimator metadata read. Replay completion flags are reported, not rerun.'],
 ['Limits','Repeated historical adaptive search, overlapping 300h labels, no fresh holdout or significance claim. A diagnostic candidate is not automatically a validated deployment model.']],[125,WIDTH-125])

for t in ['UP','DOWN']:
 page(t+' | model-policy ranking','Ordered by pooled MAE; model identifiers map to source stages A-G. MedianAE is median absolute error.')
 rr=pol[pol.target.eq(t)].to_dict('records')
 table(['Rank','Policy','Role','MAE','MedianAE','RMSE','IC','Bias'],[[int(r['MAE_rank']),r['run']+'.'+r['forecast'],r['role'],fmt(r['MAE']),fmt(r['MedianAE']),fmt(r['RMSE']),fmt(r['IC']),fmt(r['Bias'])] for r in rr],[40,145,130,80,85,85,65,WIDTH-630])
for t in ['UP','DOWN']:
 for k,part in enumerate(chunks(full[full.target.eq(t)].to_dict('records'),12)):
  page(t+f' | all distinct complete forecasts {k+1}','Includes fixed diagnostic price variants. Aliases identify identical forecasts; all alias names are preserved in the support CSV.')
  table(['MAE rank','Forecast','Role','MAE','MedianAE','RMSE','IC'],[[int(r['MAE_rank']),r['aliases'],r['role'],fmt(r['MAE']),fmt(r['MedianAE']),fmt(r['RMSE']),fmt(r['IC'])] for r in part],[45,245,130,90,90,90,WIDTH-690])
for t,col in [('UP','candidate'),('DOWN','selected')]:
 page('Best '+t+' | period scores','October fitted model also serves January 2025. Pooled improvement does not imply improvement every period.')
 g=data['G'][data['G'].target.str.startswith(t.lower())]
 rows=[]
 for period,a in g.groupby('period'):
  m=metrics(a.actual.to_numpy(),a[col].to_numpy());med=metrics(a.actual.to_numpy(),a.training_median.to_numpy());mn=metrics(a.actual.to_numpy(),a.training_mean.to_numpy());old=metrics(a.actual.to_numpy(),a.previous_reference.to_numpy())
  rows.append([period,len(a),fmt(m['MAE']),fmt(old['MAE']),fmt(med['MAE']),fmt(mn['MAE']),fmt(m['MedianAE']),fmt(m['RMSE']),fmt(m['IC'])])
 table(['Period','N','Model MAE','Prior MAE','Median baseline MAE','Mean baseline MAE','MedianAE','RMSE','IC'],rows,[65,45,90,90,110,110,90,90,WIDTH-690])

page('Models and hyperparameters | original stages A-D')
table(['Stage / model','Architecture, training and selection'],[
 ['A original search','522 masked F4Z candidates; training coverage>=80%, >=1000 finite rows and variation. Empty-start shortlist16, up to6 features. Ridge alpha=0.01*n or HGB absolute_error, 80 iterations, depth3, 7 leaves, minleaf128, lr0.05, L2=1, bins255, seed48, no early stopping.'],
 ['A screening','Absolute-error DecisionTreeRegressor depth2/minleaf128, seed48; at most4096 evenly spaced screening rows. Three earlier purged quarters; >=0.1% MAE gain and >=2/3 wins; fallback median.'],
 ['B price follow-up','Same Ridge/HGB families; add causal price blocks or remove original features. 51 new price features. Earlier-only selection relative to original. Fixed alternative forecasts are diagnostic, not separate tuned policies. Selected original forecast retained in all eight target/quarter cases.'],
 ['C long-window packs','Original HGB settings; packs current, trend/path31, volatility23, all54 replacement and current+54. Median model blend0/.25/.5/.75/1. Three earlier windows, .1% MAE gain and2wins vs median else median.'],
 ['D compact fresh features','Empty-start54 LONG300 candidates, quality>=80%/1000/variation; single-feature screen4096 rows; shortlist12, rank-correlation redundancy .95, greedy up to6. Ridge alpha=n or10*n; absolute-error decision tree depth2/minleaf256/seed48.'],
 ['D gate','Three earlier windows; .1% pooled median-MAE improvement,2/3 wins, no window>2% worse; otherwise median. No reserved old feature. All exact saved quarterly feature packs and estimator settings are recorded later.']],[145,WIDTH-145])

page('Models and hyperparameters | recent stages E-G')
table(['Stage','Search and neural/tree budget'],[
 ['E fast joint search','Combined62; HGB/MLP16/GRU8, each L1 and L2 losses. HGB40 iterations/depth3/7leaves/minleaf128/L2=10/bins63/lr.05, cap3000. MLP16 and GRU8 lr.002/decay.1,8epochs/cap2048.'],
 ['F UP MAE refinement','Leverage38/combined62; HGB/MLP16/GRU8 L1 only. Same HGB settings; neural12epochs/cap2048. UP only;96 inner fits.'],
 ['G both-target refinement','Leverage38/flow30/combined62; HGB L1 depth2/4leaves/minleaf256/60iterations/L2=10/bins63/lr.05/cap3000. GRU16 L1 lr.001/decay.03/12epochs/cap2048.192 inner fits.'],
 ['Neural architecture','MLP16: 2d channels -> 16 ReLU -> linear scalar -> bounded tanh residual. GRU8/16: 4 x 2d channels -> one GRU layer -> scalar head -> bounded tanh residual. Hidden initial state zero.'],
 ['Shared neural optimization','AdamW; batch512; gradient norm clip1; deterministic seed48; zero-initialized scalar head. Constant target normalization by max(training median,1), no extra target log. Estimator row subsampling is evenly spaced; preprocessing uses all mature rows.'],
 ['Blend search','E/F model weight .1/.25/.5/1; G .05/.1/.25/.5/.75/1. Constant anchor=(1-a)*median+a*mean, a in0/.5/1. Input coverage<80% returns this anchor.'],
 ['Selection','E requires both MAE and RMSE gains against both baselines. F/G optimize MAE only. All use four earlier quarters, .1% pooled gain,3/4 wins-or-ties, no window>2% worse; else median. Diagnostic candidate separately saved.']],[145,WIDTH-145])
text('The interrupted full 300h search is not a completed forecast and is not ranked. Configurations tested only inside validation receive no fabricated outer-evaluation rank.')

page('Prediction formulas | preprocessing and model components')
table(['Component','Formula / exact meaning'],[
 ['Preprocessing','For each raw column, impute missing values with training median. Append a binary missing flag. Standardize the resulting value+flag matrix using training mean and population std; std<1e-8 becomes1. Neural inputs then clip to[-8,8]. No preprocessing is refitted on evaluation data.'],
 ['HGB','H=max(0,b0+sum_k contribution_k(z)). Each learned tree contribution includes learning-rate scaling. Conceptually b0+eta*sum raw_leaf_update; do not multiply sklearn stored contributions by eta again.'],
 ['Ridge / decision tree','Ridge prediction=max(0,beta0+sum beta_j*z_j); squared-error plus alpha*||beta||^2. Absolute-error tree predicts fitted leaf medians; depth and leaf support constrain complexity.'],
 ['GRU reset / update','r=sigmoid(W_ir*z+b_ir+W_hr*h_prev+b_hr); u=sigmoid(W_iu*z+b_iu+W_hu*h_prev+b_hu).'],
 ['GRU candidate / hidden','n=tanh(W_in*z+b_in+r*(W_hn*h_prev+b_hn)); h=(1-u)*n+u*h_prev. This is the PyTorch GRU convention.'],
 ['Neural forecast','R=max(0,m+0.75*max(m,1)*tanh(w_h*h_last+b_h)); MLP uses its last ReLU vector in place of recurrent h_last.'],
 ['Recent policy blend','anchor=(1-a)*m+a*mu; final=w*model+(1-w)*anchor. At input coverage<80%, final=anchor. Median-only fallback predicts m.'],
 ['Reproducibility','Architecture and hyperparameters do not uniquely define fitted numerical predictions. Exact neural weights, tree thresholds and imputation/scaling arrays remain in the original SHA256-verified joblib bundles. Source paths and hashes are in the support archive.']],[145,WIDTH-145])

page('Feature engineering | hourly LONG price features')
text('Let r_j be an hourly log-bps return formed from exactly12 five-minute endpoints in[t-1h,t). Let N_h=sum r, V_h=sqrt(sum r^2), A_h=sum abs(r); C is the current cumulative hourly level and H_h/L_h are the local high/low including endpoints. Incomplete price windows are missing. Half-hour origins use the latest completed hour, with less than one-hour tolerance.')
table(['Measurement','Formula'],[
 ['Net / volatility per sqrt hour','N_h; V_h/sqrt(h)'],['Up variance share','sum(max(r,0)^2)/max(V_h^2,1e-16)'],
 ['Efficiency / normalized momentum','N_h/max(A_h,1e-8); N_h/max(V_h,1e-8)'],['Range / range position','H_h-L_h; (C-L_h)/max(H_h-L_h,1e-8)'],
 ['Drawdown / rebound in vol units','(H_h-C)/max(V_h,1e-8); (C-L_h)/max(V_h,1e-8)'],['Positive-hour share / jump share','fraction(r>0); max(r^2)/max(sum r^2,1e-16)'],
 ['Hourly skew','Rolling sample skewness, pandas convention.'],['Cross-window contrasts','Vol ratio=(V_s/sqrt(s))/max(V_l/sqrt(l),1e-8); drift change=N_s/s-N_l/l.'],
 ['Generated universe','12 measures at24/150/168/300/336/600/720/1440h, plus8 contrasts for24:300,300:600,168:720,336:1440 =>104 features.'],
 ['Six focused price inputs','300h volatility per sqrt hour, normalized momentum, range position, up variance share;168h drawdown in vol units;336h jump variance share.']],[200,WIDTH-200])

page('Feature engineering | market state and interactions')
table(['Group','Formula / availability'],[
 ['Source measurements','11 masked5m source columns: funding level/change,basis level/change,OI range position/unwind,flow pressure/activity,local and peer historical liquidation activity rates,venue breadth. Source percentiles are consumed as supplied.'],
 ['STATE','For s(u), past window[t-h,t): mean, population std, deviation=s(t-30min)-mean, high_share=fraction(s>=.8). Require>=80% expected half-hour snapshots; no filling. Windows6/24/168/300h, plus lag30min.'],
 ['funding_oi','mean(funding_level)*mean(oi_position)'],['oi_lowflow','mean(oi_position)*(1-mean(flow_pressure))'],
 ['unwind_negative_momentum','mean(oi_unwind)*max(-LONG normalized_momentum,0)'],['basis_funding','(mean(basis_level)-.5)*(mean(funding_level)-.5)'],
 ['liq_volatility','mean(liq_history)*LONG volatility per sqrt hour'],['flow_momentum','(mean(flow_pressure)-.5)*LONG normalized_momentum'],
 ['Leverage38','6 price +6 funding/basis/OI families x2 statistics(mean/deviation)x2 windows(24/300h) +4 interactions x2windows. oi_lowflow additionally depends on flow_pressure upstream.'],
 ['Flow30 / combined62','6 price +5 flow/liquidation families x2stats x2windows +2 interactions x2windows. Combined is the union. Liquidation history measures activity rates, not current liquidation sizes.'],
 ['GRU sequences','Base feature snapshots at90/60/30/0h lag, exact timestamp reindexing, no fill. Assembled on full X inventory BEFORE label-validity filtering.38baseinputs ->152values+152flags, arranged4x76.30baseinputs ->4x60.']],[170,WIDTH-170])

page('Market source columns | exact upstream identities','These are the masked source measurements aggregated into STATE and INTERACT features. Upstream percentile calculations are inherited, not re-estimated.')
source_path=track(EXP/RUNS['G'][1]/'market_state_features_300h.py')
source_ast=ast.parse(source_path.read_text())
sources=next(ast.literal_eval(n.value) for n in source_ast.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
table(['Family alias','Exact source column'],list(sources.items()),[145,WIDTH-145],codecol=1)
text('The alias is used in every STATE feature name. For example, STATE__oi_unwind_mean__300h averages the exact OI leverage-unwind percentile source over the strictly past 300-hour window. Interactions combine the corresponding historical source means with each other or with causal price features.')

page('Feature engineering | older NEWPRICE and LONG300')
table(['Universe','Formula and construction'],[
 ['B NEWPRICE51','Complete strictly past five-minute returns at1/4/12/24/48/168h.8 measures/window: RV,upRV,downRV,net,past range,distance from high,distance from low,max absolute return. Plus volatility-rate ratios1:24,4:24,24:168.'],
 ['NEWPRICE equations','RV=sqrt(sum r^2); upRV=sqrt(sum max(r,0)^2); downRV=sqrt(sum min(r,0)^2); net=sum r. From local cumulative past prices:range=H-L,drawdown=H-C,rebound=C-L. Ratio=(RV_s/sqrt(s))/max(RV_l/sqrt(l),1e-6).'],
 ['C/D LONG30054','12 measures at168/300/720/1440h on five-minute returns.7 trend/path measures and5 volatility/regime measures perwindow, plus6 cross-window contrasts. Strictly past complete windows; different sampling from hourly LONG.'],
 ['Trend/path31','Net N; N/max(V,1e-6); N/max(sum abs(r),1e-6); (C-L)/max(H-L,1e-6); H-C; C-L; (H-L)/max(V,1e-6). Plus drift-rate contrasts N_s/s-N_l/l.'],
 ['Volatility/regime23','V=sqrt(sum r^2); positive and negative RV; downsidevariance/max(totalvariance,1e-12); max(r^2)/max(totalvariance,1e-12). Plus volatility-rate ratios using denominator floor1e-6.'],
 ['Cross-window pairs','168:720,300:720,720:1440h. One vol-rate ratio and one trend-rate change perpair.'],
 ['F4Z inherited inputs','Preserve exact family,measurement,timeframe in column name. Masked availability and source limitations inherited. Upstream robust-Z/percentile algorithms are not invented or independently rebuilt here.']],[155,WIDTH-155])

# Exact winning feature sets and full used-feature index.
for t in ['UP','DOWN']:
 page('Best '+t+' | exact quarterly feature packs','Pack IDs expand below; feature IDs map to exact column names and formulas. No feature importance is inferred from inclusion.')
 rows=[]
 for r in best:
  if r['target']!=t:continue
  sp=r['specification'];cols=r['features'];seq=any(c.startswith('SEQ') for c in cols)
  bases=list(dict.fromkeys(re.sub(r'^SEQ\d+__','',c) for c in cols))
  rows.append([r['issue'],r['architecture'],r['pack'],len(cols),('4 snapshots: ' if seq else '')+', '.join(fid[c] for c in bases) if cols else 'No predictor inputs; training median'])
 table(['Issue','Architecture','Pack','Raw values','Exact base feature IDs'],rows,[90,150,50,65,WIDTH-355])
 text('For GRUs the ordered base list is repeated at lag90,60,30,0h, then preprocessing forms value and missing-flag channels for each step. Each quarter is independently fitted; same feature list does not mean same weights.')

winning_names=sorted(set(re.sub(r'^SEQ\d+__','',c) for r in best for c in r['features']))
for k,part in enumerate(chunks(winning_names,10)):
 page(f'Best models | feature names and formulas {k+1}','Exact engineered features used by the chosen best-MAE quarterly forecasts. Formula window is given by the name suffix.')
 table(['ID','Exact feature name','Formula'],[[fid[c],c,formula(c)] for c in part],[40,350,WIDTH-390],codecol=1)

page('Best-model validation scores | why selected differs from candidate')
rows=[]
for s in cards['G']:
 target=side(s['target'])
 for branch in ['selected','candidate']:
  a=s[branch]
  rows.append([target,s['issue'][:7],branch,a.get('config','median'),a.get('pack','--'),fmt(a.get('validation_mae',np.nan)),str(a.get('eligible','fallback'))])
table(['Side','Issue','Forecast','Config','Pack','Earlier MAE','Eligible'],rows,[45,85,80,100,140,100,WIDTH-550])
text('Candidate is the lowest earlier pooled-MAE option before consistency gates. Selected is the lowest earlier-MAE eligible option, or median. DOWN January/April selected weights differ from the unconstrained candidate; rejected options can have better earlier pooled MAE but poor window consistency. UP has no eligible option in this run.')

for label,(name,_) in RUNS.items():
 rr=[r for r in models if r['run']==label and (label!='B' or r['specification'].get('selected'))]
 for k,part in enumerate(chunks(rr,8)):
  page(label+' | '+name+f' / saved quarterly models {k+1}','Actual saved architectures and inputs. Complete hyperparameter dictionaries and B fixed-variant metadata are in model_details.json.')
  rows=[]
  for r in part:
   sp=r['specification'];hp=r['hyperparameters']
   settings='; '.join(k+'='+str(hp[k]) for k in ['loss','alpha','max_iter','max_depth','max_leaf_nodes','min_samples_leaf','hidden','lr','decay','epochs','estimator_rows'] if k in hp)
   rows.append([r['target'],r['issue'],r['forecast'],r['architecture'],r['pack'],settings or str(hp)])
  table(['Side','Issue','Forecast','Architecture','Pack','Settings'],rows,[40,75,85,170,45,WIDTH-415])
for t in ['UP','DOWN']:
 page('B fixed variants | '+t+' exact input-pack mapping','These variants are diagnostic. The original variant was selected for every issue. Each entry points to the exact ordered feature list in the pack appendix.')
 rr=[r for r in models if r['run']=='B' and r['target']==t]
 variants=list(dict.fromkeys(r['forecast'] for r in rr))
 rows=[]
 for v in variants:
  sub=[r for r in rr if r['forecast']==v]
  rows.append([v,', '.join(sorted(set(r['architecture'] for r in sub)))]+[next((r['pack'] for r in sub if r['issue']==q),'not fitted') for q in ['2024-01','2024-04','2024-07','2024-10']])
 table(['Variant','Saved family','Jan pack','Apr pack','Jul pack','Oct pack'],rows,[205,245]+[(WIDTH-450)/4]*4)

for k,part in enumerate(chunks(list(pack_features.items()),6)):
 page(f'All used feature packs | {k+1}','Ordered input IDs. B fixed diagnostic variants included. Full raw ordered column names exported in feature_packs.json.')
 rows=[]
 for pid,cols in part:
  seq=any(c.startswith('SEQ') for c in cols);bases=list(dict.fromkeys(re.sub(r'^SEQ\d+__','',c) for c in cols))
  rows.append([pid,len(cols),('Sequence90/60/30/0h: ' if seq else 'Current: ')+', '.join(fid[c] for c in bases) if cols else 'None'])
 table(['Pack','Raw values','Ordered base feature IDs'],rows,[50,65,WIDTH-115])
others=[c for c in base_names if c not in winning_names]
for k,part in enumerate(chunks(others,12)):
 page(f'Other model features | exact names and formulas {k+1}','Includes original F4Z, long-price and price-block features used in saved selected or fixed diagnostic models.')
 table(['ID','Exact feature name','Formula'],[[fid[c],c,formula(c)] for c in part],[40,350,WIDTH-390],codecol=1)
if partial:
 page('Partial-coverage fixed variants','Not ranked against full-period policies; available-row scores remain in the support CSV.')
 table(['Run','Target','Forecast','Available rows'],[[r['run'],r['target'],r['forecast'],r['rows']] for r in partial],[45,65,WIDTH-220,110])
for k,part in enumerate(chunks(audit,4)):
 page(f'Sources and reproducibility | {k+1}','Experiment paths relative to Week2/Code/experiments. Full hashes, contracts and selection cards are in the support archive.')
 table(['ID','Experiment','Source path'],[[r['run'],r['name'],r['path']] for r in part],[35,180,WIDTH-215])
 text('Companion BTC_300h_Report_Support_2026-11-14.zip: complete rankings, period scores, feature IDs/formulas/packs, model hyperparameters, validation trials, selection cards and hash audit. Fitted numerical parameters remain in the original model files referenced by path and SHA256.')
 text('Scope: every completed 300h stage with saved predictions discovered for this report is included. The incomplete full magnitude-model search is excluded. Source data, historical forecasts and fitted models are unchanged.')
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#c9d6df'));canvas.line(36,32,W-36,32)
 canvas.setFont('Helvetica',8);canvas.setFillColor(NAVY);canvas.drawString(36,20,'BTC 300h | Saved research evidence | 14 Nov 2026');canvas.drawRightString(W-36,20,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=43,title='BTC 300h models, features and rankings',author='Research report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
texts=[p.extract_text() for p in PdfReader(PDF).pages]
assert all(len(t)>100 for t in texts)
assert '454.893' in ''.join(texts) and '455.151' in ''.join(texts)
(SUP/'source_audit.json').write_text(json.dumps(dict(runs=audit,files=hashes,models=model_hashes),indent=2))
(SUP/'REPORT_CHECKS.json').write_text(json.dumps(dict(pages=len(texts),prediction_hashes_verified=len(RUNS),model_hashes_verified=len(model_hashes),origins_per_target=17473,metrics_recomputed=True,no_training=True,no_inference=True,feature_count=len(base_names),packs=len(packs)),indent=2))
with zipfile.ZipFile(OUT/'BTC_300h_Report_Support_2026-11-14.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in SUP.rglob('*'):
  if q.is_file():z.write(q,q.relative_to(SUP))
print(json.dumps(dict(pdf=str(PDF),pages=len(texts),rank_counts=full.groupby('target').size().to_dict(),models=len(model_hashes),features=len(base_names),packs=len(packs))))
