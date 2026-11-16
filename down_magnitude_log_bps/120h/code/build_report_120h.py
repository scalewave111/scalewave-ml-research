"""Audit all completed120h experiments and create a PDF; no training/inference."""
from pathlib import Path
import json,hashlib,shutil,zipfile,re
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib import colors
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'Code/experiments'
OUT=ROOT/'Report';SUP=OUT/'120h_report_support_2026-11-14'
PDF=OUT/'BTC_120h_Models_Features_Rankings_2026-11-14.pdf'
RUNS={
 'A':('Initial feature search','excursion_pipeline_120h_v1/20261113T114805583307Z/search/20261113T114812390024Z'),
 'B':('Price-block follow-up','excursion_pipeline_120h_v1/20261113T114805583307Z/price_followup/20261113T120541578982Z'),
 'C':('HGB refinement','excursion_tuning_120h_v1/20261113T121934748718Z'),
 'D':('Long-feature search','long_features_120h_v1/20261113T123515172594Z'),
 'E':('Compact MLP','mlp_compact_120h_v1/20261113T130228651265Z'),
 'F':('Market HGB/MLP/GRU','magnitude_models_120h_v1/20261114T124741356674Z'),
 'G':('Joint MAE/RMSE refinement','dual_objective_120h_v1/20261114T132002835184Z'),
 'H':('Transformed joint refinement','transformed_dual_120h_v1/20261114T133705097838Z')}
PRIMARY={'A':'prediction','B':'selected','C':'selected','D':'new_features','E':'mlp_selected','F':'selected','G':'selected','H':'selected'}
SUP.mkdir(parents=True,exist_ok=True)
hashes={}
def track(p):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(track(p))
def metrics(y,p):
 return dict(MAE=float(np.abs(y-p).mean()),RMSE=float(np.sqrt(np.square(y-p).mean())),IC=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if np.ptp(y)>0 and np.ptp(p)>0 else float('nan'),Bias=float(np.mean(p-y)))
def side(t):return t.split('_')[0].upper()
data={};cards={};contracts={};audit=[];allmetrics=[];policies=[];vectors={};periodrows=[]
for label,(name,relative) in RUNS.items():
 path=EXP/relative;done=js(path/'COMPLETE.json')
 assert done['status']=='COMPLETE' and hashlib.sha256((path/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']
 f=csv(path/'predictions.csv').sort_values(['target','decision_time']).reset_index(drop=True)
 assert not f.duplicated(['target','decision_time']).any() and f.groupby('target').size().eq(18193).all()
 if data:
  assert f[['target','decision_time','issue']].equals(data['A'][['target','decision_time','issue']])
  np.testing.assert_allclose(f.actual,data['A'].actual,atol=1e-10,rtol=0)
  for c in ['training_median','training_mean']:np.testing.assert_allclose(f[c],data['A'][c],atol=1e-8,rtol=0)
 data[label]=f;cards[label]=js(path/'selections.json');contracts[label]=js(path/'CONTRACT.json')
 for filename in ['COMPLETE.json','CONTRACT.json','selections.json']:shutil.copyfile(path/filename,SUP/f'{label}_{filename}')
 for p in path.glob('*.py'):track(p)
 source_metrics=csv(path/('metrics_all_variants.csv' if label=='B' else 'metrics.csv'))
 source_metrics.to_csv(SUP/f'{label}_saved_metrics.csv',index=False)
 for target,g in f.groupby('target'):
  for col in f.columns:
   if col in ['decision_time','issue','target','actual','period'] or not pd.api.types.is_numeric_dtype(f[col]):continue
   for period,part in [('ALL',g)]+list(g.groupby('period')):
    usable=np.isfinite(part[col]);sub=part.loc[usable]
    if not len(sub):continue
    values=metrics(sub.actual.to_numpy(),sub[col].to_numpy())
    allmetrics.append(dict(run=label,target=side(target),forecast=col,period=period,rows=len(sub),full_coverage=len(sub)==len(part),**values))
    saved=source_metrics[(source_metrics.target==target)&(source_metrics.model==col)&(source_metrics.period==period)]
    if len(saved)==1:
     for lower,upper in [('mae','MAE'),('rmse','RMSE'),('ic','IC')]:np.testing.assert_allclose(saved.iloc[0][lower],values[upper],rtol=1e-8,atol=1e-8,equal_nan=True)
   if col==PRIMARY[label] or (col=='candidate' and label in 'FGH'):
    v=g[col].to_numpy();assert np.isfinite(v).all()
    row=dict(run=label,target=side(target),forecast=col,name=name,role='candidate' if col=='candidate' else 'selected',**metrics(g.actual.to_numpy(),v))
    policies.append(row);vectors[(side(target),label,col)]=v
  for period,part in g.groupby('period'):
   periodrows.append(dict(run=label,target=side(target),period=period,**metrics(part.actual.to_numpy(),part[PRIMARY[label]].to_numpy())))
 audit.append(dict(run=label,name=name,path=relative,predictions_sha256=done['predictions_sha256'],rows=len(f),hash_verified=True,replay_reported=done.get('replay_passed',done.get('original_replay_passed'))))
for target,g in data['A'].groupby('target'):
 for col in ['training_median','training_mean']:
  policies.append(dict(run='BASE',target=side(target),forecast=col,name=col.replace('_',' ').title(),role='baseline',**metrics(g.actual.to_numpy(),g[col].to_numpy())))
  vectors[(side(target),'BASE',col)]=g[col].to_numpy()
pol=pd.DataFrame(policies)
for target in ['UP','DOWN']:
 ix=pol.target.eq(target)
 for metric,ascending in [('MAE',True),('RMSE',True),('IC',False)]:pol.loc[ix,metric+'_rank']=pol.loc[ix,metric].rank(method='min',ascending=ascending)
pol=pol.sort_values(['target','MAE','run']);pol.to_csv(SUP/'policy_rankings.csv',index=False)
allm=pd.DataFrame(allmetrics);allm.to_csv(SUP/'all_forecasts_period_metrics.csv',index=False)
pd.DataFrame(periodrows).to_csv(SUP/'selected_period_metrics.csv',index=False)
# Full-period fixed variants join the policy ranking, with duplicate predictions collapsed.
full=[];partial=[]
for target in ['UP','DOWN']:
 seen=[]
 candidates=[(r['run'],r['forecast'],r['role']) for r in policies if r['target']==target]
 candidates += [('B',c,'fixed diagnostic') for c in data['B'].columns if c not in ['decision_time','issue','target','actual','period','selected','training_mean','training_median','original']]
 for run,col,role in candidates:
  g=data['A' if run=='BASE' else run];g=g[g.target.str.startswith(target.lower())]
  values=g[col].to_numpy(float);valid=np.isfinite(values)
  if not valid.all():
   partial.append(dict(target=target,run=run,forecast=col,rows=int(valid.sum())))
   continue
  match=next((r for v,r in seen if np.allclose(v,values,atol=1e-7,rtol=0)),None)
  if match is not None:match['aliases']+='; '+run+'.'+col;continue
  row=dict(target=target,run=run,forecast=col,role=role,aliases=run+'.'+col,**metrics(g.actual.to_numpy(),values))
  seen.append((values,row));full.append(row)
full=pd.DataFrame(full).sort_values(['target','MAE','run'])
for target in ['UP','DOWN']:
 ix=full.target.eq(target)
 for m,a in [('MAE',True),('RMSE',True),('IC',False)]:full.loc[ix,m+'_rank']=full.loc[ix,m].rank(method='min',ascending=a)
full.to_csv(SUP/'all_full_coverage_rankings.csv',index=False);pd.DataFrame(partial).to_csv(SUP/'partial_variant_coverage.csv',index=False)
# Exact quarterly policy inputs. Dual policies include the saved anchor and admitted correction inputs.
packs={};feature_sets={};selection_rows=[]
def pack_id(cols):
 seq=any(c.startswith('SEQ') for c in cols)
 names=list(dict.fromkeys(re.sub(r'^SEQ\d+__','',c) for c in cols))
 key=(tuple(names),seq)
 if key not in packs:packs[key]=f'P{len(packs)+1:02d}'
 feature_sets[packs[key]]=names
 return packs[key]+(' x4 snapshots' if seq else '')
def findcard(label,target,issue):return next(s for s in cards[label] if s['target']==target and s['issue']==issue)
for label in RUNS:
 for s in cards[label]:
  target,issue=s['target'],s['issue']
  branches=['selected','candidate'] if label in 'FGH' else ['selected']
  for branch in branches:
   spec=s.get(branch) if label in 'CFGH' else s
   anchor=[]
   if label in 'GH':anchor=findcard('B' if target.startswith('up_') else 'D',target,issue)['features']
   if spec is None:
    spec=dict(model='unchanged B',features=findcard('B',target,issue)['features'])
   cols=list(spec.get('features',[]))
   if label in 'GH':
    if spec['model']=='anchor':cols=anchor
    else:
     # Actual admitted inputs are saved separately; preserve unadmitted plans in evidence JSON.
     q=EXP/RUNS[label][1]/target/issue[:7]/f'{branch}_final_quality.json'
     if q.exists():cols=[v['feature'] for v in js(q) if v['eligible']]
     cols=list(dict.fromkeys(anchor+cols))
   if label=='C' and spec.get('weight',1)<1:cols=list(dict.fromkeys(findcard('B',target,issue)['features']+cols))
   model=spec.get('config',spec.get('model','HGB'))
   if label=='B':model='HGB / '+spec['variant']
   details='; '.join(f'{k}={spec[k]}' for k in ['pack','mode','weight','mean_weight','update_weight','days'] if k in spec)
   if label=='C' and spec.get('model')!='unchanged B':model='HGB '+model
   pid=pack_id(cols) if cols else 'No predictors'
   selection_rows.append(dict(run=label,target=side(target),issue=issue[:7],forecast=branch,model=model,settings=details,pack=pid,inputs=len(cols),features=cols))
   
feature_names=sorted(set(c for cols in feature_sets.values() for c in cols));fid={c:f'F{i+1:03d}' for i,c in enumerate(feature_names)}
pd.DataFrame(selection_rows).to_csv(SUP/'quarterly_models_and_features.csv',index=False)
pd.DataFrame([dict(id=fid[c],column=c) for c in feature_names]).to_csv(SUP/'feature_ids.csv',index=False)
pd.DataFrame([dict(pack=p,features=' | '.join(cols)) for p,cols in feature_sets.items()]).to_csv(SUP/'feature_packs.csv',index=False)
# Preserve every fixed variant card and validation score, not only reported winners.
trials=[]
for label,(_,relative) in RUNS.items():
 path=EXP/relative
 for p in path.rglob('validation_trials.csv'):
  f=csv(p);f.insert(0,'run',label);f.insert(1,'folder',str(p.parent.relative_to(path)));trials.append(f)
 for p in path.rglob('validation_trials.json'):
  obj=js(p)
  if isinstance(obj,list):
   f=pd.DataFrame(obj);f.insert(0,'run',label);f.insert(1,'folder',str(p.parent.relative_to(path)));trials.append(f)
 for p in path.rglob('selection.json'):
  dest=SUP/'cards'/label/p.relative_to(path);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);track(p)
if trials:pd.concat(trials,ignore_index=True).to_csv(SUP/'all_validation_trials.csv',index=False)

W,H=landscape(A4);WIDTH=W-72;navy=colors.HexColor('#17354a');teal=colors.HexColor('#087f8c')
styles={n:ParagraphStyle(n,fontName=f,fontSize=s,leading=l,textColor=c,spaceAfter=a,splitLongWords=True)
 for n,f,s,l,c,a in [('title','Helvetica-Bold',23,27,navy,13),('head','Helvetica-Bold',13,17,teal,8),
 ('body','Helvetica',10,14,navy,8),('small','Helvetica',8.5,11,navy,6),('cell','Helvetica',8.2,10.5,navy,0),
 ('th','Helvetica-Bold',8.2,10.5,colors.white,0),('code','Courier',7.3,9.5,navy,0)]}
story=[]
def para(s,style='body'):return Paragraph(escape(str(s)),styles[style])
def text(s,style='body'):story.append(para(s,style))
def page(title,subtitle=None):
 if story:story.append(PageBreak())
 text(title,'title')
 if subtitle:text(subtitle,'small')
def table(head,rows,widths=None,codecol=None):
 t=Table([[para(h,'th') for h in head]]+[[para(v,'code' if i==codecol else 'cell') for i,v in enumerate(row)] for row in rows],colWidths=widths or [WIDTH/len(head)]*len(head),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4f6'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,8))
def fmt(v):return '--' if pd.isna(v) else f'{v:.3f}'
def chunks(a,n):return [a[i:i+n] for i in range(0,len(a),n)]
page('BTC | 120-hour magnitude models','Research report | 14 November 2026 | All eight completed stages | Log basis points')
text('Models, architectures, hyperparameters, feature engineering and ranked results','head')
text('This report audits saved UP and DOWN excursion forecasts on the same 18,193 origins per target. It covers eight completed stages, including the compact MLP omitted from the recent conversational comparisons. Models are not retrained for this report.')
winner=[]
for target in ['UP','DOWN']:
 selected=pol[(pol.target==target)&(pol.role=='selected')].sort_values('MAE').iloc[0]
 candidate=pol[(pol.target==target)&(pol.role=='candidate')].sort_values('MAE').iloc[0]
 winner.extend([[target,'Best selected policy',selected['run']+' / '+selected['name'],fmt(selected.MAE),fmt(selected.RMSE),fmt(selected.IC)],
 [target,'Best diagnostic candidate',candidate['run']+' / '+candidate['name'],fmt(candidate.MAE),fmt(candidate.RMSE),fmt(candidate.IC)]])
table(['Target','Status','Workflow','MAE','RMSE','Rank IC'],winner,[45,150,WIDTH-390,65,65,65])
text('Correction: the completed compact MLP (E) has DOWN MAE 297.850, below the recent transformed selected policy at 300.909. Earlier statements that the latter was the best overall DOWN model were incomplete. This report includes E in every relevant ranking.')
text('A selected policy is chosen by its run-specific earlier-validation rules. A candidate is an ungated diagnostic. Fixed feature variants are evaluated research alternatives. None is promoted using the ranking in this report. The mean-baseline RMSE remains an unmet criterion for the selected and candidate policies. See the expanded ranking for fixed variants.')
page('Best 120-hour models | explicit scorecard','Best means lowest pooled MAE among selected policies, separately for UP and DOWN. It does not mean best on every metric.')
bestrows=[]
for target,label in [('UP','B'),('DOWN','E')]:
 r=pol[(pol.target==target)&(pol.run==label)&(pol.role=='selected')].iloc[0]
 med=pol[(pol.target==target)&(pol.forecast=='training_median')].iloc[0]
 mean=pol[(pol.target==target)&(pol.forecast=='training_mean')].iloc[0]
 bestrows.append([target,'HGB price-block policy' if target=='UP' else 'Compact MLP ensemble',fmt(r.MAE),fmt(r.RMSE),fmt(r.IC),f'{100*(med.MAE-r.MAE)/med.MAE:.2f}%',f'{100*(mean.MAE-r.MAE)/mean.MAE:.2f}%'])
table(['Side','Best selected model','MAE','RMSE','Rank IC','MAE gain vs median','MAE gain vs mean'],bestrows,[40,210,70,70,65,105,WIDTH-560])
table(['Baseline','UP MAE','UP RMSE','DOWN MAE','DOWN RMSE'],[
 [name]+[fmt(pol[(pol.target==t)&(pol.forecast==col)].iloc[0][m]) for t in ['UP','DOWN'] for m in ['MAE','RMSE']]
 for name,col in [('Training median','training_median'),('Training mean','training_mean')]],[180]+[(WIDTH-180)/4]*4)
text('Both selected winners beat BOTH the mean and median on MAE. Neither beats the mean baseline on RMSE. The UP best selected model is B; G and H selected policies exactly retain those UP forecasts. The DOWN winner is the earlier E compact MLP, not the later transformed HGB.')
text('If all diagnostic candidates are included, the lowest observed UP MAE is G.candidate: 312.677, RMSE 415.780, IC 0.130. It failed the acceptance gates and is not the best selected policy. DOWN E remains the lowest-MAE forecast even when diagnostic candidates and complete fixed variants are included.')
text('These scores pool 18,193 identical evaluation origins per side. Quarterly models use different features; there is no single static coefficient vector or feature list for the entire evaluation. The next pages specify every quarterly winner and its mathematical prediction rule.')
page('Best UP model | architecture and prediction formula','B: price-block follow-up / HistGradientBoostingRegressor / absolute-error regression')
table(['Item','Exact winning workflow'],[
 ['Architecture','A separate histogram gradient-boosted tree ensemble is fitted for each Jan/Apr/Jul/Oct issue. 80 boosting iterations; depth at most 3; at most 7 leaves per tree; minimum 128 rows per leaf.'],
 ['Hyperparameters','loss=absolute_error; learning_rate=0.05; max_iter=80; max_depth=3; max_leaf_nodes=7; min_samples_leaf=128; l2_regularization=1.0; early_stopping=False; random_state=48.'],
 ['Input vector','Take the ordered quarterly raw feature vector x. Replace missing entries by training medians and append one missing indicator per feature. Center and scale each expanded column by training mean and standard deviation; a standard deviation below 1e-8 becomes 1. Call the result z.'],
 ['Prediction formula','UP_hat(t) = max(0, b0 + 0.05 * sum_{k=1}^{80} T_k(z(t))). Here b0 is the fitted initial prediction and T_k is the kth fitted leaf-update function before learning-rate scaling. In the saved sklearn estimator, learning-rate scaling is already incorporated into stored tree contributions; do not multiply them by 0.05 again.'],
 ['Training objective','Minimize mean absolute error in original log-bps units. Absolute-error boosting starts from a median-like constant and uses fitted piecewise-constant tree updates; this is not one linear regression equation.'],
 ['Model files','Run B / each YYYY-MM_up_magnitude_log_bps/selection.json gives selected_variant and its model_XX.joblib artifact. The artifact contains preprocessing statistics and fitted tree thresholds/leaf values, required for exact numerical reproduction.'],
 ['Scores','MAE 317.631061; RMSE 432.635213; rank IC 0.065508. The exact saved tree policy is used without an additional mean/median blend.']],[130,WIDTH-130])
page('Best UP model | exact quarterly features','The F4Z columns are inherited masked source inputs. Their upstream normalization is not recomputed by this experiment. Names below preserve exact measurements and timeframes.')
table(['Issue / use period','F4Z inputs (prefix F4Z__, suffix __value)','New price inputs','Count'],[
 ['Jan 2024 / Q1','basis__basis_level_rz__15m; basis__basis_change_percentile__1h','Eight-feature price block at 12h and at 168h',18],
 ['Apr 2024 / Q2','basis__basis_level_rz__15m; liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m; oi__oi_range_position_percentile__1h','None',3],
 ['Jul 2024 / Q3','basis__basis_level_rz__15m; basis__basis_persistence_percentile__1h; liquidation__liq_side_pressure_rz__1h; oi__remaining_recent_oi_fraction_percentile__1h','Eight-feature price block at 1h and at 168h',20],
 ['Oct 2024 / Q4 + Jan 2025','liquidation__liq_activity_severity_historical_active_rate__15m; basis__basis_level_rz__15m; funding__funding_level_percentile__15m; price__price_acceleration_rz__5m__15m','None',4]],[110,WIDTH-300,150,40])
text('Exact new-column syntax: NEWPRICE__measurement__{h}h. For every price block, measurement is each of: rv, up_rv, down_rv, net, past_range, distance_from_high, distance_from_low, max_abs_return.')
table(['Measurement','Formula over five-minute returns r strictly before origin'],[
 ['rv; up_rv; down_rv','sqrt(sum r^2); sqrt(sum max(r,0)^2); sqrt(sum min(r,0)^2)'],
 ['net; max_abs_return','sum r; max(abs(r))'],
 ['past_range; distance_from_high; distance_from_low','H-L; H-C; C-L, where H/L are high/low of the local cumulative past log-price path and C is current path level. Complete h-hour windows require 12*h finite returns.']],[210,WIDTH-210])
text('The equations above define the newly engineered price inputs. F4Z fields are consumed as supplied: an exact reconstruction of their upstream RZ/percentile/activity-rate calculations would require the original source implementation, not a guessed formula. Full column names also appear in the feature-name index.','small')
page('Best DOWN model | architecture and prediction formula','E: compact MLP / two-seed ensemble / four raw model inputs plus missing indicators')
table(['Item','Exact winning model'],[
 ['Architecture','Each seed model: 8 input channels (4 values + 4 missing flags) -> Dense(8) -> ReLU -> Dense(1) -> Softplus. Average two independently trained models, seeds 48 and 148. All four quarterly DOWN selections use this small architecture.'],
 ['Training','40 epochs; AdamW learning rate 0.001; weight decay 0.1; batch size 256; gradient norm clip 1; deterministic algorithms. Train-only median imputation, mean/std scaling and input clipping to [-8,8]. Original-unit MAE objective.'],
 ['Neural formula','h_s(t)=ReLU(W1_s*z(t)+b1_s); n_s(t)=Softplus(W2_s*h_s(t)+b2_s), for s in {48,148}. ReLU(a)=max(0,a); Softplus(a)=log(1+exp(a)).'],
 ['Output formula','p(t)=a(t)*u*(n_48(t)+n_148(t))/2; DOWN_hat(t)=w*p(t)+(1-w)*m. Here m=training target median; u=max(median_training(y/a),0.001); w is the quarterly model weight below.'],
 ['Volatility scaling','For Jan/Apr: raw scale a0(t)=RMS_120h(t)*sqrt(120). Set a(t)=max(valid a0(t), floor), with nonfinite a0 replaced by the training median positive scale (finite zero scales are raised to the floor); floor=max(training P10 positive scale,1). Jul/Oct raw mode uses a(t)=1.'],
 ['Numerical target unit','Training divides target and forecast by max(m,1) when computing MAE. This positive constant does not change the data-loss optimum; forecasts are restored to log bps. There is no log transform of the target.'],
 ['Scores and exact parameters','MAE 297.849596; RMSE 436.477133; rank IC 0.133295. Fitted W/b arrays, preprocessing and output scale parameters live in run E / YYYY-MM_down_magnitude_log_bps/model.joblib. These learned numbers cannot be recovered from architecture hyperparameters alone.']],[130,WIDTH-130])
page('Best DOWN model | features, formulas and quarterly weights')
table(['Issue / use period','Four neural-network inputs','Output mode','Model weight w'],[
 ['Jan 2024 / Q1','T1, T2, T3, T4','Volatility scaled',0.75],
 ['Apr 2024 / Q2','T1, T2, T3, T4','Volatility scaled',1.0],
 ['Jul 2024 / Q3','R1, R2, R3, R4','Raw',0.75],
 ['Oct 2024 / Q4 + Jan 2025','R1, R2, R3, R4','Raw',0.75]],[150,240,WIDTH-500,110])
text('Notation: r_j is a completed hourly log-bps return; N_h=sum r_j, V_h=sqrt(sum r_j^2), A_h=sum abs(r_j). Let C be current cumulative hourly log-price level and H_h/L_h its local high/low. Hourly bins and all lookbacks use only past observations.')
table(['ID','Exact column name','Feature formula'],[
 ['T1','LONG__normalized_momentum__120h','N_120 / max(V_120,1e-8)'],
 ['T2','LONG__trend_efficiency__168h','N_168 / max(A_168,1e-8)'],
 ['T3','LONG__drift_change_120__240h','N_120/120 - N_240/240'],
 ['T4','LONG__net__720h','N_720: sum of past 720 completed hourly returns'],
 ['R1','LONG__range_position__120h','(C-L_120) / max(H_120-L_120,1e-8)'],
 ['R2','LONG__drawdown_vol_units__336h','(H_336-C) / max(V_336,1e-8)'],
 ['R3','LONG__range__240h','H_240-L_240'],
 ['R4','LONG__range__1440h','H_1440-L_1440'],
 ['S1','LONG__vol_per_sqrt_hour__120h','V_120/sqrt(120). Additional output-scale input in Jan/Apr; not one of the four neural input columns.']],[35,335,WIDTH-370])
text('Thus Jan/Apr forecasts depend on five distinct raw measurements: T1-T4 plus scale input S1. Jul/Oct use only R1-R4. Missing-feature flags and learned normalization statistics are also part of inference. Range features describe hourly sampled paths, not intrabar extremes.','small')
page('Evaluation and scoring contract')
table(['Item','Definition'],[
 ['Target','Separate nonnegative canonical UP/DOWN path excursions over 7,200 minutes. Conceptually: UP=max(0,max future log-price change); DOWN=max(0,-min future log-price change). Not signed endpoint return. Valid zeros retained.'],
 ['Population',f"BTCUSDT; {data['A'].decision_time.min()} through {data['A'].decision_time.max()}; 18,193 matched origins per side. 47,808 eligible total rows; 288 development exclusions (144 invalid price path, 144 outside release)."],
 ['Timing','Jan/Apr/Jul/Oct 2024 expanding fits; October model retained in January 2025. All training labels mature strictly before the fitting boundary: origin+120h < boundary. Original 48h inventory gap is retained; the 120h purge governs label maturity.'],
 ['MAE','mean(abs(actual-prediction)), in log bps. Lower is better. Compare against BOTH training-median and training-mean MAE.'],
 ['RMSE','sqrt(mean((actual-prediction)^2)), in log bps. Lower is better. Sensitive to large mistakes; compare against training-mean RMSE.'],
 ['Rank IC / bias','IC is Spearman rank correlation; higher is better, but ranking is not magnitude calibration. Bias=mean(prediction-actual); negative means average underprediction.'],
 ['Ranking','Primary ordering by pooled MAE; separate RMSE and IC ranks are supplied. Period scores use matched rows. Partial-coverage variants do not enter the full-period ranking.'],
 ['Verification','Eight prediction SHA256 hashes verified; timestamp, issue, truth and baseline alignment checked; pooled and period metrics independently recomputed and compared with saved metrics. Saved-model replay success is reported by runs, not rerun here.'],
 ['Limitations','Repeatedly inspected historical evaluation; overlapping120h outcomes are dependent. No untouched-test, significance, or future-performance claim. Different stages have different acceptance gates.']],[100,WIDTH-100])
for target in ['UP','DOWN']:
 page(target+' | selected policies and candidates','Ranked by MAE; identical predictions retain tied ranks so workflow history is visible. BASE entries are controls.')
 rows=[]
 for _,r in pol[pol.target==target].iterrows():rows.append([int(r.MAE_rank),r.run+'.'+r.forecast,r.role,fmt(r.MAE),fmt(r.RMSE),fmt(r.IC),int(r.RMSE_rank),int(r.IC_rank),fmt(r.Bias)])
 table(['MAE rank','Forecast','Status','MAE','RMSE','IC','RMSE rank','IC rank','Bias'],rows,[48,185,65,62,62,55,64,56,WIDTH-597])
 text('Selected policies can switch architecture or retain an earlier model by quarter. These rows rank complete forecast policies, not isolated network types. A model trained only during inner validation has no full-period evaluation score.')
for target in ['UP','DOWN']:
 entries=full[full.target==target]
 for k,part in enumerate(chunks(entries.to_dict('records'),12)):
  page(target+' | all complete saved forecasts'+(f' ({k+1})' if k else ''),'Includes fixed feature variants from B. Identical predictions are consolidated; aliases are in the supporting CSV. Diagnostic rows are not selected policies.')
  table(['MAE rank','Forecast','Role','MAE','RMSE','IC','RMSE rank','IC rank'],[[int(r['MAE_rank']),r['run']+'.'+r['forecast'],r['role'],fmt(r['MAE']),fmt(r['RMSE']),fmt(r['IC']),int(r['RMSE_rank']),int(r['IC_rank'])] for r in part],[45,245,85,65,65,55,60,WIDTH-620])
page('Period results | best selected policies','UP: B price-block follow-up. DOWN: E compact MLP. Quarterly/January rows use exactly the matched forecast origins.')
rows=[]
for target,label in [('UP','B'),('DOWN','E')]:
 f=data[label];f=f[f.target.str.startswith(target.lower())]
 for period,g in f.groupby('period'):
  m=metrics(g.actual.to_numpy(),g[PRIMARY[label]].to_numpy());med=metrics(g.actual.to_numpy(),g.training_median.to_numpy());mean=metrics(g.actual.to_numpy(),g.training_mean.to_numpy())
  rows.append([target,period,len(g),fmt(m['MAE']),fmt(med['MAE']),fmt(mean['MAE']),fmt(m['RMSE']),fmt(mean['RMSE']),fmt(m['IC'])])
table(['Side','Period','Rows','Model MAE','Median MAE','Mean MAE','Model RMSE','Mean RMSE','IC'],rows,[45,65,55,90,90,90,90,90,WIDTH-625])
page('Architectures | original trees and compact MLP')
table(['Stage / family','Architecture, objective and hyperparameters'],[
 ['A / D: HGB','HistGradientBoostingRegressor; absolute_error;80iterations,rate0.05,7leaves,depth3,minleaf128,L2=1,seed48,no random early stopping. Training imputation, standardization and missing flags. All saved A/D winners are HGB. Ridge(alpha0.01*n) was a search alternative, not a saved winner. Shallow absolute-error decision trees (depth2,leaf128) were used in screening.'],
 ['B: price-block HGB','Same original HGB settings. Earlier validation selects additions/removals or unchanged original. Original F4Z feature list is retained as anchor; past-price blocks can be added. No neural network here.'],
 ['C: tuned HGB','Absolute-error HGB,rate0.05,seed48,native NaN,early_stopping=False. Small:120iterations/4leaves/depth2/leaf128/L2=10. Medium:160/7/depth3/leaf96/L2=5. Flexible:200/15/depth4/leaf64/L2=10. Full or365day history; pure or50% blend with B.'],
 ['E: compact MLP','4raw features+4missing flags -> either8ReLU ->1Softplus, or16ReLU ->8ReLU ->1Softplus. Small40epochs;two_layer60epochs. AdamW lr0.001,decay0.1,batch256,gradientnormclip1;seed48 and148 models averaged. MAE in original log-bps units; train-only imputation/standardization/inputclip8.'],
 ['E: output scaling','Raw mode uses factor1. Volatility mode uses past120h hourly RMS*sqrt(120), floored at training P10 (minimum1) and filled with training median scale. Softplus output multiplied by scale and training output unit before MAE loss. Blend weights0.25/0.5/0.75/1 with training median. No extra log transform of the target.']],[125,WIDTH-125])
page('Architectures | market HGB, MLP and GRU')
table(['Stage F','Configuration'],[
 ['HGB','Absolute-error histogram booster;60iterations,depth3,7leaves,minleaf128,rate0.05,L2=10,maxbins63,seed48,no early stopping.6000evenly spaced estimator rows;full eligible training data for preprocessing and median.'],
 ['MLP16','For d raw pack inputs:2d inputs including missing flags ->16ReLU ->1head. AdamW lr0.002,decay0.1.'],
 ['MLP32/16','2d inputs ->32ReLU ->16ReLU ->1head. AdamW lr0.001,decay0.03.'],
 ['GRU8 / GRU16','4snapshots at30/20/10/0hours,2d channels per step including flags;single GRU layer(hidden8 or16) ->linear head. GRU8 lr0.002/decay0.1;GRU16 lr0.001/decay0.03.'],
 ['Shared neural training','24epochs,batch512,AdamW,gradientclip1,seed48,deterministic algorithms,4096evenly spaced estimator rows. All training rows supply preprocessing and median. ReLU for MLP;GRU uses its standard gated recurrent cell.'],
 ['Output and blending','median +0.75*max(median,1)*tanh(head), then blend adjustment weight0.1/0.25/0.5/1. MAE training. Prediction rows with<80%finite required inputs revert to median.'],
 ['Sequence construction','Exact historical timestamp lookups on the full X timeline before invalid-label filtering;no filling. Each of the four snapshot channels uses the same feature pack. Neural packs require all inputs>=80%coverage and1000finite observations in training.']],[135,WIDTH-135])
page('Architectures | joint objectives and transformations')
table(['Stage','Configuration'],[
 ['G: paired HGB','Two models per pack/config:absolute_error andsquared_error. Shallow:depth2/4leaves/leaf256/60iterations. Regularized:depth3/7leaves/leaf128/80iterations. Both rate0.05,L2=10,maxbins63,seed48,no early stopping;ALL mature rows.'],
 ['G/H: two blends','Pair forecast=(1-m)*L1+m*L2, m in0/0.25/0.5/0.75/1. Final=(1-u)*original anchor+u*pair, u in0.25/0.5/0.75/1. Anchors:UP B,DOWN D. An anchor is refitted inside each validation training window. Outer fallback is exact saved anchor.'],
 ['H: four linear pairs','One pair for each raw/signed_log/rank/quadratic input basis. L1 side:SGDRegressor epsilon_insensitive,epsilon0,elasticnet alpha0.001,l1_ratio0.1,adaptive eta0.005,max1500iterations,tol1e-5,n_iter_no_change15,average=True,seed48,no random early-stopping split. L2 side:Ridge alpha0.01*n.'],
 ['H: quadratic HGB pair','Two losses on quadratic basis;depth2/4leaves/leaf256/60iterations,rate0.05,L2=10,maxbins63,seed48. Target divided by max(training median,1) and restored before scoring. Input transformations do not imply calibrated conditional medians/means.'],
 ['G/H: final selection','Four earlier purged windows. Require>=0.1%pooled improvement versusmedian MAE ANDmean RMSE,no pooled degradation versusanchor in either metric,3/4wins/ties each,no>2%windowloss. Minimize worst relativeMAE/RMSE ratio. Otherwise unchanged anchor;best candidate separately saved.'],
 ['G/H: missingness','Per-fit features need80%finite,1000observations,andvariation. Training imputation/flags. If<80%prediction inputs finite, useanchor. No target clipping,tail trimming or outcome log transform.']],[135,WIDTH-135])
page('How price features were made | five-minute paths')
text('Let r be canonical five-minute log-bps returns. Every window excludes the forecast-origin endpoint. The local cumulative log-price path is built from past returns; its high/low refer to sampled closes, not exchange intrabar highs/lows. A complete h-hour window requires12*h finite returns.')
table(['NEWPRICE measurement','Formula / meaning'],[
 ['rv / up_rv / down_rv','sqrt(sum r^2); sqrt(sum max(r,0)^2); sqrt(sum min(r,0)^2).'],
 ['net','sum r over the past window.'],['past_range','maximum minus minimum local cumulative log-price over the window.'],
 ['distance_from_high / low','past high minus current level;current level minus past low.'],['max_abs_return','max(abs(r)) over the window.'],
 ['rv_ratio','(rv_short/sqrt(short))/(rv_long/sqrt(long)), denominator floored1e-6. Pairs1/24,4/24,24/168hours.'],
 ['Window sets','B:1,4,12,24,48,168hours =>48measurements+3ratios=51. C adds120hours =>59. Names use NEWPRICE__measurement__hH,with hH replaced by e.g.168h.']],[175,WIDTH-175])
text('Example: hourly log-bps returns [10,-20,30] have net20 and realized volatility sqrt(100+400+900)=37.417. Positive volatility=sqrt(100+900)=31.623;negative volatility=20. This illustrates the formula, not a real-data result.')
page('How long features were made | hourly aggregation')
text('Hourly bin labelled t contains five-minute endpoints in[t-1h,t). Exactly12finite returns required. Half-hour origins use only the latest completed hour (maximum59m59s old). No backfilling. Complete rolling windows are required. Lookbacks24/60/120/168/240/336/720/1440hours yield96measurements plus8cross-window contrasts=104features.')
table(['LONG measurement','Formula'],[
 ['net;vol_per_sqrt_hour','sum hourly returns; sqrt(sum hourly returns^2 / h).'],['up_variance_share','sum positive-hour returns^2 / sum all-hour returns^2; denominatorfloor1e-16.'],
 ['trend_efficiency;normalized_momentum','net/sum(abs(hourly return)); net/sqrt(sum hourly return^2). Denominatorfloor1e-8.'],
 ['range;range_position','local cumulative hourly-price max-min;(current-low)/range.'],['drawdown_vol_units;rebound_vol_units','(high-current)/realized volatility;(current-low)/realized volatility.'],
 ['positive_hour_share;hourly_skew','fraction of positive hourly returns;sample skewness of hourly returns.'],['jump_variance_share','largest squared hourly return / sum squared hourly returns. It is concentration in one hour, not a jump-event label.'],
 ['vol_ratio_short__long','(RMS_short/RMS_long),floor1e-8. Pairs24/120,120/240,168/720,336/1440hours.'],
 ['drift_change_short__long','net_short/short - net_long/long for the same four pairs. Range/volatility divisions usefloor1e-8.']],[200,WIDTH-200])
page('How market and sequence features were made')
text('Source snapshots are the existing masked F4Z inputs. This report does not claim to reconstruct their vendor preprocessing. Funding,basis and OI fields are positions/percentiles; liquidation history fields are activity rates, not current liquidation size. No availability rules are inferred from names alone.')
table(['Construction','Definition'],[
 ['11source families','funding_level,funding_change,basis_level,basis_change,oi_position,oi_unwind,flow_pressure,flow_activity,liq_history,peer_liq_history,liq_breadth. Exact F4Z names are archived in market_state_features_120h.py.'],
 ['STATE windows','Lag30min plus6/24/120/168hour mean,populationstd,lag-minus-mean deviation,and fraction>=0.8. All rolling windows closed on the left,>=80%expected half-hour coverage,no filling. Generated187state columns.'],
 ['INTERACT windows','24/120/168h products:funding_mean*OI_mean;OI_mean*(1-flow_pressure_mean);OI_unwind_mean*max(-normalized_momentum,0);liq_history_mean*price_RMS;(basis_mean-0.5)*(funding_mean-0.5);(flow_pressure_mean-0.5)*normalized_momentum.18generatedcolumns.'],
 ['Focused price core','120h RMS,normalized momentum,range position,up-variance share;168h drawdown-vol units;336h jump-variance share.'],
 ['Focused packs','Leverage38:6price+6funding/basis/OIfamilies x2stats(mean/deviation)x2windows(24/120h)+4interactionsx2windows. Flow30:6price+5flow/liquidationfamiliesx2statsx2windows+2interactionsx2windows. Combined62 is the union.'],
 ['GRU sequence','Repeat the same pack at exact origins minus30,20,10,0hours,oldest first. SEQ{lag}__ prefixes identify the snapshot. Each input adds a missingness channel.'],
 ['Dual feature packs','Original per-quarteranchor features;12compactLONG features;union of those12 with focusedcombined62 yields69distinct inputs. Feature quality can remove columns inside a fit.']],[140,WIDTH-140])
page('Transformations | exact training-only operations')
table(['Basis / operation','Calculation'],[
 ['Imputation','Replace nonfinite inputs with training medians;append missing flags. Scale raw values by training IQR;zero/tiny IQR becomes1.'],
 ['Raw control','z=(x-median)/IQR;clip z to[-8,8]. This is affine standardization plus clipping, not the original unscaled input.'],
 ['Signed log','sign(z)*log(1+abs(z));clip[-8,8]. Example:z=-3 becomes-log4=-1.386. Compresses extreme input values while keeping sign.'],
 ['Rank','2*training empirical midrank CDF(x)-1. Ties use average left/right insertion positions divided by training count. Example:training[1,1,3,5] maps to[-0.5,-0.5,0.25,0.75].'],
 ['Quadratic','Signed-log transformed values plus their individual squares;then missing flags. No full pairwise interaction expansion. A signed-log value-1.386 contributes squared term1.922.'],
 ['Final scaling','Learn each basis column mean/std on training data only;tiny std becomes1;standardize andclip[-8,8]. Stored statistics are reused unchanged for prediction.'],
 ['Model dimensions','Linear raw/log/rank uses2d columns (values+flags). Quadratic uses3d (values+squares+flags). Linear prediction is intercept+sum coefficient*basis. HGB can split on these columns.'],
 ['Target stays log bps','H uses only positive constant target scaling,not log1p(y),rank(y),or tail clipping. Forecasts restored to logbps and clipped nonnegative. Transformations were not fitted using evaluation-batch statistics.']],[145,WIDTH-145])
for label,(name,relative) in RUNS.items():
 rows=[r for r in selection_rows if r['run']==label]
 for k,part in enumerate(chunks(rows,8)):
  page(label+' | '+name+f' / quarterly decisions {k+1}','P identifiers expand in the feature-pack appendix. F identifiers map to exact column names. October applies to Q4 and January2025.')
  table(['Side','Issue','Forecast','Model / configuration','Settings','Inputs','Feature pack'],[[r['target'],r['issue'],r['forecast'],r['model'],r['settings'] or '--',r['inputs'],r['pack']] for r in part],[40,65,60,150,215,40,WIDTH-570])
  if label in 'GH':text('Features shown for a correction policy are the union of anchor inputs and admitted correction inputs. The machine-readable selection plan also retains proposed feature lists; final-quality files identify actual admission. mean_weight=0 uses only the L1 correction; update_weight=1 replaces anchor prediction on usable rows.','small')
  if label=='E':text('small=8ReLU/40epochs;two_layer=16/8ReLU/60epochs. Both use two seeds and Softplus output. All DOWNquarters select small;Jan/Apr trend pack with volatility-scaled output,Jul/Oct range pack with raw output.','small')
page('Search and selection rules | what the scores mean')
table(['Stage','Earlier-data selection rules'],[
 ['A','Empty-start522maskedF4Z universe;training coverage/variation,screening/redundancy shortlist and bounded forward/removal search;Ridge orMAEHGB. Three earlier windows;0.1%pooled medianMAEgain and2wins;otherwisemedian.'],
 ['B','Test past-price additions (up to2blocks),all-price+original,price-only and removals. Earlier-validation MAE improvement versusoriginal with2wins. Fixed alternatives evaluated only after selection;they remain diagnostics.'],
 ['C','Three packs x3HGBsettings x full/365dayhistory x pure/halfblend. Require0.1%MAEgain and2wins versuscurrent andmedian;pooledMSE no worse thancurrent. OtherwiseB.'],
 ['D','104newLONGfeatures only,empty-start. Training-quality screen,top20,max6features,3earlier windows,Ridge/HGB;0.1%medianMAEgain and2wins elsemedian.'],
 ['E','Four4featurepacks x2architectures x raw/volatility output x4medianweights. Three earlier windows;0.1%MAEgain,2wins,no>2%windowloss vsmedian;elsemedian.'],
 ['F','Threepacks x5configurations x4blendweights;four earlier windows;0.1%medianMAEgain,3wins/ties,no>2%windowloss. Candidate is lowestvalidationMAE without gate.'],
 ['G/H','Both criteria and anchor guard as described in architectures. Candidate minimizes worst relative score even if final gates fail. Validation ratios do not imply the same later-data gains.']],[65,WIDTH-65])
text('There is no retrospective search for a single globally best network using the evaluation period. Rankings describe saved forecasts. Earlier feature anchors were themselves chosen through historical experiments, so the full development process is adaptive.')
packrows=[]
for (names,seq),pid in packs.items():packrows.append([pid,len(names),'30/20/10/0h' if seq else 'Current snapshot',', '.join(fid[c] for c in names)])
for k,part in enumerate(chunks(packrows,10)):
 page(f'Feature-pack appendix | {k+1}','Exact ordered base-feature IDs. For sequence packs, repeat the list at each lag, then supply matching missing flags. Other model preprocessing is documented earlier.')
 table(['Pack','Base inputs','Snapshot lags','Exact ordered feature IDs'],part,[45,65,110,WIDTH-220])
for k,part in enumerate(chunks(feature_names,20)):
 page(f'Exact feature-name index | {k+1}','Column names are preserved from saved selections. F4Z columns are inherited inputs; NEWPRICE/LONG/STATE/INTERACT columns follow the documented engineering formulas.')
 table(['ID','Exact column name'],[[fid[c],c] for c in part],[50,WIDTH-50],codecol=1)
partialrows=[]
for r in partial:
 partialrows.append([r['target'],r['forecast'],r['rows'],'Not ranked against full-period models'])
for k,part in enumerate(chunks(partialrows,17)):
 page('Partial variants | coverage'+(f' ({k+1})' if k else ''),'These B variants were not produced in every quarter. Their numerical scores are available by matching period in all_forecasts_period_metrics.csv, not pooled into the full ranking.')
 table(['Target','Variant','Available origins','Treatment'],part,[50,300,100,WIDTH-450])
page('Fixed variant definitions and evidence map')
table(['B variant','Feature / model interpretation'],[
 ['original','Per-quarter A feature/model card.'],['add_price_h','A features +all8NEWPRICEmeasurements at hhours.'],['add_volatility_ratios','A features +3volatility ratios.'],
 ['add_price_a+price_b','A features +both8featureblocks.'],['all_price_plus_original','A features +all51past-price inputs.'],['price_only','All51past-price inputs withoutA.'],['remove_original_features','Selected forward-added blocks withoutA.'],['remove_price_h','Selected forward pack with indicated price block removed. Other included features are retained.'],['selected','Earlier-selected variant;may differ by target andquarter. Exactcards/validationtrials are included in the supporting archive.']],[180,WIDTH-180])
text('Full rankings include only variants with18,193finite forecasts for that target. Consolidated aliases and exact per-quarter variant inputs are in the supporting archive. All selected/candidate inputs are printed in this PDF via pack and feature IDs. Unselected inner-only models have validation scores, not an invented evaluation ranking.')
for k,part in enumerate(chunks(audit,4)):
 page('Run provenance'+(f' | {k+1}' if k else ''),'All paths below are relative to Week2/Code/experiments. Prediction hashes are recorded in full in source_audit.json.')
 table(['ID','Experiment','Source path','Hash / replay'],[[r['run'],r['name'],r['path'],'Hash verified; replay reported '+str(r['replay_reported'])] for r in part],[35,145,WIDTH-300,120])
 text('Supporting folder: Report/120h_report_support_2026-11-14. Includes policy/full-coverage rankings,all forecast period metrics,exactquarterlyinputs,allvalidationtrials,savedcontracts,selectioncards,andSHA256provenance. The adjacent ZIP is a portable copy. No raw datasets or large model binaries are modified or repackaged.')
 text('Audit correction: the earlier MLP result was absent from recent best-model summaries. E is now included and is the strongest selected DOWN policy by MAE among the completed runs audited. This correction does not establish that its RMSE beats the mean baseline. The report does not select models for live deployment.')
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#c9d6df'));canvas.line(36,32,W-36,32)
 canvas.setFont('Helvetica',8);canvas.setFillColor(navy);canvas.drawString(36,20,'BTC 120h | Saved research evidence | 14 Nov 2026');canvas.drawRightString(W-36,20,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=43,title='BTC 120h models, features and rankings',author='Research report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF);texts=[p.extract_text() for p in reader.pages]
assert all(len(t)>100 for t in texts)
assert '297.850' in ''.join(texts) and '317.631' in ''.join(texts)
(SUP/'source_audit.json').write_text(json.dumps(dict(runs=audit,files=hashes),indent=2),encoding='utf-8')
(SUP/'REPORT_CHECKS.json').write_text(json.dumps(dict(pages=len(reader.pages),prediction_hashes_verified=8,origins_per_target=18193,metrics_recomputed=True,no_training=True,feature_count=len(feature_names),packs=len(packs)),indent=2))
zip_path=OUT/'BTC_120h_Report_Support_2026-11-14.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in SUP.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(SUP))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(zip_path),full_rank_counts=full.groupby('target').size().to_dict(),features=len(feature_names),packs=len(packs))))
print(pol[['target','run','forecast','MAE','RMSE']].to_string(index=False))
