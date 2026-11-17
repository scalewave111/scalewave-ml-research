"""Audit and report saved300h net refined quarterly policy; no fitting or inference."""
from pathlib import Path
import json,hashlib,shutil
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'Code/experiments';OUT=ROOT/'Report'
SUP=OUT/'net_magnitude_300h_report_support_2026-11-16';SUP.mkdir(exist_ok=True)
PDF=OUT/'BTC_300h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
RUNS={'original':EXP/'net_magnitude_300h_downstyle_v1/20261116T111254635059Z','refined':EXP/'net_magnitude_300h_refined_v1/20261116T112350667836Z'}
hashes={};data={};contracts={};plans={};cards={};scores=[];model_count=0;trial_count=0;reference=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(f,col):
 e=f[col].to_numpy(float)-f.actual.to_numpy(float)
 return dict(rows=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),median_absolute_error=float(np.median(np.abs(e))),bias=float(e.mean()))
for key,path in RUNS.items():
 done=js(path/'COMPLETE.json');assert done['status']=='COMPLETE' and sha(track(path/'predictions.csv'))==done['prediction_sha256']
 f=pd.read_csv(path/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True);f=f.sort_values('decision_time').reset_index(drop=True)
 assert len(f)==16873 and f.decision_time.is_unique and np.isfinite(f[['actual','selected','candidate','mean','median']]).all().all()
 if reference is None:reference=f[['decision_time','actual','mean','median']]
 else:pd.testing.assert_frame_equal(reference,f[['decision_time','actual','mean','median']])
 data[key]=f;contracts[key]=js(path/'CONTRACT.json');plans[key]=js(path/'selections.json');cards[key]=[]
 saved=pd.read_csv(track(path/'metrics.csv'))
 for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
  for col in ['selected','candidate','mean','median']:
   m=metric(g,col);r=saved[(saved.period==period)&(saved.model==col)].iloc[0]
   assert all(abs(m[v]-float(r[v]))<1e-5 for v in ['mae','rmse','median_absolute_error'])
   scores.append(dict(run=key,forecast=col,period=period,**m))
 for entry in plans[key]:
  folder=path/pd.Timestamp(entry['issue']).strftime('%Y%m');trials=js(folder/'validation_trials.json');trial_count+=len(trials)
  eligible=[v for v in trials if v['eligible']]
  assert bool(eligible)==entry['plan']['accepted']
  if eligible:assert np.isclose(entry['plan']['selected']['validation_mae'],min(v['validation_mae'] for v in eligible))
  assert np.isclose(entry['plan']['candidate']['validation_mae'],min(v['validation_mae'] for v in trials))
  boundaries=js(folder/'fit_boundaries.json')
  assert all(pd.Timestamp(v['max_training_label_end'])<pd.Timestamp(v['cutoff'])<=pd.Timestamp(v['validation_start']) for v in boundaries)
  for name in ['validation_trials.json','quality.json','fit_boundaries.json','selection.json']:shutil.copyfile(track(folder/name),SUP/(key+'_'+folder.name+'_'+name))
  for record in entry['models']:
   file=path/record['artifact'];assert sha(track(file))==record['sha256'];model_count+=1
   a=joblib.load(file);assert a['features']==record['features']
   card=dict(issue=entry['issue'],**record)
   card.update(preprocessing_shapes=[list(v.shape) for v in a['prep']] if 'prep' in a else None)
   if 'constant' in a:card['constant']=a['constant']
   elif a['backend']=='hgb':card.update(parameters=a['model'].get_params(),iterations=a['model'].n_iter_,median=a['median'],anchor=a['anchor'],estimator_rows=a['estimator_training_rows'])
   else:card.update(config=a['config'],tensor_shapes={k:list(v.shape) for k,v in a['state'].items()},parameters=sum(v.size for v in a['state'].values()),median=a['median'],unit=a['unit'],anchor=a['anchor'],estimator_rows=a['estimator_training_rows'])
   cards[key].append(card)
 for name in ['CONTRACT.json','selections.json','feature_dictionary.csv','population_audit.csv']:shutil.copyfile(track(path/name),SUP/(key+'_'+name))
assert sha(RUNS['original']/'predictions.csv')==js(RUNS['refined']/'COMPLETE.json')['previous_prediction_sha256']
np.testing.assert_allclose(data['refined'].previous_selected,data['original'].selected,atol=1e-8,rtol=0)
active=[a for a in cards['refined'] if a['policy']=='selected' and a['features']]
used=sorted(set(name.split('__',1)[1] if name.startswith('SEQ') else name for a in active for name in a['features']))
assert len(used)==38
formulas=pd.read_csv(RUNS['refined']/'feature_dictionary.csv').fillna('');formulas=formulas.set_index('feature')
exact=[]
for name in used:
 row=formulas.loc[name];exact.append(dict(feature=name,formula=row['formula'],sources=row.get('sources',''),April='yes',July='yes'))
pd.DataFrame(exact).to_csv(SUP/'selected_38_features_formulas.csv',index=False)
pd.DataFrame([dict(issue=a['issue'],policy=a['policy'],feature=n) for a in active for n in a['features']]).to_csv(SUP/'exact_fitted_inputs.csv',index=False)
dump(SUP/'model_metadata.json',cards)
dump(SUP/'selected_policy.json',dict(selected_run=str(RUNS['refined']),selection_basis='User selected refined quarterly GRU/HGB policy after exposed historical comparison',quarterly_rules=plans['refined'],active_models=active))
pd.DataFrame(scores).to_csv(SUP/'period_metrics.csv',index=False)
NAVY=colors.HexColor('#17354A');TEAL=colors.HexColor('#087E8B');LIGHT=colors.HexColor('#EDF4F7')
styles={
    'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=26,leading=31,textColor=NAVY,spaceAfter=14),
    'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=12),
    'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=TEAL,spaceBefore=8,spaceAfter=6),
    'body':ParagraphStyle('body',fontName='Helvetica',fontSize=12,leading=16,spaceAfter=9),
    'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=11,leading=14,splitLongWords=True),
    'small':ParagraphStyle('small',fontName='Helvetica',fontSize=10.5,leading=13,splitLongWords=True),
}
story=[]
def para(t,kind='body'):return Paragraph(escape(str(t)).replace('\n','<br/>'),styles[kind])
def p(t,kind='body'):story.append(para(t,kind))
def page(title):
    if story:story.append(PageBreak())
    p(title,'h')
def table(headers,rows,widths,small=False):
    body=[[para(c,'small' if small else 'cell') for c in headers]]+[[para(c,'small' if small else 'cell') for c in row] for row in rows]
    t=Table(body,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,TEAL)]))
    # Paragraph text colour must be set explicitly inside dark header cells.
    for j,h in enumerate(headers):body[0][j]=Paragraph('<font color="white">'+escape(str(h))+'</font>',styles['cell'])
    t._cellvalues[0]=body[0]
    story.append(t)



page('300-hour net magnitude | selected refined policy')
p('BTCUSDT - GRU/HGB models, features and rankings','title')
p('16 November2026 | 16,873 matched historical evaluation origins | log bps')
p('Target: net_magnitude_log_bps=10,000 x abs[ln(P(t+300h)/P(t))]. This is absolute endpoint return, not maximum DOWN excursion or UP minus DOWN.')
p('Selected: the refined quarterly GRU/HGB policy. Pooled MAE444.847768 log bps is0.65% below median and1.96% below mean. It improves the previous selected445.150944 by0.068%. RMSE621.203552 remains worse than the mean582.187424.')
rank=[]
for key,col,label in [('refined','selected','Selected refined policy'),('original','selected','Previous selected policy'),('refined','median','Median baseline'),('refined','mean','Mean baseline')]:
 m=metric(data[key],col);rank.append([label,m['mae'],m['rmse'],m['median_absolute_error']])
rank.sort(key=lambda r:r[1]);pd.DataFrame(rank,columns=['policy','mae','rmse','median_absolute_error']).to_csv(SUP/'selected_policy_rankings.csv',index=False)
table(['Policy / baseline','MAE','RMSE','Median absolute error'],[[r[0]]+[f'{v:.6f}' for v in r[1:]] for r in rank],[290,155,155,155])
p('This is one quarterly selected policy that uses GRU, HGB and constant fallbacks in different periods. It is not a single pure-model forecast. User selection follows historical exposed evaluation, not independent confirmation.')

page('Selected policy | exact issue rules')
rows=[]
for entry in plans['refined']:
 issue=pd.Timestamp(entry['issue']);spec=entry['plan']['selected'];g=data['refined'];g=g[g.issue==issue];m=metric(g,'selected')
 rule='100% mature training median' if spec['model']=='median' else 'GRU16 with no external baseline blend' if spec['config'].startswith('gru') else '60% HGB +40% mature training median'
 rows.append([issue.strftime('%b%Y'),rule,0 if spec['model']=='median' else 38,f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
table(['Issue','Forecast rule','Base inputs','MAE','RMSE'],rows,[90,355,90,110,110])
p('April GRU16 uses leverage38 at exact90/60/30/0hour snapshots. July HGB uses the current leverage38 snapshot. Both add missing indicators. January/October selected forecasts consume no learned-model features. October extends through finalJanuary2025 origins.')
p('Refinement retained the original model configurations and feature pack. July changed from50% to60% model weight. The shallower HGB and higher-decay GRU alternatives were not selected.')

page('Selected HGB | architecture, loss and settings')
a=next(v for v in active if v['spec']['config']=='hgb_l1')
p('July: histogram gradient boosting with absolute-error loss. Additive tree model p_HGB=max(0,initial_constant+0.05*sum(tree_outputs)). Final p=0.6*p_HGB+0.4*training_median, subject to input-coverage fallback.')
p('max_depth=2;max_leaf_nodes=4;min_samples_leaf=256;max_iter=60;learning_rate=0.05;l2_regularization=10;max_bins=63;early_stopping=False;random_state=48. Actual saved iteration count='+str(a['iterations'])+'.')
p('Uses38base values plus38missing indicators:76transformed inputs. Preprocessing uses all mature training rows; estimator fits at most3,000evenly spaced rows. Target is unscaled net magnitude in log bps.')
table(['Training rows','Estimator rows','Mature training median','Baseline blend'],[[a['training_rows'],a['estimator_rows'],f'{a["median"]:.6f}','60%model/40%median']],[180,180,220,175])
p('Training preprocessing: impute each base value with its training median; append one missing flag; standardize each resulting column using training mean and population std. Std<1e-8 becomes1. HGB branch does not clip standardized inputs. HGB admits base features with>=80%finite coverage,>=1,000finite readings,and variation.')

page('Selected GRU16 | architecture and training')
a=next(v for v in active if v['spec']['config']=='gru16_l1')
p('April: four snapshots x76inputs -> one GRU layer(hidden16) -> Linear(16,1) ->0.75*tanh. The final linear weights/bias start at zero. GRU gates follow PyTorch standard reset/update/candidate equations; last sequence hidden state enters the output head.')
p('Let m=training median and u=max(m,1). Train normalized residual(y-m)/u with MAE/L1. Model p=max(0,m+u*0.75*tanh(head(h_last))). The learned correction is bounded to+/-0.75u. This median-centered bounded residual is intrinsic to the GRU; external model weight in April is1.')
p('hidden16;one layer;unidirectional;12epochs;AdamWlr0.001;weight_decay0.03;batch512;gradientnormclip1;seed48;deterministic shuffled minibatches;CPUfloat32. At most2,048evenly spaced estimator rows; full mature data fit preprocessing and baseline statistics.')
table(['Training rows','Estimator rows','Parameters','Median / residual unit'],[[a['training_rows'],a['estimator_rows'],a['parameters'],f'{a["median"]:.6f} / {a["unit"]:.6f}']],[180,180,160,235])
p('152exact sequence-value names (38x4) plus152missing flags are standardized with training mean/populationstd,then clipped to[-8,8] and reshaped into4x76. All sequence columns must have>=80%finite training coverage and>=1,000readings; otherwise the pack returns a constant median. Exact shapes and feature order are in support.')

page('Missing inputs, preprocessing and blending equations')
p('Every model is fit only on labels ending strictly before its cutoff. Training-only column medians, standardized-value centers/scales, missing-indicator centers/scales and baseline anchors remain fixed during that forecast period.')
p('For an anchor meanweight q, anchor=(1-q)*median_train+q*mean_train. External prediction=w*p_model+(1-w)*anchor. Refined q grid0/.25/.5/.75/1;w grid.05/.1/.25/.4/.5/.6/.75/.9/1. Original q0/.5/1 andw.05/.1/.25/.5/.75/1.')
p('Inference coverage requires>=80%finite retained inputs. Otherwise the forecast returns its training anchor. GRU also checks its sequence inputs; its internal median correction is disabled for insufficient coverage. Constant fallback uses mature training median. No future values are filled backward.')
p('MAE=mean(abs(p-y));RMSE=sqrt(mean((p-y)^2));medianAE=median(abs(p-y));bias=mean(p-y). All metrics stay in logbps. No simple-bps or percent conversion is made.')

page('Time split, canonical population and selection')
audit=contracts['refined']['audit']
p('Canonical BTCUSDT horizon_minutes=18000;labels are checksum verified by the original run. Saved audit:48,096original origins,46,488eligible rows,16,873evaluation origins;1,008missing/invalidlabels and600source-binding crossings excluded. Original masked half-hour X grid retained; not all canonical-minute observations.')
p('At Jan/Apr/Jul/Oct2024 issues, use four earlier three-month validation windows starting12/9/6/3months before issue. Training label_end=cutoff is excluded;only label_end<cutoff trains. Validation labels must mature strictly before their window end. GRU sequences are built on the full X timeline before label eligibility filtering.')
p('Select minimum row-count-weighted pooled validation MAE among eligible candidates. Eligibility:>=0.1%pooled MAE improvement against BOTH mean/median,>=3/4window wins/ties against the stronger windowbaseline,and no window>2%worse. Otherwise select median. RMSE andmedianAE are reported,not gate objectives.')
p('A separate diagnostic candidate minimizes earlier-validation MAE without the acceptance gate. It may achieve lower later MAE yet remain unaccepted. Selection is frozen before scoring outer outcomes; no diagnostic forecast is automatically promoted.')

page('Original search and focused refinement')
table(['Stage','Features / models','Inner fits','Changes'],[['Original','leverage38,flow30,combined62;HGB/GRU16 L1',96,'Broader feature-pack comparison.'],['Refined','leverage38;two HGB and two GRU settings',64,'Original controls;depth1/2leaf HGB;decay.1 GRU16;finer blends.']],[95,275,85,300])
p('Original accepted choices used leverage in both learned periods. Refinement narrows to that observed feature group, an adaptive historical decision. It includes the original HGBdepth2/4leaves and GRU16decay.03 controls;the alternatives retain other training settings. Blend trials reuse fitted predictions.')
p('The64innerfits are1pack x4configurations x4windows x4issues. Up to8outerpolicy artifacts are saved per run(selected/candidate in each issue),including constants and potentially repeated configurations. Fewer fits is not an independently benchmarked runtime promise.')
rows=[]
for key in RUNS:
 for col in ['selected','candidate']:
  m=metric(data[key],col);rows.append([key,col,f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}',f'{m["median_absolute_error"]:.6f}'])
table(['Run','Policy','MAE','RMSE','MedianAE'],rows,[100,150,170,170,165])
p('Refined diagnostic437.168257MAE is lower than selected444.847768,but is not accepted in all issues. Keep it separate from the selected-policy ranking. Neither forecast beats meanRMSE.')

for start in [0,2]:
 page('Quarterly error comparison | '+('January / April' if start==0 else 'July / October'))
 rows=[]
 for issue in sorted(data['refined'].issue.unique())[start:start+2]:
  for key,col,label in [('refined','selected','Refined selected'),('original','selected','Previous selected'),('refined','candidate','Diagnostic candidate'),('refined','mean','Mean'),('refined','median','Median')]:
   g=data[key];g=g[g.issue==issue];m=metric(g,col);rows.append([pd.Timestamp(issue).strftime('%b%Y'),label,len(g),f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}',f'{m["median_absolute_error"]:.4f}'])
 table(['Issue','Policy','Rows','MAE','RMSE','MedianAE'],rows,[90,205,70,130,130,130])

page('Selected feature groups | all38 base inputs')
table(['Group','Count','Contents'],[['Price',6,'300hvolatility,normalizedmomentum,rangeposition,upvariance share;168hdrawdown;336hjump share.'],['Market states',24,'Funding level/change,basis level/change,OIposition,leverageunwind:mean/deviation over24h and300h.'],['Interactions',8,'FundingOI,OI-lowflow,unwind-negative momentum,basis-funding;24h and300h each.']],[140,65,550])
p('April and July use the same38base definitions;April samples each at90/60/30/0h. July samples current t only. Missing flags are preprocessing inputs,not independently engineered economic features. This is feature-group validation,not individual marginal contribution or causal importance.')
p('Price hourly return at hour endpoint t aggregates the12five-minute endpoints in[t-1h,t);requires all12finite. The latest completed hourly feature can be carried to a half-hour origin with tolerance59m59s;never backward-filled. Rolling price windows require all hourly returns finite.')
p('Market means/deviations use original masked source snapshots in[t-h,t),>=80%expected half-hour coverage. Deviation=source(t-30min)-historical mean. Exact original11source identifiers are supplied in contracts;the leverage pack uses six directly,flowpressure additionally enters the OI-lowflow interaction.')

page('Formula notation and numerical safeguards')
p('Hourly r is in logbps. Over h completed hours:Q=sum(r^2),N=sum(r),U=sum(max(r,0)^2),RV=sqrt(Q). L is cumulative hourly log-price level;high/low includeh+1levels. width=high-low. Features requireh finite hourly returns.')
table(['Price feature','Exact form'],[['Volatility per sqrt hour300h','sqrt(Q300/300)'],['Normalized momentum300h','N300/max(RV300,1e-8)'],['Range position300h','(L-low300)/max(width300,1e-8)'],['Upside variance share300h','U300/max(Q300,1e-16)'],['Drawdown in volatility units168h','(high168-L)/max(RV168,1e-8)'],['Jump variance share336h','max(r^2 in336h)/max(Q336,1e-16)']],[300,455])
p('Historical funding/basis/OI values are inherited upstream percentile inputs. Their generating percentile/reference normalization is not independently reconstructed. Interaction formulas are hypotheses,not observed causal events. Stored feature dictionary formulas are reproduced on the following pages.')

for start in range(0,len(exact),5):
 page(f'Exact selected feature formulas | {start+1}-{min(start+5,len(exact))}')
 rows=[[start+i+1,r['feature'],r['formula'],r['sources'] or 'Canonical hourly price returns'] for i,r in enumerate(exact[start:start+5])]
 table(['ID','Exact base identifier','Formula','Source'],rows,[30,260,265,200],True)

page('Exact model inputs and sequence membership')
p('April has152stored sequence-value names and July38stored current-value names. Both refer to the same38base feature definitions. For April,each base name c expands toSEQ90__c,SEQ60__c,SEQ30__c,SEQ0__c. Order is snapshot-major and then pack order. No learned model is active inJanuary/October.')
p('Support exact_fitted_inputs.csv enumerates all190active stored value names with issue and policy. model_metadata.json preserves ordered features,preprocessing shapes,saved GRU state shapes,parametercounts,HGBparams,anchors,and training/estimatorrows. selected_policy.json binds the quarterly rules to original saved artifacts.')
table(['Issue','Base definitions','Stored values','Missing flags','Model tensor'],[['Jan',0,0,0,'Constant median'],['Apr',38,152,152,'4snapshots x76inputs'],['Jul',38,38,38,'76inputs'],['Oct',0,0,0,'Constant median']],[90,145,145,145,230])

page('Audit, reproducibility and limits')
p(f'Two completed runs verified;2prediction hashes and{model_count}saved policy-artifact hashes checked,including constant fallbacks.16,873unique timestamps with identical actuals/mean/median aligned. All pooled/period MAE,RMSE andmedianAE independently recomputed from saved CSVs. Previous-selected comparison aligns with original saved predictions.')
p(f'{trial_count}validation ledger rows copied;accepted/candidate minimum validation scores verified. Training cutoff receipts checked;saved model parameters,GRU state shapes and actual fitted feature names inspected. No model fitting or inference performed to create this report. Source COMPLETE replay success is reported,not rerun.')
p('Selection remains a quarterly policy,not a single universalmodel. Selected improves MAE overbothbaselines,not meanRMSE. Diagnosticcandidate is not promoted. The later validation improvement does not certify futureaccuracy,causality or profitability;repeated experiments use already exposed historical2022-2025data.')
for key,path in RUNS.items():p(key+': '+str(path.relative_to(ROOT)),'small')
p('Support includes full original/refined contracts,selections,populationaudit,featuredictionaries,allquarterlyvalidation/quality/boundaryrecords,rankings,periodmetrics,selected38formulas,exact190activeinputnames,modelmetadata,and audit hashes. Raw datasets and saved research outputs remain unchanged.','small')

def footer(canvas,doc):
 canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34);canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY)
 canvas.drawString(40,21,'BTCUSDT | 300h net magnitude | Selected refined GRU/HGB policy | 16Nov2026 | log bps');canvas.drawRightString(802,21,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC300h Net Magnitude - Selected GRU/HGB Policy, Features and Rankings',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));text='\n'.join(v.extract_text() for v in reader.pages)
assert all(v in text for v in ['444.847768','582.187424','GRU16','60% HGB'])
assert all(name in text.replace('\n','') for name in used)
dump(SUP/'audit.json',dict(pdf=str(PDF),pages=len(reader.pages),prediction_runs_verified=2,model_artifact_hashes_verified=model_count,observations=16873,selected_base_features=38,validation_records=trial_count,hashes=hashes,pdf_sha256=sha(PDF),training_performed=False,inference_performed=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP))))

