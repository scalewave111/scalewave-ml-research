"""Audit saved72h net forecasts and build selected MLP policy report; no inference."""
from pathlib import Path
import json,hashlib,shutil,ast
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
from fit_net_magnitude_72h import GRURegressor
from fit_net_magnitude_72h_blended import MAEMLP,FastGRU
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'Code/experiments';OUT=ROOT/'Report'
SUP=OUT/'net_magnitude_72h_report_support_2026-11-16';SUP.mkdir(exist_ok=True)
PDF=OUT/'BTC_72h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
PURE={'hgb':'20261116T100037607594Z','extra':'20261116T100215431081Z','mlp':'20261116T100603349486Z','gru':'20261116T100801089046Z'}
FAST={'hgb':'20261116T103544644698Z','extra':'20261116T103617881321Z','mlp':'20261116T103818828158Z','gru':'20261116T103840269263Z'}
LABEL={'hgb':'HGB','extra':'ExtraTrees','mlp':'MLP','gru':'GRU'}
hashes={};data={};contracts={};selections={};metadata={};scores=[];reference=None;model_count=0;trial_count=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(f,col):
 e=f[col].to_numpy(float)-f.actual.to_numpy(float)
 return dict(rows=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()))
runs=[('pure_'+fam,EXP/f'net_magnitude_72h_{fam}_only_v1'/stamp,fam,False) for fam,stamp in PURE.items()]+[( 'fast_'+fam,EXP/f'net_magnitude_72h_{fam}_blended_fast_v1'/stamp,fam,True) for fam,stamp in FAST.items()]+[('refined_mlp',EXP/'net_magnitude_72h_mlp_refined_v1/20261116T104429218079Z','mlp',True)]
for key,path,family,blended in runs:
 done=js(path/'COMPLETE.json');assert done['status']=='COMPLETE' and sha(track(path/'predictions.csv'))==done['prediction_sha256']
 f=pd.read_csv(path/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True);f=f.sort_values('decision_time').reset_index(drop=True)
 assert len(f)==18241 and f.decision_time.is_unique
 cols=['selected','candidate'] if blended else [family]
 assert np.isfinite(f[['actual','mean','median']+cols]).all().all()
 if reference is None:reference=f[['decision_time','actual','mean','median']]
 else:pd.testing.assert_frame_equal(reference,f[['decision_time','actual','mean','median']])
 data[key]=f;contracts[key]=js(path/'CONTRACT.json');choices=js(path/'selections.json');selections[key]=choices;metadata[key]=[]
 saved=pd.read_csv(track(path/'metrics.csv'))
 for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
  for col in cols+['mean','median']:
   m=metric(g,col);r=saved[(saved.period==period)&(saved.model==col)].iloc[0]
   assert abs(m['mae']-r.mae)<1e-5 and abs(m['rmse']-r.rmse)<1e-5
   scores.append(dict(run=key,forecast=col,period=period,**m))
 for choice in choices:
  for record in choice['models'] if blended else [choice]:
   modelpath=path/record['artifact'];assert sha(track(modelpath))==record['sha256'];model_count+=1
   a=joblib.load(modelpath);assert a['prep']['cols']==record['features'] and a['max_label_end']<pd.Timestamp(choice['issue'])
   model=a['model'];card=dict(issue=choice['issue'],artifact=record['artifact'],features=record['features'],spec=record['spec'],target_center=a['center'],target_scale=a['scale'],training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],inputs=2*len(record['features']))
   if hasattr(model,'weights'):card.update(config=model.config,layer_shapes=[list(w.shape) for w,b in model.weights],parameters=sum(w.size+b.size for w,b in model.weights))
   elif hasattr(model,'state'):card.update(state_shapes={k:list(v.shape) for k,v in model.state.items()},parameters=sum(v.numel() for v in model.state.values()))
   else:card['estimator_parameters']=model.get_params(deep=False)
   metadata[key].append(card)
 trialfile=path/('validation_trials.json' if blended else 'validation_trials.csv');track(trialfile)
 if blended:
  trials=js(trialfile);trial_count+=len(trials)
  for choice in choices:
   t=[v for v in trials if v['issue']==choice['issue']]
   assert np.isclose(choice['candidate']['mae'],min(v['mae'] for v in t))
   eligible=[v for v in t if v['accepted']]
   assert bool(eligible)==(choice['selected'] is not None)
   if eligible:assert np.isclose(choice['selected']['mae'],min(v['mae'] for v in eligible))
 else:trial_count+=len(pd.read_csv(trialfile))
 shutil.copyfile(trialfile,SUP/(key+'_'+trialfile.name));shutil.copyfile(path/'CONTRACT.json',SUP/(key+'_contract.json'));shutil.copyfile(path/'selections.json',SUP/(key+'_selections.json'))
 for name in ['runner_snapshot.py','fit_net_magnitude_72h_blended.py','fit_net_magnitude_72h_mlp_refined.py']:
  if (path/name).exists():track(path/name)
np.testing.assert_array_equal(data['refined_mlp'].selected,data['fast_mlp'].selected)
dump(SUP/'model_metadata.json',metadata);pd.DataFrame(scores).to_csv(SUP/'period_metrics.csv',index=False)
selected_choices=selections['refined_mlp'];selected_cards=[]
for choice in selected_choices:
 if choice['selected'] is not None:
  card=next(v for v in metadata['refined_mlp'] if v['issue']==choice['issue'] and v['spec']==choice['selected']['spec'])
  selected_cards.append(card)
used=sorted(set(c for a in selected_cards for c in a['features']))
member=[]
for name in used:
 member.append(dict(feature=name,issues=','.join(pd.Timestamp(a['issue']).strftime('%b') for a in selected_cards if name in a['features'])))
pd.DataFrame(member).to_csv(SUP/'selected_feature_membership.csv',index=False)
formulas=pd.read_csv(track(runs[-1][1]/'new_feature_formulas.csv'));formulas.to_csv(SUP/'new_feature_formulas.csv',index=False)
dump(SUP/'candidate_feature_packs.json',contracts['refined_mlp']['packs'])
dump(SUP/'selected_mlp_policy.json',dict(selected_run=str(runs[-1][1]),equivalent_fast_run=FAST['mlp'],selection_basis='User selection after exposed historical MAE comparison',quarterly_policy=selected_choices,active_models=selected_cards))
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


page('72-hour net magnitude | selected MLP policy')
p('BTCUSDT - models, features, blends and rankings','title')
p('16 November 2026 | 18,241 matched evaluation origins | absolute endpoint log bps')
p('Target: y(t)=10,000 x abs[ln(P(t+72h)/P(t))]. This is the absolute start-to-end log return, not the UP excursion, DOWN excursion, or UP minus DOWN.')
p('Selected: guarded MLP blended policy. MAE 226.793372 log bps, 0.79% below the training-mean baseline and 2.64% below the training-median baseline. RMSE 312.860278 is worse than the mean baseline 300.507853. The refinement reproduced the earlier fast MLP policy exactly; it did not improve it further.')
rank=[]
for fam in FAST:
 m=metric(data['fast_'+fam],'selected');rank.append([LABEL[fam]+' guarded blend',m['mae'],m['rmse']])
for col in ['mean','median']:
 m=metric(reference,col);rank.append([col.title()+' baseline',m['mae'],m['rmse']])
rank.sort(key=lambda r:r[1]);pd.DataFrame(rank,columns=['policy','mae','rmse']).to_csv(SUP/'policy_rankings.csv',index=False)
table(['MAE rank','Policy / baseline','MAE','RMSE'],[[i+1,r[0],f'{r[1]:.6f}',f'{r[2]:.6f}'] for i,r in enumerate(rank)],[75,320,180,180])
p('Policies may use a baseline fallback. These are not pure neural/tree forecasts. Selection follows exposed historical results, not independent confirmation.','small')

page('Selected MLP | exact quarterly forecast rules')
rows=[]
for choice in selected_choices:
 issue=pd.Timestamp(choice['issue']);g=data['refined_mlp'];g=g[g.issue==issue];m=metric(g,'selected');pick=choice['selected']
 if pick is None:pack='None';count=0;rule='100% mature training mean'
 else:
  card=next(a for a in selected_cards if a['issue']==choice['issue']);pack=pick['spec']['pack'];count=len(card['features']);rule=f"{100*pick['weight']:.0f}% MLP + {100*(1-pick['weight']):.0f}% training {pick['anchor']}"
 rows.append([issue.strftime('%b %Y'),pack,count,rule,f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
table(['Issue','Pack','Base inputs','Forecast rule','MAE','RMSE'],rows,[85,110,80,260,110,110])
p('January has no active learned model in the selected forecast. A diagnostic January candidate was saved, but its features and parameters are excluded from the selected-input inventory. The October model continues through final January 2025 observations.')
p('The active April/July/October models all use the control MLP: 32 hidden ReLU units, 16 epochs, AdamW weight decay 0.1. Exact fitted order, training counts, normalization and layer shapes are preserved in model_metadata.json.')

page('Selected MLP | architecture and training equations')
p('d base inputs -> training-scaled values plus d missing flags -> Dense(32, ReLU) -> Linear(1). Scalar absolute-error regression; no recurrence. Normalized model output g(z)=W2 ReLU(W1 z+b1)+b2.')
p('Training input: keep columns with >=80% finite mature training coverage and >1 distinct value. Impute training median, subtract training median, divide by max(training IQR,1e-6), clip to [-8,8], append missing flags. Preprocessing uses all mature rows, not just estimator sample.')
p('Target scale s=max(median(y_train),1); normalized target=y/s. Train mean absolute error, AdamW lr=0.001, decay=0.1, batch=256, 16 epochs, gradient norm clip=1, seed=48, deterministic shuffled minibatches, CPU float32. Output bias initialized to median(y_train)/s. Estimator sample: at most 2,000 evenly spaced mature rows.')
p('Restore units: p_model=max(0,s*g(z)). Final p=w*p_model+(1-w)*anchor. Anchor is the expanding mature training median or mean selected earlier. January fallback is mean. Mean/median use every mature training label. No probability multiplication or sign conditioning.')
table(['Issue','Input shape','Hidden / output weights','Parameter count','Training rows','Estimator rows'],[[pd.Timestamp(a['issue']).strftime('%b'),str(a['inputs']),str(a['layer_shapes']),a['parameters'],a['training_rows'],a['estimator_rows']] for a in selected_cards],[65,85,230,105,140,130])

page('Why blending was introduced | original pure families')
p('All four original pure-family forecasts failed to beat the mean baseline on pooled MAE and RMSE. The blended experiment changed losses/settings and training budget as well as adding blending, so it does not isolate the causal effect of blending.')
rows=[]
for fam in PURE:
 m=metric(data['pure_'+fam],fam);rows.append([LABEL[fam],f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}'])
for col in ['mean','median']:
 m=metric(reference,col);rows.append([col.title(),f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}'])
table(['Pure model / baseline','MAE','RMSE'],rows,[355,200,200])
p('The mean remains the strongest RMSE comparison after blending. Reported MAE advantage is modest and limited to this exposed historical population.')

page('Fast family search | settings and policy comparisons')
table(['Family','Model settings','Inner fits'],[['HGB','MAE histogram boosting: 40rounds, rate.1, depth3,7leaves,minleaf64,L2=5.',24],['ExtraTrees','24absolute-error trees,depth4,minleaf32,maxfeatures.75,bootstrap cap1024/tree.',24],['MLP','32ReLU hidden,16epochs,L1,AdamWlr.001,decay.1.',24],['GRU','One GRU16 layer + Linear16to1;12epochs,L1,AdamWlr.001,decay.03; snapshots12/8/4/0h.',24]],[100,555,100])
p('All families: same 2 packs (market_new and all), same 2,000-row estimator cap and quarterly origins, seed48. HGB/GRU target center=training median and scale=max(sample std,1). ExtraTrees/MLP use median target scaling without centering. Same training-only IQR preprocessing.')
rows=[]
for fam in FAST:
 f=data['fast_'+fam]
 for col in ['selected','candidate']:
  m=metric(f,col);rows.append([LABEL[fam],col,f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}'])
table(['Family','Forecast','MAE','RMSE'],rows,[150,205,200,200])

page('Refinement | what changed and what did not')
p('The focused MLP search retained the fast control32hidden/16epochs/decay.1, added 32epochs at decay.1 and 16epochs at decay.03. Same two packs, estimator cap2,000 and seed48. Seventy-two inner fits; 8weights x3anchors per fitted configuration reuse predictions.')
p('Refined weights: .1/.25/.4/.5/.6/.75/.9/1. Anchors: training median, training mean, or midpoint=(median+mean)/2. Fast weights were .25/.5/.75/1 with median/mean anchors. The active selected forecast retained the original control and existing weights, so its CSV predictions are exactly equal to the fast selected MLP policy.')
rows=[]
for key,col,label in [('fast_mlp','selected','Fast MLP guarded policy'),('refined_mlp','selected','Refined MLP guarded policy'),('refined_mlp','candidate','Refined diagnostic candidate')]:
 m=metric(data[key],col);rows.append([label,f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}'])
table(['Forecast','MAE','RMSE'],rows,[355,200,200])
p('The diagnostic candidate is the lowest earlier-validation MAE blend without acceptance restrictions. The selected policy is the lowest earlier-validation MAE eligible blend, or mean fallback. These policies can differ; later errors never choose which candidate to promote.')

page('Time boundaries, population and acceptance rules')
audit=contracts['refined_mlp']['input_audit']
p('Canonical BTCUSDT horizon_minutes=4320 labels are checksum verified by the original runners. Their saved audit reports '+str(audit['original_rows'])+' original rows, '+str(audit['valid_rows'])+' eligible rows and 18,241 evaluation origins. Exclusions: '+str(audit['exclusions'])+'. Labels must be VALID, finite and nonnegative; source-binding transition-crossing labels are excluded.')
p('At Jan/Apr/Jul/Oct2024 issues, use three earlier two-month validation windows starting9/6/3months before the issue. Every training label_end must be strictly before its fitting cutoff; validation label_end must precede outer issue. No later outcome informs that issue selection.')
p('Rank row-count-weighted pooled validation MAE. Accept only >=0.1% pooled MAE gain against BOTH training-mean and training-median baselines, >=2/3 window wins/ties against EACH, and no >2% loss in any window against EITHER. Otherwise selected forecast is mature training mean. The gate does not guarantee later-period improvement.')
p('Feature choice is within a family. Five packs in pure runs; two in fast/refined runs. Full population preprocessing and anchors use all mature rows; only estimator fitting is capped. The original half-hour playground X grid is inherited, not the full canonical minute population.')

for start in [0,2]:
 page('Quarterly scores | '+('January and April' if start==0 else 'July and October'))
 rows=[]
 issues=sorted(data['refined_mlp'].issue.unique())
 for issue in issues[start:start+2]:
  for key,col,label in [('refined_mlp','selected','Selected MLP')]+[('fast_'+fam,'selected',LABEL[fam]+' fast policy') for fam in ['hgb','extra','gru']]+[('refined_mlp','mean','Mean'),('refined_mlp','median','Median')]:
   f=data[key];g=f[f.issue==issue];m=metric(g,col);rows.append([pd.Timestamp(issue).strftime('%b %Y'),label,len(g),f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
 table(['Issue','Policy','Rows','MAE','RMSE'],rows,[100,245,90,160,160])

page('Candidate feature packs and selected input counts')
for name,cols in contracts['pure_mlp']['packs'].items():
 p(name+': '+str(len(cols))+' candidate columns.','sub')
p('Fast/refined search compares market_new61 and all296. The training quality filter changes actual retained input count. Exact selected quarterly identifiers and order are preserved in support. January fallback consumes no model features.')
p('The selected policy uses '+str(len(used))+' distinct base feature names across its active April/July/October models. Each base feature contributes its scaled value and missing flag. This is feature-group selection, not measured individual-feature importance or exhaustive subset optimization.')

page('Feature formulas | notation and inherited transformations')
p('Canonical5minute log-bps return r_i=10,000 ln(P_i/P_(i-5min)), timestamped by endpoint. Prior window includes t-h<=i<t, requiring12h finite returns. N=sum r, A=sum abs(r), Q=sum r^2, U=sum max(r,0)^2, J=max abs(r).')
table(['New-feature family','Formula'],[['rv / net / abs_net','sqrt(Q) / N / abs(N)'],['abs_sum / efficiency','A / abs(N)/max(A,1e-12)'],['up_variance_share / jump_share','U/max(Q,1e-12) / J^2/max(Q,1e-12)'],['rms / positive_share','sqrt(Q)/sqrt(12h) / mean(r>0)'],['rv_ratio_h_168','(RV_h/sqrt(h))/max(RV_168/sqrt(168),1e-9)']],[235,520])
p('Five lookbacks4/24/48/168/336h x9families plus ratios4/24/48 versus168h =48new features. Historical lookbacks are distinct from the72h prediction horizon.')
p('Inherited transforms: reindex original inputs on30minute grid, shift one row (z=c(t-30min)). Over w=7/14/30 shifted rows require w finite values: std(ddof0), max-min range, P75-P25 IQR, (z-mean)/std (missing if std<=1e-12); ma_ratio7_30 and14_30 divide window means (missing if abs30mean<=1e-8). lag1=z. Calendar one-hots:weekday0..6, weekofmonth floor((day-1)/7)+1,quarter1..4.','small')
p('LONG24/48/168/720h features: total/up/downRV,abs(sum r),sum abs(r),range of cumulative return level; strictly prior endpoints with full finite coverage. Upstream F4Z reference distributions and normalization are inherited, not independently reconstructed. F4Z activity percentiles are not signed flow; historical activity rates are not current liquidation size.','small')
for start in range(0,len(formulas),12):
 page(f'Exact new feature formulas | {start+1}-{min(start+12,len(formulas))}')
 table(['Exact identifier','Implemented formula'],[[r.feature,r.formula] for r in formulas.iloc[start:start+12].itertuples()],[290,465],True)
for start in range(0,len(member),10):
 page(f'Selected MLP | exact feature membership {start+1}-{min(start+10,len(member))}')
 table(['ID','Exact base feature identifier','Active issues'],[[start+i+1,r['feature'],r['issues']] for i,r in enumerate(member[start:start+10])],[40,600,115],True)

page('Evidence, reproducibility and limits')
p(f'Nine saved runs audited:4pure families,4fast blended families,1MLP refinement. All9prediction hashes and{model_count}saved model-artifact hashes verified;18,241unique timestamps and identical actuals/baselines aligned. Pooled and period metrics recomputed; saved layer weights/estimator parameters inspected. {trial_count}validation ledger records copied; blended candidate/accepted minimum selection checked against ledgers.')
p('No training or new model inference performed for this report. COMPLETE receipts report serialization replay passed; replay was not rerun. Raw canonical source checks were performed by original runners, not reconstructed for this report. Run snapshots and contracts preserve provenance.')
p('Selected MLP policy includes a baseline fallback and baseline blends. It improves MAE, not mean-baseline RMSE. Refinement does not improve the fast policy. Repeated experimentation uses already exposed historical2022-2025data; no untouched-test, significance, future profitability or guaranteed-performance claim.')
p('All active selected features are listed; diagnostic-only features are excluded from that inventory. Support includes full run metadata, all selection records and ledgers, exact pack inventory, rankings, period errors, formulas, audit hashes and selected policy definition.')
for key,path,fam,blendflag in runs:p(key+': '+str(path.relative_to(ROOT)),'small')

def footer(canvas,doc):
 canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34);canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY)
 canvas.drawString(40,21,'BTCUSDT | 72h net magnitude | Selected MLP policy | 16 Nov2026 | log bps');canvas.drawRightString(802,21,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC72h Net Magnitude - Selected MLP, Features and Rankings',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));text='\n'.join(v.extract_text() for v in reader.pages)
assert all(v in text for v in ['226.793372','300.507853','January','NET72__rv_ratio_48_168'])
assert all(name in text.replace('\n','') for name in used)
dump(SUP/'audit.json',dict(pdf=str(PDF),pages=len(reader.pages),prediction_runs_verified=9,model_artifact_hashes_verified=model_count,observations=18241,selected_base_features=len(used),validation_records=trial_count,hashes=hashes,pdf_sha256=sha(PDF),training_performed=False,inference_performed=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP),selected_features=len(used))))
