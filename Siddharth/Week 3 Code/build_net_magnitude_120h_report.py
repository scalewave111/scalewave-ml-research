"""Saved120h report: selected MLP wide16; no training/inference."""
from pathlib import Path
import json,hashlib,shutil
from xml.sax.saxutils import escape
import joblib,numpy as np,pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
from fit_net_magnitude_120h_bounded_refined import SmallCorrection
from fit_net_magnitude_120h import GRURegressor
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'Code/experiments';OUT=ROOT/'Report'
SUP=OUT/'net_magnitude_120h_report_support_2026-11-16';SUP.mkdir(exist_ok=True)
PDF=OUT/'BTC_120h_Net_Magnitude_Models_Features_Rankings_2026-11-16.pdf'
PURE={'hgb':'20261116T101322630960Z','extra':'20261116T101506990782Z','mlp':'20261116T101737342721Z','gru':'20261116T101825668821Z'}
LABEL={'hgb':'HGB','extra':'ExtraTrees','mlp':'MLP','gru':'GRU'}
STAMP='20261116T122202106711Z';hashes={};data={};contracts={};plans={};cards={};reference=None;model_count=0;allmetrics=[];trialcount=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(f,col):
 e=f[col].to_numpy(float)-f.actual.to_numpy(float)
 return dict(rows=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean()))

RUNS={k:(EXP/f'net_magnitude_120h_{k}_only_v1'/v) for k,v in PURE.items()}
RUNS.update({k+'_refined':EXP/f'net_magnitude_120h_{k}_bounded_refined_v1'/STAMP for k in ['mlp','gru']})
for key,path in RUNS.items():
 fam=key.split('_')[0];done=js(path/'COMPLETE.json');assert done['status']=='COMPLETE'
 assert sha(track(path/'predictions.csv'))==done['prediction_sha256']
 f=pd.read_csv(path/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True)
 assert len(f)==17953 and f.decision_time.is_unique
 if reference is None:reference=f[['decision_time','actual','mean','median']]
 else:pd.testing.assert_frame_equal(reference,f[['decision_time','actual','mean','median']])
 data[key]=f;contracts[key]=js(path/'CONTRACT.json');plans[key]=js(path/'selections.json');cards[key]=[]
 saved=pd.read_csv(track(path/'metrics.csv'))
 for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
  for col in saved[saved.period==period].model:
   m=metric(g,col);r=saved[(saved.period==period)&(saved.model==col)].iloc[0]
   np.testing.assert_allclose([m['mae'],m['rmse'],m['bias']],[r.mae,r.rmse,r.bias],rtol=0,atol=1e-5)
   allmetrics.append(dict(run=key,period=period,forecast=col,**m))
 for plan in plans[key]:
  records=plan['models'] if key.endswith('refined') else [plan]
  for record in records:
   mp=track(path/record['artifact']);assert sha(mp)==record['sha256'];model_count+=1
   a=joblib.load(mp);assert a['prep']['cols']==record['features'];assert pd.Timestamp(a['max_label_end'])<pd.Timestamp(plan['issue'])
   model=a['model'];card=dict(issue=plan['issue'],artifact=record['artifact'],sha256=record['sha256'],features=record['features'],spec=a['spec'],training_rows=a['training_rows'],estimator_rows=a['estimator_rows'],max_label_end=a['max_label_end'],center=a['center'],scale=a['scale'],prep_median=a['prep']['median'].to_dict(),prep_iqr=a['prep']['scale'].to_dict())
   if hasattr(model,'state'):
    card.update(state_shapes={k:list(v.shape) for k,v in model.state.items()},parameter_count=sum(v.numel() for v in model.state.values()),inputs=model.inputs)
    if hasattr(model,'config'):card['config']=model.config
   else:card['parameters']=model.get_params(deep=False)
   cards[key].append(card)
 if key.endswith('refined'):
  trials=js(path/'validation_trials.json');trialcount+=len(trials)
  for plan in plans[key]:
   g=[t for t in trials if t['issue']==plan['issue']];order=lambda t:(t['stability_score'],t['mae'],t['cell'],t['weight'])
   assert min(g,key=order)==plan['candidate'];ok=[t for t in g if t['accepted']]
   assert (min(ok,key=order) if ok else None)==plan['selected']
  for filename in ['validation_trials.json','training_validation_diagnostics.csv','outer_cell_diagnostics.csv']:
   shutil.copyfile(track(path/filename),SUP/(key+'_'+filename))
 else:
  trials=pd.read_csv(track(path/'validation_trials.csv'));trialcount+=len(trials);trials.to_csv(SUP/(key+'_validation_trials.csv'),index=False)
 shutil.copyfile(track(path/'CONTRACT.json'),SUP/(key+'_contract.json'))
 track(path/'model_'+('notused')) if False else None
 if (path/'runner_snapshot.py').exists():track(path/'runner_snapshot.py')
 elif (path/'fit_net_magnitude_120h_bounded_refined.py').exists():track(path/'fit_net_magnitude_120h_bounded_refined.py')

pd.DataFrame(allmetrics).to_csv(SUP/'all_run_period_metrics.csv',index=False)
dump(SUP/'model_metadata.json',cards)
selected=[c for c in cards['mlp_refined'] if c['spec'].get('config')=='wide16'];assert len(selected)==4
assert all(c['config']['bound']==.4 and c['config']['hidden']==16 and c['config']['epochs']==12 for c in selected)
used=sorted(set(n for c in selected for n in c['features']))
formulas=pd.read_csv(track(RUNS['mlp_refined']/'feature_formulas.csv')).fillna('');formulas=formulas[formulas.feature.isin(used)].drop_duplicates('feature',keep='last');assert set(used)==set(formulas.feature)
formulas.to_csv(SUP/'selected_feature_formulas.csv',index=False)
pd.DataFrame([dict(feature=n,**{pd.Timestamp(c['issue']).strftime('%b'):n in c['features'] for c in selected}) for n in used]).to_csv(SUP/'selected_feature_membership.csv',index=False)
chosen=data['mlp_refined'];selected_scores=metric(chosen,'wide16')
dump(SUP/'selected_mlp.json',dict(selected_by='User instruction 2026-11-16 after historical fixed-configuration comparison',family='mlp',configuration='wide16',forecast_column='wide16',run=STAMP,automated_selected=False,architecture='inputs2d -> Linear16 -> ReLU -> Linear1 -> tanh',prediction='training_mean + .4*training_mean*tanh(net);nonnegative clamp',outer_shrinkage_weight=1,metrics=selected_scores,quarters=selected,refit=False,deployed=False,limitations='Posthoc fixedconfiguration choice,exposed history,no independent confirmation'))
ranking=[]
for fam in PURE:ranking.append(dict(model=LABEL[fam]+' original pure',**metric(data[fam],fam)))
ranking += [dict(model='Selected MLP wide16',**selected_scores)]+[dict(model=b.title()+' baseline',**metric(chosen,b)) for b in ['mean','median']]
ranking=pd.DataFrame(ranking).sort_values('mae');ranking['mae_rank']=ranking.mae.rank(method='min').astype(int);ranking['rmse_rank']=ranking.rmse.rank(method='min').astype(int);ranking.to_csv(SUP/'model_rankings.csv',index=False)
layout=(ROOT/'Code/build_net_magnitude_48h_report.py').read_text(encoding='utf-8');a=layout.index('NAVY=');b=layout.index("page('48-hour",a);exec(layout[a:b],globals())
page('120-hour net magnitude | selected MLP report')
p('BTCUSDT - models, features, formulas and rankings','title')
p('16 November 2026 | 17,953 aligned observations | Selected fixed configuration: MLP wide16')
p('Target = 10,000 x abs[ln(P(t+120h)/P(t))], in log bps. User selection records the four saved quarterly wide16 estimators; no retraining or deployment.')
p('Selected MLP MAE 291.685813: 0.31% below the mean baseline and 5.48% below the median. RMSE 381.681299 is 1.12% above the mean baseline. The MAE gain is small and follows repeatedly inspected historical evaluation.')
table(['MAE rank','Forecast','MAE','RMSE','RMSE rank'],[[r.mae_rank,r.model,f'{r.mae:.6f}',f'{r.rmse:.6f}',r.rmse_rank] for r in ranking.itertuples()],[65,260,145,145,140])
p('wide16 is the user-chosen fixed configuration, not the automated selected column. That validation-selected MLP policy uses long16 with quarterly shrinkage. Family and configuration choice now follow exposed outer scores.','small')
page('Baseline comparison and forecast identity')
table(['Forecast','MAE','RMSE','Bias'],[[label,f'{metric(chosen,col)["mae"]:.6f}',f'{metric(chosen,col)["rmse"]:.6f}',f'{metric(chosen,col)["bias"]:.6f}'] for label,col in [('User-chosen MLP wide16','wide16'),('Automated MLP selected','selected'),('MLP diagnostic candidate','candidate'),('Mean baseline','mean'),('Median baseline','median')]],[300,150,150,155])
p('Mean MAE reduction: 0.905445 log bps (0.3095%). Median MAE reduction: 16.918673 log bps (5.4823%). These are relative reductions in aggregate MAE, not percentages of price or trading returns.')
p('The chosen forecast is the raw bounded wide16 output in every issue, with weight1. It does not apply the selector\'s long16 blend or mean fallback. The prior tiny8 and strong16 controls reproduce their saved earlier CSV metrics.')
page('Selected MLP | architecture and optimization')
p('For d retained base features: 2d transformed inputs -> Dense16 -> ReLU -> Dense1 -> tanh. One hidden layer; CPU float32; seed48; deterministic torch algorithms. Each raw feature contributes a transformed value and a missing-value flag.')
p('AdamW learning_rate=0.0005; weight_decay=1.0; 12 epochs; batch256; gradient-norm clip0.5. Each epoch uses a deterministic random permutation. No early stopping or dropout. Final layer weights and bias start at zero, so the initial forecast equals the training mean.')
p('At each fit, mu=mean(Y_mature training), s=max(0.4*mu,1e-6), target=(Y-mu)/s. Optimize mean absolute error between tanh(net(inputs)) and normalized target. With fixed positive s this is original-unit MAE up to a constant; target values are not clipped.')
p('Prediction=max(0,mu+s*tanh(net(inputs))). Because mu is positive and s=0.4mu here, forecasts lie between 0.6mu and 1.4mu. This output bound trades flexibility for smaller errors from unstable adjustments.')
table(['Issue','Base d','Inputs','Parameters','Mature rows','Estimator rows'],[[pd.Timestamp(c['issue']).strftime('%b %Y'),len(c['features']),c['inputs'],c['parameter_count'],c['training_rows'],c['estimator_rows']] for c in selected],[110,90,100,130,160,165])
page('Selected MLP | quarterly saved estimators')
table(['Issue','Training mean mu','Correction scale .4mu','Latest training label end'],[[c['issue'][:10],f'{c["center"]:.6f}',f'{c["scale"]:.6f}',str(c['max_label_end'])] for c in selected],[100,180,180,295])
for c in selected:p(c['issue'][:10]+' artifact: '+c['artifact']+' | SHA256 '+c['sha256'],'small')
p('Exact feature order, tensor shapes, all preprocessing medians/IQRs, training counts, centers/scales and artifact hashes are included in model_metadata.json and selected_mlp.json. Saved replay was reported by the source run; no inference replay is performed for this report.')
for start in range(4):
 page('Scores by period | '+pd.Timestamp(list(chosen.issue.unique())[start]).strftime('%B %Y'))
 rows=[]
 for issue in list(chosen.issue.unique())[start:start+1]:
  for label,key,col in [('Chosen MLP wide16','mlp_refined','wide16'),('Automated MLP','mlp_refined','selected'),('Automated GRU','gru_refined','selected')]+[(LABEL[f]+' original',f,f) for f in PURE]+[('Mean','mlp_refined','mean'),('Median','mlp_refined','median')]:
   g=data[key][data[key].issue==issue];m=metric(g,col);rows.append([pd.Timestamp(issue).strftime('%b %Y'),label,m['rows'],f'{m["mae"]:.4f}',f'{m["rmse"]:.4f}'])
 table(['Issue','Forecast','Rows','MAE','RMSE'],rows,[100,255,90,155,155],True)
page('Bounded MLP and GRU configuration comparison')
rows=[]
for fam in ['mlp','gru']:
 for config in ['tiny8','strong16','tight16','wide16','long16','selected']:
  m=metric(data[fam+'_refined'],config);rows.append([LABEL[fam],config,f'{m["mae"]:.6f}',f'{m["rmse"]:.6f}'])
table(['Family','Configuration / policy','MAE','RMSE'],rows,[100,285,185,185])
p('tiny8:8hidden,decay.5,bound25%; strong16:16hidden,decay1,bound25%; tight16:bound10%; wide16:bound40%; long16:bound25%,24epochs. Other bounded settings held fixed. Automated policy blends configurations with mean using earlier validation. RMSE remains best for the mean baseline.','small')
page('Automated validation policy | distinct from user choice')
p('Four earlier three-month windows start15/12/9/6months before each issue. Training label_end is strictly before its cutoff. Validation label_end is strictly before its window end and outer issue. The feature timeline is constructed before outcome eligibility filtering.')
p('Five configs x 9 mean-shrinkage weights x 4 issues = 180 validation records per family. Weights .025/.05/.1/.2/.25/.35/.5/.75/1. Rank mean(windowMAE/meanbaselineMAE)+0.25*std(window ratios), then pooledMAE and deterministic identifiers.')
p('Acceptance: at least0.1%pooled MAEgain vsmean,at least3/4windowwins/ties,no>1%windowMAEloss. Otherwise meanfallback. RMSE diagnostic only. Earlier controlled search used pooledMAErank,2/4wins,no>2%loss; this selector change is explicit.')
table(['Family','Issue','Automated config','Model weight','User-chosen forecast'],[[LABEL[f],v['issue'][:10],v['selected']['cell'] if v['selected'] else 'mean fallback',v['selected']['weight'] if v['selected'] else 0,'wide16 weight1' if f=='mlp' else 'comparison only'] for f in ['mlp','gru'] for v in plans[f+'_refined']],[80,100,195,130,250])
page('Target, population and preprocessing')
p('Canonical BTCUSDT horizon_minutes=7200 absolute endpoint label,checksum verified by source runner. Original 48,096 origins;47,568 eligible training/evaluation timeline rows;17,953 evaluation origins. Exclusions 288 missing/invalidlabels and 240 source-binding crossings. Original half-hour origin grid; not full canonical-minute population.')
p(f'Evaluation coverage: {reference.decision_time.min()} to {reference.decision_time.max()}. October model continues into January2025. Labels overlap heavily at a120h horizon; rows are not independent trials.')
p('At each fit retain features with>=80%finite mature training coverage and variation. Fit medians/IQR on all mature rows: z=clip((x-imputed training median)/max(IQR,1e-6),-8,8); signedlog=sign(z)*ln(1+abs(z)); append missing flags. No target or evaluation outcomes enter features or preprocessing.')
p('Estimator training uses up to2,000evenly spaced mature rows; preprocessing,mean/medianbaselines and normalization use all mature rows. GRU comparison uses exact36/24/12/0h snapshots from fulltimeline. Selected MLP uses only current feature row.')
page('Exact selected inputs | formula notation')
p('Canonical 5min log-bps return r_i is indexed by its ending time. A completed hour H_t=sum r_i for t-1h<=i<t,requiring12finite returns. Historical h-hour features use h completed hourly returns ending no later than the decision time; the current/future5min endpoint is excluded.')
p('N=sum H; A=sum abs(H); Q=sum H^2; RV=sqrt(Q); U=sum max(H,0)^2. Local log-price levels are cumulative completed hourly returns. High/low over h+1levels define range. Denominator floors are1e-8 for RV/range/A and1e-16 for squaredenergy. All h hourly values required.')
p('normalized_momentum=N/RV; trend_efficiency=N/A; up_variance_share=U/Q; range_position=(currentlevel-local low)/(local high-local low). Each occurs at60/120/240h. Drawdown168h=(local high-currentlevel)/RV168; jump336h=max(H^2)/sum(H^2).')
p('Backward120h absolute endpoint magnitude M_t=abs(sum last120completed hourly returns). Over720hourly observations: mean720=mean(M),CV720=population_std(M)/max(mean(M),1e-8). These overlapping past magnitudes are historical predictor values,not future labels.')
p('Latest completed hourly values are forward-aligned to half-hour origins with at most59min59sec age; no backfill. Prefix/future checks were reported by input audit and source runners; not rerun for report creation.')
for start in range(0,len(formulas),8):
 page(f'Exact selected feature formulas | {start+1}-{min(start+8,len(formulas))}')
 table(['Feature identifier','Lookback','Implemented formula'],[[r.feature,str(getattr(r,'lookback','')),r.formula] for r in formulas.iloc[start:start+8].itertuples()],[325,80,350],True)
for start in [0,8]:
 page('Exact MLP feature order | '+str(start+1)+'-'+str(start+8))
 p('All 16 fixed inputs are used in every quarter,each with its missing flag. Listed order is the actual saved estimator order. Membership does not establish individual causal importance. Selected MLP has no direct F4Z input.','small')
 names=selected[0]['features']
 table(['Order','Base feature','Jan','Apr','Jul','Oct'],[[i+1,names[i]]+['yes' if names[i] in c['features'] else '-' for c in selected] for i in range(start,min(start+8,len(names)))],[45,450,65,65,65,65],True)
page('Original four models | architecture and comparison settings')
table(['Family','Original pure-run settings'],[['HGB','L1 histogram trees;60rounds,7leaves,depth3,minleaf64,L2=5,no earlystop.'],['ExtraTrees','48squared-error trees,depth10,minleaf32,maxfeatures.8,n_jobs1,seed48.'],['MLP','Sklearn16ReLUhidden,squared-error,alpha1,batch256,lr.001,maxiter40,no shuffle/earlystop.'],['GRU','GRU16hidden+linear,L1,24epochs,AdamWlr.001decay.03,clip1;12/8/4/0h snapshots.']],[130,625])
p('Original pure-run feature packs selected within each family using earlier three 2-month windows9/6/3months. Primary forecasts have no baseline fallback. Later bounded MLP uses a fixed16feature set and different training/validation/regularization. Improvements across these experiments cannot be assigned solely to model architecture.')
page('Research progression and interpretation')
p('Original four pure models all lost meanMAE. Baseline blending,neuralL1/L2,endpoint-specific features,tree-neuralhybrids andLSTM/RandomForest alternatives failed to establish a baseline-winning selected policy. Larger controlled budgets generally reduced apparent training error but worsened laterMAE.')
p('Small bounded networks produced fixedconfigMAEgains:strong16MLP291.806,tiny8MLP292.047;the earlier selector still chose the old control. Refined bounded grid yielded selectedMLP292.052 and user-chosen fixedwide16MLP291.686. Controls reproduce earlier scores.')
p('This report does not claim significance,profitability or generalization. Many searches used this same history. The fixedwide16 choice follows outer evaluation and requires genuinely unseen chronological confirmation. MeanRMSE377.446 remains better than381.681 for the chosenMLP.')
page('Audit, reproducibility and retained evidence')
p(f'Six prediction runs hash-verified;{model_count}saved model-artifact hashes verified. Identical17,953 timestamps/actuals/baselines aligned. All saved pooled/period metrics recomputed directly from CSVs. Fitted feature lists,counts,normalization,model state shapes/parameters inspected;{trialcount}validation records retained. Refined selector candidate/accepted minima checked.')
p('No training or new inference performed to build this report. Serialization replay reported by original COMPLETE files,not rerun. Source-run checksums establish consistency with receipts,not independent verification of upstream data-generation formulas.')
for key,path in RUNS.items():p(key+': '+str(path.relative_to(ROOT)),'small')
p('Support includes selected_mlp.json,modelmetadata,rankings,allperiodmetrics,exact16featureformulas/membership,contracts,validationledgers,diagnostics,source snapshots and audit hashes. Saved models remain in original run folders; support ZIP does not duplicate model binaries.','small')
def footer(canvas,doc):
 canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34);canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY);canvas.drawString(40,21,'BTCUSDT | 120h net magnitude | selected MLP wide16 | 16 Nov 2026');canvas.drawRightString(802,21,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC120h net magnitude - selected MLP,features and rankings',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));text='\n'.join(p.extract_text() for p in reader.pages)
assert all(v in text for v in ['291.685813','wide16','377.446','ExtraTrees'])
assert all(n in text.replace('\n','') for n in used)
dump(SUP/'audit.json',dict(pdf=str(PDF),pdf_sha256=sha(PDF),pages=len(reader.pages),predictions_verified=6,model_hashes_verified=model_count,validation_records=trialcount,observations=17953,selected_features=len(used),hashes=hashes,training=False,inference=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),support=str(SUP),models=model_count)))
