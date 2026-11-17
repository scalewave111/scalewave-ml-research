"""Build audited direction48h report from saved forecasts. No training/inference."""
from pathlib import Path
import json,hashlib,shutil
from xml.sax.saxutils import escape
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
from fit_direction_48h_classifiers import NeuralClassifier
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'Code/experiments';OUT=ROOT/'Report'
SUP=OUT/'direction_48h_report_support_2026-11-16';SUP.mkdir(exist_ok=True)
PDF=OUT/'BTC_48h_Direction_Models_Features_Rankings_2026-11-16.pdf'
RUNS={'original':EXP/'direction_48h_classifiers_v1/20261116T130206171832Z','calibrated':EXP/'direction_48h_calibrated_refined_v1/20261116T131140263083Z'}
hashes={};data={};contracts={};plans={};cards={};rows=[];reference=None;model_count=0;trial_count=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(f,col):
 y=f.y_up.to_numpy();p=np.clip(f[col].to_numpy(float),1e-7,1-1e-7)
 return dict(rows=len(y),log_loss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),brier=float(((p-y)**2).mean()),accuracy=float(((p>=.5)==y).mean()),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,mean_probability=float(p.mean()),up_rate=float(y.mean()))
for key,path in RUNS.items():
 done=js(path/'COMPLETE.json');assert done['status']=='COMPLETE' and sha(track(path/'predictions.csv'))==done['prediction_sha256']
 f=pd.read_csv(path/'predictions.csv');f.decision_time=pd.to_datetime(f.decision_time,utc=True);f.issue=pd.to_datetime(f.issue,utc=True)
 assert len(f)==18385 and f.decision_time.is_unique
 for col in ['p_up','p_down','candidate','prevalence','fair']:assert f[col].between(0,1).all()
 np.testing.assert_allclose(f.p_up+f.p_down,1,atol=1e-12)
 assert np.array_equal(f.y_up,f.signed_return_log_bps.gt(0).astype(int))
 if reference is None:reference=f[['decision_time','y_up','signed_return_log_bps','prevalence']]
 else:pd.testing.assert_frame_equal(reference,f[['decision_time','y_up','signed_return_log_bps','prevalence']])
 data[key]=f;contracts[key]=js(path/'CONTRACT.json');plans[key]=js(path/'selections.json');cards[key]=[]
 saved=pd.read_csv(track(path/'metrics.csv'));trials=js(path/'validation_trials.json');trial_count+=len(trials)
 for period,g in [('ALL',f)]+[(str(k),v) for k,v in f.groupby('issue')]:
  for col in saved[saved.period==period].model:
   m=metric(g,col);r=saved[(saved.period==period)&(saved.model==col)].iloc[0]
   for field in ['log_loss','brier','accuracy','auc']:
    if m[field] is not None:assert abs(m[field]-r[field])<1e-9
   rows.append(dict(run=key,period=period,forecast=col,**m))
 for plan in plans[key]:
  trial=[t for t in trials if t['issue']==plan['issue']]
  order=(lambda t:(t['score'],t['id'],t['weight'])) if key=='original' else (lambda t:(t['score'],t['id'],t['mode'],t['weight']))
  assert min(trial,key=order)==plan['candidate'];ok=[t for t in trial if t['accepted']];assert (min(ok,key=order) if ok else None)==plan['selected']
  for record in plan['models']:
   mp=track(path/record['artifact']);assert sha(mp)==record['sha256'];model_count+=1
   a=joblib.load(mp);assert a['prep']['cols']==record['features'];assert pd.Timestamp(a['max_label_end'])<pd.Timestamp(plan['issue'])
   m=a['model'];card=dict(issue=plan['issue'],artifact=record['artifact'],sha256=record['sha256'],spec=a['spec'],features=a['prep']['cols'],training_rows=a.get('training_rows'),estimator_rows=a.get('estimator_rows'),max_label_end=a['max_label_end'],median=a['prep']['median'].to_dict(),iqr=a['prep']['iqr'].to_dict())
   if hasattr(m,'state'):card.update(inputs=m.inputs,state_shapes={k:list(v.shape) for k,v in m.state.items()},parameters=sum(v.numel() for v in m.state.values()))
   else:card['parameters']=m.get_params(deep=False)
   cards[key].append(card)
 for fn in ['CONTRACT.json','selections.json','validation_trials.json','metrics.csv']:
  shutil.copyfile(track(path/fn),SUP/(key+'_'+fn))
 for fn in ['feature_pack_comparison.csv','calibration.csv','calibration_audit.json']:
  if (path/fn).exists():shutil.copyfile(track(path/fn),SUP/(key+'_'+fn))
 if (path/'fit_direction_48h_classifiers.py').exists():track(path/'fit_direction_48h_classifiers.py')
f=data['original'];np.testing.assert_allclose(data['calibrated'].previous_selected,f.p_up,atol=1e-12);np.testing.assert_allclose(data['calibrated'].previous_candidate,f.candidate,atol=1e-12)
pd.DataFrame(rows).to_csv(SUP/'all_period_metrics.csv',index=False);dump(SUP/'model_metadata.json',cards)
active=[];membership=[]
for plan in plans['original']:
 t=plan['selected']
 if t:
  c=next(c for c in cards['original'] if c['issue']==plan['issue'] and c['spec']==t['spec']);active.append(c)
used=sorted(set(n for c in active for n in c['features']))
for n in used:membership.append(dict(feature=n,**{pd.Timestamp(p['issue']).strftime('%b'):('yes' if p['selected'] and any(c['issue']==p['issue'] and n in c['features'] for c in active) else '-') for p in plans['original']}))
pd.DataFrame(membership).to_csv(SUP/'active_feature_membership.csv',index=False)
new=pd.read_csv(track(RUNS['original']/'new_direction_feature_formulas.csv'));assert len(new)==94;new.to_csv(SUP/'new_direction_feature_formulas.csv',index=False)
dump(SUP/'selected_direction_policy.json',dict(selected_by='User requested report of best guarded policy on2026-11-16',run='20261116T130206171832Z',forecast_column='p_up',p_down='1-p_up; non-UP includes zero',metrics=metric(f,'p_up'),quarterly=plans['original'],active_feature_count=len(used),refit=False,deployed=False,limitations='Earlier validation guard approval,not independent confirmation. Diagnosticcandidate not selected. Historical384feature inventory and evaluation already exposed.'))
f[['decision_time','y_up','actual_direction','predicted_direction','p_up','p_down','signed_return_log_bps','prevalence']].to_csv(SUP/'selected_direction_predictions.csv',index=False)
layout=(ROOT/'Code/build_net_magnitude_48h_report.py').read_text();a=layout.index('NAVY=');b=layout.index("page('48-hour",a);exec(layout[a:b],globals())
labels={'prevalence':'Training-prevalence baseline','fair':'50/50 baseline','p_up':'Selected quarterly policy','candidate':'Diagnostic candidate','logistic':'Logistic family policy','hgb':'HGB family policy','random_forest':'RandomForest family policy','extra_trees':'ExtraTrees family policy','mlp_logloss':'MLP log-loss policy','mlp_brier':'MLP Brier policy','gru_logloss':'GRU log-loss policy','gru_brier':'GRU Brier policy'}
page('48-hour direction | selected classification policy')
p('BTCUSDT - model architectures, features, formulas and probability scores','title')
p('16 November 2026 | 18,385 matched observations | Log loss and Brier score')
p('Chosen forecast: guarded p_up from run 20261116T130206171832Z. January/October prevalence fallback; April 50% Brier-trained MLP +50% prevalence; July ExtraTrees. This is a quarterly policy, not one classifier throughout.')
p('Pooled log loss 0.691272143 and Brier 0.249061993 beat prevalence 0.691908047/0.249380363. Relative gains 0.0919%/0.1277%; accuracy 55.7248% vs 54.9089%. Gains are small and historical.')
table(['Forecast','Log loss','Brier','Accuracy'],[[labels[k],f'{metric(f,k)["log_loss"]:.9f}',f'{metric(f,k)["brier"]:.9f}',f'{100*metric(f,k)["accuracy"]:.3f}%'] for k in ['prevalence','fair','p_up','candidate']],[310,160,160,125])
p('Diagnostic candidate 0.686161080/0.246523019 is explicitly excluded from user-selected policy because January/October forecasts failed the earlier acceptance checks. Guard approval itself is no guarantee of future improvement.','small')
page('Exact selected quarterly policy and actual retained inputs')
rows=[]
for plan in plans['original']:
 t=plan['selected'];c=next((c for c in active if c['issue']==plan['issue']),None)
 rows.append([plan['issue'][:10],t['spec']['model'] if t else 'prevalence',t['spec']['pack'] if t else '-',t['weight'] if t else 0,len(c['features']) if c else 0,'PASS' if t else 'fallback'])
table(['Issue','Classifier','Candidate pack','Model weight','Retained inputs','Status'],rows,[90,150,175,100,130,110])
p('Important distinction: April compared the384-feature original inventory but training-only quality screening retained 157 base inputs. July compared the5-feature core but retained only 2 inputs. January/October have no learned model inputs in the selected forecast.')
p('p_up = weight*classifier_probability+(1-weight)*training prevalence. p_down=1-p_up. Predicted class is UP when p_up>=0.5;otherwise non-UP. Canonical zeros belong to non-UP,so p_down is not a separate strict-negative probability.')
page('Family-policy rankings | same evaluation timestamps')
rr=[]
for k in labels:
 if k=='candidate':continue
 m=metric(f,k);rr.append([labels[k],f'{m["log_loss"]:.9f}',f'{m["brier"]:.9f}',f'{100*m["accuracy"]:.3f}%',f'{m["auc"]:.4f}'])
rr.sort(key=lambda r:float(r[1]));table(['Policy / baseline','Log loss','Brier','Accuracy','AUC'],rr,[275,145,145,100,90],True)
p('Family columns are guarded policies and may use prevalence. HGB,RandomForest and bothGRU variants fell back in all four quarters. ExtraTrees is active onlyJuly;MLPs activeApril/July;logistic activeApril/July. Thus equal baseline scores are not proof trained models had identical raw predictions.','small')
for issue in f.issue.unique():
 page('Quarterly scores | '+pd.Timestamp(issue).strftime('%B %Y'))
 g=f[f.issue==issue]
 table(['Forecast','Rows','Log loss','Brier','Accuracy'],[[labels[k],len(g),f'{metric(g,k)["log_loss"]:.9f}',f'{metric(g,k)["brier"]:.9f}',f'{100*metric(g,k)["accuracy"]:.3f}%'] for k in ['prevalence','fair','p_up','candidate','extra_trees','mlp_logloss','mlp_brier']],[275,80,155,145,100])
 p('October issue continues through the finalJanuary2025 origins. These are issue-based periods,not four equal-duration independent tests.','small')
page('Probability losses and baselines | exact definitions')
p('For binary outcome y=1 if endpoint return>0,else0,and UP probability p: log loss = -mean[y*ln(p)+(1-y)*ln(1-p)]. Predictions clipped to[1e-7,1-1e-7] for scoring. Brier = mean[(p-y)^2],using the single-UP-probability convention,not twice that value.')
p('Both are proper probability scores: smaller is better. Log loss penalizes confident wrong predictions more strongly. Accuracy uses threshold0.5;AUC measures ranking and does not measure calibration. Equal0.5 predictsUP under the>=0.5 tie rule,so its accuracy equals the observed UP rate.')
p('Prevalence baseline = mean(y_UP) over all eligible labels with label_end<the fitting cutoff. The baseline is refitted per issue/window. 50/50 baseline is constant0.5. Neither uses future class frequencies.')
p('Classification outputs probability and class,not return magnitude. To combine with NET later,align horizon,timestamps,label definitions and units,then evaluate signed-return errors separately. Probability-score gains do not guarantee signed-return gains.')
page('April MLP Brier classifier | architecture and optimization')
apr=next(c for c in active if c['spec']['model']=='mlp_brier');p('157 base inputs ->314 transformed inputs ->Dense 16 ->ReLU ->Dense 1 ->sigmoid. One hidden layer;'+str(apr['parameters'])+' learned parameters. No dropout. Output weights startzero;output bias=logit(training prevalence).')
p('Loss=mean[(sigmoid(logit)-y)^2]. AdamWlr0.0005,weight_decay0.1,16 epochs,batch256,gradientnormclip1,seed48,CPUfloat32,deterministic torch algorithms. Each epoch uses a deterministic random permutation. No oversampling,class weighting or random early-stopping split.')
p('At inference,p_model=sigmoid(W2*ReLU(W1*z+b1)+b2). Selected April p_up=.5*p_model+.5*training prevalence. All preprocessing/statistics come from mature training only.')
table(['Issue','Base inputs','Transformed','Parameters','Mature rows','Estimator rows'],[[apr['issue'][:10],len(apr['features']),apr['inputs'],apr['parameters'],apr['training_rows'],apr['estimator_rows']]],[100,100,130,120,155,150])
page('July ExtraTrees classifier | fitted settings')
jul=next(c for c in active if c['spec']['model']=='extra_trees');p('Training retained only these two of the five core features:')
table(['Exact retained feature'],[[n] for n in jul['features']],[755],True)
p('32 randomized trees;criterion=log_loss,max_depth5,min_samples_leaf32,max_features0.75,bootstrap=True,max_samples1024,n_jobs1,seed48. Predict probability by averaging treeleafclass proportions. Selected July weight1: no extra prevalence shrinkage.')
p('Fit uses up to2,000 evenly spaced mature rows. All mature rows supply feature-quality screening,imputation/scaling and baseline prevalence. Randomized tree splits differ from RandomForest best-threshold splits.')
table(['Issue','Mature rows','Estimator rows','Latest training label end'],[[jul['issue'][:10],jul['training_rows'],jul['estimator_rows'],str(jul['max_label_end'])]],[100,180,180,295])
page('All candidate classifiers | architecture and objectives')
table(['Variant','Architecture and training'],[['Logistic','L2 logistic,C.01,maxiter300. Native log-loss fit.'],['HGB','Loglossboosting40rounds,depth3,7leaves,leaf64,L2reg10,rate.05,noearlystop.'],['RandomForest','32 trees,Gini,depth5,leaf32,maxfeatures.75,bootstrap1024.'],['ExtraTrees','32 trees,loglossimpurity,depth5,leaf32,maxfeatures.75,bootstrap1024.'],['MLP logloss / Brier','16ReLUhidden,16 epochs;BCEwithlogits or squaredprobabilityloss.'],['GRU logloss / Brier','GRU8+linear,16 epochs;BCEwithlogits or squaredprobabilityloss;36/24/12/0h snapshots.']],[180,575])
p('Gini is related to Brier impurity,but RandomForest is not trained by the neural Brier optimizer. Trees use native split objectives;all forecasts evaluated using both held-out probability metrics. Neuraloptimizer settings shared with AprilMLP. Logistic and tree configs saved in supportmetadata.')
page('Population, source labels and preprocessing')
p('CanonicalBTCUSDT horizon2880minutes. Source-run checksums verified againstOUTPUT_FILES receipt;VALID labels;UP iff direction>0;zerosnon-UP. Original48,096origins ->48,000eligible;18,385 evaluation observations after 96 binding-crossing exclusions. Half-hour playground grid,not full canonicalminute population.')
p(f'Coverage: {f.decision_time.min()} to {f.decision_time.max()}. ActualUP rate {100*f.y_up.mean():.4f}%; zero endpoint count {int((f.signed_return_log_bps==0).sum())}. Actual labels never imputed.')
p('Features come directly from original maskedX_train/X_development,using the historical directioninventory. Magnitude model features/predictions are not reused. Fulltimeline directionfeatures are constructed before filtering future label availability.')
p('Perfit retain columns with>=80%nonmissing mature training coverage and more than one distinct value. Median imputation;z=clip((x-trainingmedian)/max(trainingIQR,1e-6),-8,8);append one missing flag per retained input. Thus dinputs become2d. There is no signedlog transform in this directionrunner.')
page('Validation and acceptance | distinct from diagnostic candidate')
p('Jan/Apr/Jul/Oct2024 expanding fits. Four earlier3monthwindows start15/12/9/6months before eachissue. Training label_end<fitcutoff;validation label_end<windowend and<outerissue;48h strict maturity. No outer score selects its own issue model,featurepack or weight.')
p('8variants x3packs x4windows x4issues =384inner classifier fits. Each fitted candidate is scored with probabilityweights.25/.5/.75/1 toward training prevalence,without refitting.384trialrecords encode those candidate/weight/issue combinations;window metrics nested within each.')
p('Rank mean(pooledlogloss/prevalencelogloss,pooledBrier/prevalenceBrier). Accept>=0.1%pooledgain BOTH,>=3/4windowwins/ties,and no>2%windowloss for each. Otherwise prevalencefallback. Bestcandidate ignores acceptance only for diagnostic reporting. Classificationaccuracy is never the selection objective.')
p('The report verifies saved candidate/accepted minima against ledgers;it does not independently reconstruct each window\'s training baseline labels. Original 384featureinventory was historically selected;the repeated evaluation is exposed,not an untouched test.')
page('Direction feature packs and selected inventory')
table(['Pack','Candidate inputs','Contents'],[['direction_core',5,'Historical directioncore;trainingquality screening may reduce it.'],['direction_original',384,'Historical directioninventory:basis40,flow85,funding32,liquidation92,OI30,price45,volatility60.'],['direction_new_state',99,'Core5+94new past-only state/interaction measurements.']],[190,130,435])
p(f'The selected learned policy uses {len(used)}distinct base features:April 157 andJuly 2,with overlap allowed. January/October use prevalence only. Exact active identifiers and quarter membership follow; actual fitted order is in model_metadata.json. The full 384/core5/new99 candidate inventories are retained in original_CONTRACT.json.')
p('F4Z percentile/robust-z source normalization and upstream availability masks are inherited. This report preserves exact identifiers and auditable downstream formulas;it does not claim to reconstruct upstream reference-distribution formulas. Feature membership is not individual causal importance.')
for start in range(0,len(membership),8):
 page(f'Exact active feature inventory | {start+1}-{min(start+8,len(membership))}')
 table(['Base feature identifier','Jan','Apr','Jul','Oct'],[[r['feature'],r['Jan'],r['Apr'],r['Jul'],r['Oct']] for r in membership[start:start+8]],[475,70,70,70,70],True)
page('New direction formulas | notation and source channels')
p('New features are tested hypotheses;none is active in the selected April/July policy. Seven 1h masked source channels:flowpressurepercentile,CVDpathefficiencypercentile,fundingchangepercentile,basischangepercentile,basispersistencepercentile,OIrange-positionpercentile,leverage-unwindpercentile.')
p('For source x,lag=x(t-30min). For h=24/48/120h,use snapshots in[t-h,t),requiring80%expected half-hour observations. Mean=historicalmean;deviation=lag-mean;z=clip(deviation/max(populationstd,1e-6),-5,5);change=lag(t)-lag(t-h). Missing half-hour gridrows remainmissing.')
p('Each source yields lag plus4measurements x3windows =13features. Seven x13=91;add3centeredproducts =94. Products pair flowpressure withbasischange,flowpressure withleverageunwind,andfundingchange withbasischange: (lagA-.5)*(lagB-.5). Percentilemidpoint centering does not establish a signed buy/sell effect.')
p('Newpack versus core/original comparisons use earlier windows only. For some models new features improved validation,but April selected original andJuly selected core;do not claim new features caused the selected pooled gain.')
for start in range(0,len(new),7):
 page(f'Exact new direction formulas | {start+1}-{min(start+7,len(new))}')
 table(['Feature identifier','Implemented formula / source'],[[r.feature,r.formula] for r in new.iloc[start:start+7].itertuples()],[315,440],True)
page('Calibration follow-up | did not improve the guarded result')
c=data['calibrated']
table(['Forecast','Log loss','Brier','Accuracy'],[[label,f'{metric(df,col)["log_loss"]:.9f}',f'{metric(df,col)["brier"]:.9f}',f'{100*metric(df,col)["accuracy"]:.3f}%'] for label,df,col in [('Previous selected',f,'p_up'),('Refined selected',c,'p_up'),('Previous candidate',f,'candidate'),('Refined candidate',c,'candidate'),('Prevalence',f,'prevalence')]],[300,165,165,125])
p('Follow-up tested HGB/ExtraTreescontrol/regularizedsettings;raw,shrunkintercept andL2Plattcalibration. Each validationblockcalibrator used only earlier purgedOOFblocks;first block identity. Finalcalibration used oldOOFlabels maturebeforeissue. Originalacceptanceguard retained. ItsAprilcalibratedHGBreplacement worsened pooled selected scores.')
p('The new selected policy is not promoted. The earlier diagnosticcandidate remains lowest observed logloss in these two experiments,but failed originalstabilitychecks and is not the reported chosen forecast.')
page('Audit, evidence and interpretation limits')
p(f'Two prediction files hash-verified;{model_count}savedclassifier artifact hashes verified.18,385 unique timestamps and labels/signedreturns/prevalence aligned. Allsavedpooled/period logloss,Brier,accuracy,AUC recomputed. {trial_count}validationrecords retained;candidate/accepted minima checked. Activefeatures,preprocessing,neuralstates andestimatorparameters inspected.')
p('No training or new inference performed for reportcreation. Modelreplay reported by originalCOMPLETEfiles,not rerun. Source inputchecksums and causalprefix/futurechecks were performed by originalrunner/audit;not rerun here. UpstreamF4Znormalization inherited.')
p('Small historical probability-score gains do not establish statisticalsignificance,profitabletrading or better signedreturnerrors. Overlapping48hlabels reduce independent information. Familyguardedpolicies differ from rawclassifiers;fixedhistoricalinventory and repeatedmodelsearch cause adaptiveexposure. Freshchronologicalevaluation remains required.')
for key,path in RUNS.items():p(key+': '+str(path.relative_to(ROOT)),'small')
p('Support: selectedpolicy/predictionextract,allmodelmetadata,periodmetrics,activefeaturemembership,exact94newformulas,fullcontracts/packinventories,validationledgers,calibration/featurecomparison audits,sourcesnapshots andhashaudit. Modelbinaries stay in originalrunfolders.','small')
def footer(canvas,doc):
 canvas.setStrokeColor(TEAL);canvas.line(40,34,802,34);canvas.setFont('Helvetica',9);canvas.setFillColor(NAVY);canvas.drawString(40,21,'BTCUSDT | 48h direction classification | guarded quarterly policy | 16 Nov2026');canvas.drawRightString(802,21,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=48,title='BTC48h direction - selected policy,models,features and scores',author='ScaleWave research');doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(PDF));txt='\n'.join(p.extract_text() for p in reader.pages)
assert all(v in txt for v in ['0.691272143','0.249061993','0.686161080','ExtraTrees'])
assert all(n in txt.replace('\n','') for n in used)
dump(SUP/'audit.json',dict(pdf=str(PDF),pdf_sha256=sha(PDF),pages=len(reader.pages),prediction_runs=2,model_hashes=model_count,validation_records=trial_count,observations=18385,active_feature_union=len(used),new_features=94,hashes=hashes,training=False,inference=False))
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),models=model_count,activefeatures=len(used),support=str(SUP))))
