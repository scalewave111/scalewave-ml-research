"""Generate factual model comparison from saved results; no training."""
from pathlib import Path
import sys,json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape
R=Path(__file__).resolve().parents[1]; E=R/'Experiments'
O=R/'output/pdf'; O.mkdir(parents=True,exist_ok=True)
T=R/'tmp/pdfs/model_review'; T.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#13394C'),spaceAfter=12))
styles.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=colors.HexColor('#137A83'),spaceBefore=11,spaceAfter=7))
styles.add(ParagraphStyle(name='BodyX',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='CellX',fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name='SmallX',fontSize=8,leading=11,spaceAfter=6))
story=[]
def p(t,style='BodyX'): return Paragraph(t,styles[style])
def add(t,style='BodyX'): story.append(p(t,style))
def title(k,t): add('SCALEWAVE  |  MODEL RESULTS  |  03 NOVEMBER 2026','SmallX');add(t,'TitleX');add(k,'SmallX')
def table(rows,widths):
 header_style=ParagraphStyle('HeaderCell',parent=styles['CellX'],textColor=colors.white,fontName='Helvetica-Bold')
 data=[[Paragraph(escape(str(v)),header_style if i==0 else styles['CellX']) for v in row] for i,row in enumerate(rows)]
 tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#13394C')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F6F7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#137A83'))]))
 for cell in data[0]: cell.style=ParagraphStyle('white',parent=styles['CellX'],textColor=colors.white,fontName='Helvetica-Bold')
 story.append(tab);story.append(Spacer(1,8))
def page():story.append(PageBreak())
records=[]
def rec(name,row,source):records.append(dict(model=name,mae=float(row['mae']),ic=float(row['rank_ic']),source=source))
def score(folder,model,name):
 d=pd.read_csv(E/folder/'scores.csv');q=d[d.split.eq('test')]
 if 'model' in q:q=q[q.model.eq(model)]
 rec(name,q.iloc[0],folder+'/scores.csv')
c=pd.read_csv(E/'all_model_hybrid_v1/component_scores.csv').set_index('model')
for key,name in [('baseline','Baseline: RF + median regression'),('yesterday:v1_RF__blend_0.25','Earlier hybrid: 25% RF + 75% pair'),('yesterday:v1_RF','Earlier standalone Random Forest'),('yesterday:frozen_pair','Original frozen median regression'),('reference:report_pair','Expanding median regression'),('reference:report_rf','Expanding two-feature Random Forest')]:
 rec(name,c.loc[key],'all_model_hybrid_v1/component_scores.csv')
h=pd.read_csv(E/'all_model_hybrid_v1/HYBRID_COMPARISON.csv').set_index('model')
for key,name in [('family_balanced','Family-balanced ensemble'),('all_model_equal','Equal-weight all-model ensemble'),('selected_hybrid','Walk-forward selected all-model hybrid')]:
 rec(name,h.loc[key],'all_model_hybrid_v1/HYBRID_COMPARISON.csv')
for f,m,n in [('residual_incremental_v1','selected','Baseline + residual correction'),('filtered_walkforward_v1','selected','Filtered walk-forward selection'),('targeted_challenger_no_fallback_v1','expanded_hybrid','Expanded-feature RF hybrid'),('targeted_challenger_no_fallback_v1','expanded_rf','Expanded-feature standalone RF'),('three_feature_tree_tuning_v1','hybrid','Three-feature tuned hybrid'),('three_feature_tree_tuning_v1','standalone','Three-feature tuned tree'),('tree_document_methods_v2',None,'Document-method CatBoost / LightGBM')]:score(f,m,n)
q=json.loads((E/'tree_improvement_campaign_v3/48h/COMPLETE.json').read_text())
rec('Regime-aware tree improvement campaign',next(x for x in q['scores'] if x['split']=='test'),'tree_improvement_campaign_v3/48h/COMPLETE.json')
mh=pd.read_csv(E/'tree_magnitude_filter_v1/scores_all_horizons.csv')
rec('Eight-test filtered tree models',mh[(mh.horizon_hours==48)&(mh.split=='test')].iloc[0],'tree_magnitude_filter_v1/scores_all_horizons.csv')
records.sort(key=lambda x:x['mae'])
icr={r['model']:i+1 for i,r in enumerate(sorted(records,key=lambda x:-x['ic']))}
pd.DataFrame(records).to_csv(O/'model_ranking_sources.csv',index=False)
title('48-hour return magnitude | 18,481 evaluation rows | 01 Jan 2024 - 22 Jan 2025','Model ranking')
add('Ranked by lowest test MAE. The IC rank independently orders the same models by highest Spearman IC. Scores are rounded to three MAE decimals and five IC decimals.')
table([['MAE rank','Model / strategy','MAE (bps)','IC','IC rank']]+[[i+1,r['model'],f"{r['mae']:.3f}",f"{r['ic']:.5f}",icr[r['model']]] for i,r in enumerate(records)],[40,252,65,63,43])
add('<b>Lowest MAE:</b> baseline hybrid, 181.526 bps. <b>Highest IC in this table:</b> residual correction, 0.08465. No listed model achieved 48-hour MAE below 180 bps.','SmallX')
add('The baseline row uses the saved reconstruction. The original PDF reports 181.523868 bps; the saved reconstruction scores 181.525993 bps. The earlier gated feature-extension run returned the baseline and is not counted as another model.','SmallX')
page()
title('Model inputs and construction','What each model used')
table([['Model group','Features used','Technique'],
['Baseline and expanding two-feature RF','Liquidation historical active rate at 5m and 15m.','Quarterly expanding-history training; shallow RF; baseline blends RF with median regression.'],
['Earlier standalone RF / 181.800 hybrid','89-92 liquidation columns per window; 93 distinct columns across windows. Hybrid also uses the frozen two-feature pair.','Monthly refits using trailing three months; RF hybrid uses 25% RF and 75% pair.'],
['Original / expanding median regression','The same two liquidation active-rate columns.','Predicts the conditional median; original model fixed, expanding version periodically refitted.'],
['All-model ensembles','Inputs inherited from constituent models: liquidation, filtered flow/volatility/funding, engineered inputs and history.','Combine prediction streams equally, balance model families, or select weights using earlier windows.'],
['Residual correction','Baseline prediction plus selected 24h-average volatility feature in January; baseline forecast as correction input in April.','Three-bin median correction learned from earlier prediction errors. July and October keep baseline.'],
['Filtered walk-forward selection','Original pair plus screened flow, volatility, funding; engineered changes, rolling averages and mature target history.','Compare feature sets, history lengths, CatBoost/LightGBM and blends in earlier validation.'],
['Eight-test filtered trees','Original pair; selected additions: liquidation venue breadth, its 24h average, and signed-log 5m control.','Feature tests, permutation and add/remove comparisons; RF/CatBoost selected by window.'],
['Regime-aware campaign','Mostly original controls; January also uses trailing 4h mean absolute return. Routing uses past 168h volatility and 7-day return.','Monthly tree ensembles, regime experts, target transformations and recency weights.'],
['Document-method models','Selected 5m and/or 15m liquidation controls.','CatBoost / LightGBM; history-length selection; early stopping; VIF filtering.'],
['Expanded-feature RF / hybrid','27, 16, 10 and 18 inputs by quarterly window; 35 distinct inputs overall: 29 existing + 6 engineered.','Candidate pools cover liquidation, flow, volatility, funding, basis and price; RF plus optional median-regression blend.'],
['Latest three-feature models','Original two liquidation controls + 4h flow activity RZ; three missing-value flags.','Tune RF, CatBoost and LightGBM; select trees and blend weights using earlier validation.']
],[115,202,146])
add('A feature is an input column. A missing-value flag is an extra column marking whether the original input was missing.','SmallX')
page()
title('Recorded feature tests and model importance','Features that contributed')
add('Original two liquidation inputs','SubX')
add('The lowest-MAE model uses these two market inputs. Together they account for all market-feature inputs to the baseline:')
for f in json.loads((E/'three_feature_tree_tuning_v1/COMPLETE.json').read_text())['features'][:2]:
 add(escape(f).replace('__','__<wbr/>'),'SmallX')
add('They describe historical liquidation activity across assets at 15-minute and 5-minute scales.')
add('Largest accepted validation improvements','SubX')
add('The residual-correction experiment tested inputs individually. These are the top six accepted tests, ranked by validation MAE reduction. Positive numbers mean improvement over the comparator in that validation window.')
fa=pd.read_csv(E/'residual_incremental_v1/feature_admission.csv')
top=fa[fa.eligible].sort_values('mae_gain',ascending=False).head(6)
labels={'MEAN24H__F4Z__volatility__realized_variance_level_percentile__5m__value':'24h average of 5m volatility percentile','BASE__forecast':'Baseline model forecast','HIST__mean_30d':'30-day mean of matured target magnitudes','F4Z__funding__funding_change_percentile__15m__value':'15m funding-change percentile','MEAN24H__F4Z__funding__funding_level_rz__1h__value':'24h average of 1h funding-level RZ','HIST__median_30d':'30-day median of matured target magnitudes'}
table([['Input','Window start','MAE gain (bps)','IC gain']]+[[labels.get(x.feature,x.feature),x.outer[:10],f'{x.mae_gain:.3f}',f'{x.ic_gain:.5f}'] for x in top.itertuples()],[245,78,72,68])
add('The final correction selected the volatility average in January and baseline forecast in April. Its full-period test result was MAE 181.866 bps and IC 0.08465.')
add('Latest addition: 4-hour flow activity','SubX')
add('Exact column: <b>F4Z__flow__flow_activity_rz__4h__value</b>')
im=pd.read_csv(E/'three_feature_tree_tuning_v1/flow_internal_importance.csv')
table([['Window','Tree configuration','Flow importance share']]+[[x.issue[:10],x.config,f'{100*x.flow_internal_importance_fraction:.1f}%'] for x in im.itertuples()],[95,230,138])
add('These shares measure internal tree importance, including the flow missing-value flag. LightGBM uses split counts and Random Forest uses impurity reduction. They are model-use scores, not percentages of MAE improvement.','SmallX')
add('The latest three-feature tree scored MAE 184.198 / IC 0.06204; its hybrid scored 182.837 / 0.05318.','SmallX')
page()
title('Plain-language explanation of the methods actually used','Techniques and terms')
table([['Term / technique','Simple explanation and use'],
['MAE','Average size of prediction error. Lower is better. It is measured here in basis points (bps); 100 bps = 1 percentage point.'],
['Spearman IC','Checks whether larger predicted moves correspond to larger actual moves. Higher is better. It measures ordering, not exact prediction size.'],
['48-hour magnitude','Absolute endpoint log return over the next 48 hours, expressed in bps. It measures move size without up/down direction.'],
['Walk-forward','Train on earlier observations, then predict the next time period; repeat as time moves forward.'],
['Nested validation / label purge','Choose features and settings inside earlier training history. Exclude training labels whose 48-hour outcome has not finished before the next window.'],
['Random Forest (RF)','Many decision trees trained on sampled data; combine their predictions.'],
['CatBoost / LightGBM','Boosted trees: each new tree helps correct the errors of earlier trees.'],
['Median regression / hybrid','Median regression predicts a typical middle outcome. A hybrid combines predictions from different models using weights.'],
['Missingness / variance tests','Remove inputs with too much missing data or almost no variation.'],
['Correlation / VIF','Check whether inputs repeat information already contained in other inputs. VIF measures how well other columns explain one column.'],
['Spearman / mutual information','Screen for relationships with the target: ordered relationships and more general nonlinear relationships.'],
['KS / lift','Compare feature distributions for large versus ordinary moves; compare typical target size across feature groups.'],
['Permutation / ablation','Shuffle an input or remove it, then measure how predictions change. Used in the feature-filtering experiments.'],
['Feature engineering','Create rolling averages, changes, ratios, absolute values and deviations from past averages using earlier data.'],
['Feature transformation','Rescale inputs or apply signed-log transforms. RZ denotes robust z-score scaling; percentile expresses relative position.'],
['Hyperparameter tuning / regularization','Try tree settings such as depth and leaf size. Restrict model complexity with shallow trees, larger leaves and penalties.'],
['Early stopping / target transformation','Stop boosting using an earlier validation window; test log, square-root or volatility-normalized targets in relevant campaigns.'],
['Regime models / ensembles','Use different models for market states such as volatility/trend conditions, or combine several model predictions.'],
['Residual correction / error analysis','Model the errors of an existing forecast. Review errors by time period and magnitude group.']
],[145,318])
page()
title('Latest experiment | original pair + 4-hour flow','Latest model details')
table([['Window start','Standalone tree','Hybrid composition'],
['2024-01-01','LightGBM, depth 1','100% LightGBM'],
['2024-04-01','LightGBM, depth 2','50% LightGBM + 50% median regression'],
['2024-07-01','Random Forest, depth 3','25% LightGBM + 75% median regression'],
['2024-10-01','Random Forest, depth 5','10% RF + 90% median regression']
],[95,173,195])
add('Eight configurations were evaluated: three Random Forest, two CatBoost and three LightGBM. CatBoost was tested but not selected. There were 32 inner tuning fits and 5 final retraining fits: <b>37 fits in total</b>. A fit means training one model once.')
s=pd.read_csv(E/'three_feature_tree_tuning_v1/scores.csv')
table([['Split','Approach','MAE (bps)','Spearman IC']]+[[x.split,x.model,f'{x.mae:.3f}',f'{x.rank_ic:.5f}'] for x in s.itertuples() if x.split in ['training','validation','test']],[100,160,100,103])
add('Training: 144,140 model-row observations, including repeated timestamps across expanding windows; 42,575 unique timestamps. Validation: 17,088 rows. Test: 18,481 unique rows.','SmallX')
add('Last-three-month results: separate comparison','SubX')
n=pd.read_csv(E/'nested_boost_48h_v1/scores.csv')
rows=[]
for key,name in [('previous_rf','Earlier standalone RF'),('report_hybrid','Baseline hybrid'),('previous_hybrid','Earlier RF25% hybrid'),('failed_nested_rf','Nested RF'),('catboost','Nested CatBoost'),('selected','Nested boosting selection'),('lightgbm','Nested LightGBM')]:
 x=n[(n.split=='test')&(n.model==key)].iloc[0];rows.append((name,x.mae,x.rank_ic))
for key,name in [('standalone','Latest three-feature tree'),('hybrid','Latest three-feature hybrid')]:
 x=s[(s.split=='last3months')&(s.model==key)].iloc[0];rows.append((name,x.mae,x.rank_ic))
table([['Model','MAE (bps)','Spearman IC']]+[[a,f'{b:.3f}',f'{c:.5f}'] for a,b,c in sorted(rows,key=lambda z:z[1])],[280,90,93])
add('This table uses 3,937 observations from November 2024 to January 2025. These scores are kept separate from the 18,481-row ranking.','SmallX')
page()
title('Additional recorded results and source files','Other results and references')
add('Feature engineering: measured comparison','SubX')
eng=pd.read_csv(E/'filtered_walkforward_v1/engineering_comparison.csv')
table([['Change','Before MAE','After MAE','Before IC','After IC']]+[[x.change,f'{x.before_mae:.3f}',f'{x.after_mae:.3f}',f'{x.before_ic:.5f}',f'{x.after_ic:.5f}'] for x in eng.itertuples()],[183,70,70,70,70])
add('All rows above use the full 48-hour evaluation period. Each row compares the saved before/after versions named in the experiment.','SmallX')
add('Earlier multi-horizon tree experiment','SubX')
table([['Horizon','Tree MAE','Tree IC','Matched RF MAE','Matched RF IC']]+[[f'{h}h',f"{mh[(mh.horizon_hours==h)&(mh.split=='test')].iloc[0].mae:.3f}",f"{mh[(mh.horizon_hours==h)&(mh.split=='test')].iloc[0].rank_ic:.5f}",f"{mh[(mh.horizon_hours==h)&(mh.split=='test_rf_baseline')].iloc[0].mae:.3f}",f"{mh[(mh.horizon_hours==h)&(mh.split=='test_rf_baseline')].iloc[0].rank_ic:.5f}"] for h in [1,4,24,48,72,120]],[63,100,100,100,100])
add('These are separate forecast targets. The latest three-feature experiment trained only the 48-hour target.','SmallX')
add('Source files','SubX')
add('Paths below are relative to C:/Users/Siddh/Desktop/ML/November/Week 1/. Scores were read from saved experiment outputs; no new model training was performed for this report.','SmallX')
sources=[
'Experiments/all_model_hybrid_v1/component_scores.csv; HYBRID_COMPARISON.csv; EARLIER_RF25_HYBRID_IC.json',
'Experiments/three_feature_tree_tuning_v1/scores.csv; SELECTIONS.json; COMPLETE.json; flow_internal_importance.csv; AUDIT.json',
'Experiments/targeted_challenger_no_fallback_v1/scores.csv; EXACT_FEATURES.txt',
'Experiments/residual_incremental_v1/scores.csv; feature_admission.csv; outer/*_selection.json',
'Experiments/filtered_walkforward_v1/scores.csv; engineering_comparison.csv',
'Experiments/tree_improvement_campaign_v3/48h/COMPLETE.json',
'Experiments/tree_document_methods_v2/scores.csv',
'Experiments/tree_magnitude_filter_v1/scores_all_horizons.csv; selected_features_all_horizons.csv',
'Experiments/nested_boost_48h_v1/scores.csv',
'Code/tree_magnitude_filter.py; tree_document_methods.py; residual_incremental.py; three_feature_tree_tuning.py']
for x in sources:add(escape(x),'SmallX')
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#C9D8DE'));c.line(42,38,553,38)
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#54707E'));c.drawString(42,25,'ScaleWave | Recorded model results | 48h primary comparison');c.drawRightString(553,25,str(d.page))
pdf=O/'ScaleWave_Model_Results_Ranked.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=48,title='ScaleWave - Ranked Model Results',author='ScaleWave research').build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
reader=PdfReader(pdf)
assert len(reader.pages)==6, f'Unexpected pages: {len(reader.pages)}'
text='\n'.join(pg.extract_text() for pg in reader.pages)
for word in ['181.526','0.08465','184.198','37 fits','Features that contributed']:assert word in text,word
(T/'extracted.txt').write_text(text,encoding='utf-8')
sys.path.insert(0,str(R/'Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/tmp/pdfs/deps'))
import pypdfium2 as pdfium
doc=pdfium.PdfDocument(str(pdf))
for i,pg in enumerate(doc):pg.render(scale=1.4).to_pil().save(T/f'page_{i+1}.png')
print(json.dumps({'pdf':str(pdf),'pages':len(reader.pages),'ranked_models':len(records),'rendered_pages':len(doc)}))
