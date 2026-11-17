"""Beginner report from recorded CSVs; no training or inferred results."""
from pathlib import Path
import sys,json
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'Code/experiments';OUT=ROOT/'Report/ScaleWave_48h_Features_and_Model_Results_2026-11-05_Revised.pdf';TMP=ROOT/'tmp/pdfs/feature_results_report_revised';TMP.mkdir(parents=True,exist_ok=True)
OUT=OUT.with_name('ScaleWave_48h_Features_and_Model_Results_2026-11-05_Ranked.pdf')
hy=pd.read_csv(E/'individual_hybrid_additions_v1/individual_results.csv');ml=pd.read_csv(E/'individual_mlp_additions_v1/individual_contributions.csv');ms=pd.read_csv(E/'individual_mlp_additions_v1/scores.csv').set_index('model');groups=pd.read_csv(E/'two_group_tuning_v1/scores.csv').set_index('model');broad=pd.read_csv(E/'report38_subset_search_v2/scores.csv').set_index('model')
features=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value','PRICE__vol_ratio_4h_24h','F4Z__liquidation__venue_breadth_percentile__1h__value','PRICE__realized_vol_168h','F4Z__basis__basis_level_rz__1h__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','F4Z__funding__funding_persistence_percentile__5m__value']
navy=colors.HexColor('#19384B');teal=colors.HexColor('#087E8B');W,H=landscape(A4);width=W-80
styles={k:ParagraphStyle(k,fontName=f,fontSize=z,leading=l,textColor=c,spaceAfter=sp) for k,f,z,l,c,sp in [('title','Helvetica-Bold',24,29,navy,10),('h','Helvetica-Bold',13,17,teal,8),('body','Helvetica',10.5,14,navy,9),('small','Helvetica',9,12,navy,7),('cell','Helvetica',8.5,11,navy,0),('head','Helvetica-Bold',8.5,11,colors.white,0),('code','Courier',8.3,11,navy,0)]}
story=[]
def p(text,style='body'):return Paragraph(text,styles[style])
def add(text,style='body'):story.append(p(text,style))
def page(title,subtitle):
 if story:story.append(PageBreak())
 add('SCALEWAVE | 5 NOVEMBER 2026','small');add(title,'title');add(subtitle,'small')
def table(headers,rows,widths,codefirst=False):
 data=[[p(escape(str(v)),'head') for v in headers]]+[[p(escape(str(v)),'code' if codefirst and j==0 else 'cell') for j,v in enumerate(row)] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)])
def fmt(v):return f'{v:.3f}'
def ic(v):return f'{v:.5f}'
page('48-hour prediction: what we tested','Recorded model results and exact feature names | Main model comparisons: 18,481 historical evaluation rows')
add('<b>Target:</b> the size of Bitcoin\'s price move from now to 48 hours later, measured as absolute endpoint log return. These models predict move size, not whether price goes up or down.')
table(['Term','Simple explanation'],[['MAE','Average absolute prediction error, in log basis points. Lower is better. It is not an accuracy percentage.'],['IC','Spearman rank correlation: whether predicted larger moves tend to be larger in reality. Higher is better; it is not an accuracy percentage.'],['Hybrid','A weighted combination of median regression and Random Forest regression. Median regression is linear quantile regression at 0.5.'],['MLP','A small neural network. Our MLP starts with a median-regression prediction and learns a nonlinear correction.']],[90,width-90])
add('What does core mean?','h')
add('<b>Core</b> means the starting set of three input features used together. It is a feature group, not a separate model. Both the hybrid and MLP can use this same group.','small')
for name in features[:3]:add(escape(name),'code')
story.append(Spacer(1,8))
add('<b>Core only:</b> the model uses these three features. <b>Core + one feature:</b> the model uses these three plus the named extra feature, giving four inputs.','small')
add('What we did','h')
add('1. Compared the original two-feature baseline with a three-feature core.<br/>2. Tested each extra feature individually with the hybrid and MLP.<br/>3. Compared seven- and eight-feature groups.<br/>4. Checked whether hybrid hyperparameter tuning improved validation performance and measured the later results.')
add('Training used earlier data and evaluation used later data in four expanding windows. A strict 48-hour label boundary prevented training outcomes from crossing the next cutoff. Validation data chose settings; evaluation data measured the resulting performance.','small')
meanings=[
'Uses 5-minute liquidation totals from the two designated peer assets, excluding Bitcoin. The value is the historical fraction of prior observations with liquidation activity. It measures frequency, not liquidation size.',
'Uses 15-minute liquidation totals from the same peer assets. The value is the historical fraction of prior observations with activity. The 15m label describes the underlying liquidation window.',
'Compares price volatility over the past 4 hours with volatility over the past 24 hours, adjusted for window length. Above 1 means recent volatility is higher on this comparable basis; below 1 means lower.',
'Measures how widely liquidation activity is spread across eligible exchanges over 1 hour, then ranks that share against its history. A higher percentile means unusually broad exchange participation.',
'Measures price fluctuations over the past 168 hours (7 days). It is calculated as the square root of the sum of squared 5-minute log returns. Larger values mean greater recent price movement.',
'Uses the latest mark-price versus index-price gap from the 1-hour window, combined across exchanges. The rz transform expresses how far that gap is above or below its historical median, using a robust scale. A positive rz means above its usual level; it does not by itself mean mark is above index.',
'Uses 1-hour liquidation totals from the two peer assets, excluding Bitcoin. It reports the historical fraction of prior observations with liquidation activity. In our evaluation data, this column was always 1.0.',
'Measures how much of the 5-minute funding path stayed on the same side of zero as its ending value, then ranks that measure against history. A higher percentile means unusually persistent funding direction, not necessarily a larger funding rate.'
]
for first,last,title in [(0,4,'Feature meanings: 1 to 4'),(4,8,'Feature meanings: 5 to 8')]:
 page(title,'Exact original column names, with meanings checked against the source dictionary and price-feature code.')
 for i in range(first,last):
  add(str(i+1)+'. '+escape(features[i]),'code');story.append(Spacer(1,6));add(meanings[i]);story.append(Spacer(1,8))
 if first==0:add('<b>Our three-feature core:</b> features 1, 2 and 3. The original baseline used features 1 and 2.','small')
 else:add('<b>Percentile:</b> a feature\'s relative position compared with its own historical values. <b>rz:</b> distance from its historical median, divided by a robust measure of spread.','small')
page('Hybrid: individual feature additions','Each row adds exactly one feature to the same three-feature core. Results use the same 18,481 evaluation rows.')
rows=[]
for row in hy.to_dict('records'):
 rows.append(['Core only (no extra feature)' if row['pack']=='core' else row['added_feature'],fmt(row['untuned_mae']),ic(row['untuned_ic']),fmt(row['selected_mae']),ic(row['selected_ic'])])
table(['Extra feature: original column name','Original settings MAE','Original settings IC','Tuning-selected MAE','Tuning-selected IC'],rows,[440,*([(width-440)/4]*4)],True)
add('<b>Tuning rule:</b> use an alternative configuration only if it reduces validation MAE by at least 0.1 log bps and improves at least two validation months. Otherwise retain the original settings. The selected blend weight also uses validation data.','small')
add('<b>Measured result:</b> adding venue breadth, 168h realized volatility, or basis level individually lowered MAE with original hybrid settings. Each had lower IC than the core. Tuning worsened these three additions on evaluation, despite passing the earlier validation rule.','small')
add('Original two-feature baseline: <b>MAE 181.526; IC 0.07739</b>. Three-feature core: <b>MAE 181.248; IC 0.10680</b>.','small')
page('MLP: individual additions and group tests','Individual-addition experiment: fixed hidden layers of 8 and 4 units, dropout 0.05, one random seed (48), same validation early-stopping rule.')
rows=[['Core only (no extra feature)',fmt(ms.loc['core','mae']),ic(ms.loc['core','rank_ic']),'--']]
for row in ml.to_dict('records'):rows.append([row['added_feature'],fmt(row['mae']),ic(row['ic']),f"{row['mae_change_vs_matched_core']:+.3f}"])
table(['Extra feature: original column name','MAE','IC','MAE change vs core'],rows,[480,75,75,width-630],True)
add('<b>Basis level</b> gave the largest individual MLP MAE improvement and also improved IC. None of the five individual MLP additions beat the original 181.526 baseline. Negative MAE change means improvement over the matched MLP core.','small')
add('Separate seven/eight-feature experiment','h')
rows=[]
for label,rkey,mkey in [('Seven features','rf_seven','mlp_seven'),('Eight features','rf_eight','mlp_eight'),('Group chosen by validation','rf_selected','mlp_selected')]:rows.append([label,fmt(groups.loc[rkey,'mae']),ic(groups.loc[rkey,'rank_ic']),fmt(groups.loc[mkey,'mae']),ic(groups.loc[mkey,'rank_ic'])])
table(['Group','Hybrid MAE','Hybrid IC','MLP MAE','MLP IC'],rows,[260,*([(width-260)/4]*4)])
add('Seven features means all eight except F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value.','small')
add('These group tests used tuning and three final MLP seeds. They are a different experiment from the fixed-setting, single-seed individual tests above.','small')
page('Feature ranking by MAE contribution','48-hour prediction | Rankings use recorded individual-addition results, not new training.')
add('Hybrid ranking: original model settings','h')
add('Each feature below was added separately to the same three-feature core (MAE 181.2481). Negative change means lower error; positive change means higher error.','small')
core_mae=float(hy.loc[hy['pack']=='core','untuned_mae'].iloc[0])
ranked=hy[hy['pack']!='core'].sort_values('untuned_mae')
rows=[]
for rank,row in enumerate(ranked.to_dict('records'),1):
 rows.append([str(rank),row['added_feature'],f"{row['untuned_mae']:.4f}",f"{row['untuned_mae']-core_mae:+.4f}"])
table(['Rank','Original feature name','Hybrid MAE','MAE change'],rows,[38,510,100,width-648])
add('The three core features','h')
for name in features[:3]:add(escape(name),'code')
story.append(Spacer(1,7))
add('Adding PRICE__vol_ratio_4h_24h to the original two-feature baseline reduced hybrid MAE from <b>181.5260 to 181.2481</b> (a reduction of 0.2779). The two liquidation features were used together; these tests do not establish their individual ranking. The ratio was tested against a different starting group, so it is not ranked alongside the five additions above.','small')
add('MLP ranking differs','h')
add('Basis level (F4Z__basis__basis_level_rz__1h__value) ranked first among the five individual MLP additions, reducing its matched core MAE by <b>0.1874</b>. There is no single measured ranking that applies equally to both models. These rankings describe the historical results shown in this report.','small')

page('What we found','Recorded results from the completed model comparisons.')
add('<b>Lowest individual-addition hybrid MAE:</b> core + F4Z__liquidation__venue_breadth_percentile__1h__value: <b>181.055</b>, with IC <b>0.09816</b>. The core alone had higher IC <b>0.10680</b>.','small')
add('<b>Other hybrid MAE improvements:</b> core + PRICE__realized_vol_168h: <b>181.083</b>; core + F4Z__basis__basis_level_rz__1h__value: <b>181.153</b>. These are separate additions; using all three together was not validated by these tests.','small')
add('<b>Third feature that improved the original baseline:</b> PRICE__vol_ratio_4h_24h. Adding it to the original pair changed hybrid MAE from <b>181.526 to 181.248</b> and IC from <b>0.07739 to 0.10680</b>.','small')
add('<b>MLP finding:</b> adding F4Z__basis__basis_level_rz__1h__value to the three-feature MLP changed MAE from <b>181.735 to 181.548</b> and IC from <b>0.06093 to 0.09019</b>. It improved that MLP, but did not beat the original hybrid baseline.','small')
add('<b>Tuning finding:</b> tuning slightly improved core hybrid MAE to <b>181.203</b>, with IC <b>0.08513</b>. For the venue-breadth, realized-volatility and basis additions, original hybrid settings gave lower evaluation MAE than the tuning-selected settings.','small')
add('Recorded sources','h')
for text in ['Code/experiments/individual_hybrid_additions_v1/individual_results.csv','Code/experiments/individual_mlp_additions_v1/individual_contributions.csv','Code/experiments/two_group_tuning_v1/scores.csv','Original definitions: HANDOFF_2_CANONICAL_XY_V1_1/FEATURE_DICTIONARY.json','Price-feature definitions: Code/canonical_return_features.py']:
 add(escape(text),'small')
def footer(c,d):
 c.setFillColor(navy);c.setFont('Helvetica',8);c.drawString(40,22,'ScaleWave | 48-hour magnitude prediction | 5 November 2026');c.drawRightString(W-40,22,str(d.page))
SimpleDocTemplate(str(OUT),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=30,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT);assert len(reader.pages)==7,len(reader.pages)
text=' '.join(pg.extract_text() for pg in reader.pages);compact=''.join(text.split())
assert 'Weekly-block resampling' not in text and 'statistical residual-group analysis' not in text
for name in features:assert name in compact,name
for row in hy.to_dict('records'):
 for key in ['untuned_mae','selected_mae']:assert fmt(row[key]) in text
 for key in ['untuned_ic','selected_ic']:assert ic(row[key]) in text
for row in ml.to_dict('records'):assert fmt(row['mae']) in text and ic(row['ic']) in text
(TMP/'extracted_text.txt').write_text(text,encoding='utf-8')
print('PDF SAVED;7pages;names and result values verified:',OUT,flush=True)
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
doc=pdfium.PdfDocument(str(OUT))
for i in range(len(doc)):
 pg=doc[i];pg.render(scale=1.3).to_pil().save(TMP/f'page_{i+1}.png');pg.close()
doc.close();print('RENDERED',TMP,flush=True)
