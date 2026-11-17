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
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'Code/experiments';OUT=ROOT/'Report/ScaleWave_48h_Features_and_Model_Results_2026-11-05.pdf';TMP=ROOT/'tmp/pdfs/feature_results_report';TMP.mkdir(parents=True,exist_ok=True)
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
add('What we did','h')
add('1. Started with a two-feature baseline, then a three-feature core.<br/>2. Screened the supplied 40 entries: 38 unique features, of which 32 were usable across training windows.<br/>3. Started a broader eight-feature combination search; it was stopped before completion.<br/>4. Used statistical diagnostics, then tested seven- and eight-feature groups with both models.<br/>5. Tested each of the five extra features separately, and checked hybrid tuning against original settings.')
add('Training used earlier data and evaluation used later data in four expanding windows. A strict 48-hour label boundary prevented training outcomes from crossing the next cutoff. Validation data chose settings; evaluation data measured the resulting performance.','small')
page('The eight original feature names','Exact dataset/model column names. The three-feature core consists of the first three rows.')
statuses=['Core feature; retained in the original baseline.','Core feature; retained in the original baseline.','Added to the original pair to form the three-feature core.','Individual hybrid addition: lower MAE, lower IC than core.','Individual hybrid addition: lower MAE, lower IC than core.','Individual addition: lower MAE and higher IC in the MLP; lower MAE and lower IC in hybrid.','Individual hybrid addition worsened MAE. Value was 1.0 for every evaluation row.','Individual MLP addition slightly improved MAE/IC; hybrid MAE worsened.']
table(['Original column name','Recorded finding'],list(zip(features,statuses)),[480,width-480],True)
add('<b>Baseline:</b> first two rows. <b>Core:</b> first three rows. <b>Seven-feature group:</b> all eight except F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value.','small')
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
add('These group tests used tuning and three final MLP seeds. They are a different experiment from the fixed-setting, single-seed individual tests above.','small')
page('Findings and checks','Measured conclusions only; the report does not assign untested causes to model performance.')
add('<b>Lowest individual-addition hybrid MAE:</b> core + F4Z__liquidation__venue_breadth_percentile__1h__value: <b>181.055</b>, with IC <b>0.09816</b>. The core alone had higher IC <b>0.10680</b>.','small')
add('<b>Other hybrid MAE improvements:</b> core + PRICE__realized_vol_168h: <b>181.083</b>; core + F4Z__basis__basis_level_rz__1h__value: <b>181.153</b>. These are separate additions; using all three together was not validated by these tests.','small')
add('<b>Earlier broad 38-feature search:</b> selected hybrid MAE 184.084 / IC 0.03235; selected MLP MAE 184.146 / IC 0.00877. Neither beat the original baseline. The later exhaustive eight-feature search was stopped and is not reported as completed.','small')
add('How we checked the results','h')
add('Saved predictions were checked against their evaluation rows and replayed from saved models. Independent checks were completed for the seven/eight-feature experiment and individual hybrid tests. Individual MLP tests included checkpoint replay and source checks.','small')
add('Weekly-block resampling was used to estimate uncertainty because nearby observations share overlapping 48-hour outcomes. The reported individual-addition MAE improvement intervals include zero. This means these data do not establish a reliable improvement. The evaluation history has been examined repeatedly; these are historical research results, not fresh confirmation.','small')
add('Other diagnostic checks','h')
add('The statistical residual-group analysis used a different assessment period and was not an RF/MLP comparison. Feature shuffling measured reliance of an existing fitted model; it did not measure the result of adding or removing a feature and retraining. Those diagnostics are kept separate from the model tables.','small')
add('Recorded sources','h')
for text in ['Code/experiments/individual_hybrid_additions_v1/individual_results.csv','Code/experiments/individual_mlp_additions_v1/individual_contributions.csv','Code/experiments/two_group_tuning_v1/scores.csv','Code/experiments/report38_subset_search_v2/scores.csv','Uncertainty and audit files are saved beside the corresponding result tables.']:
 add(escape(text),'small')
def footer(c,d):
 c.setFillColor(navy);c.setFont('Helvetica',8);c.drawString(40,22,'ScaleWave | 48-hour magnitude prediction | 5 November 2026');c.drawRightString(W-40,22,str(d.page))
SimpleDocTemplate(str(OUT),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=30,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT);assert len(reader.pages)==5,len(reader.pages)
text=' '.join(pg.extract_text() for pg in reader.pages);compact=''.join(text.split())
for name in features:assert name in compact,name
for row in hy.to_dict('records'):
 for key in ['untuned_mae','selected_mae']:assert fmt(row[key]) in text
 for key in ['untuned_ic','selected_ic']:assert ic(row[key]) in text
for row in ml.to_dict('records'):assert fmt(row['mae']) in text and ic(row['ic']) in text
(TMP/'extracted_text.txt').write_text(text,encoding='utf-8')
print('PDF SAVED;5pages;names and result values verified:',OUT,flush=True)
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
doc=pdfium.PdfDocument(str(OUT))
for i in range(len(doc)):
 pg=doc[i];pg.render(scale=1.3).to_pil().save(TMP/f'page_{i+1}.png');pg.close()
doc.close();print('RENDERED',TMP,flush=True)
