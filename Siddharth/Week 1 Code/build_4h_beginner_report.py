"""Short factual PDF from completed 4-hour comparison CSVs."""
from pathlib import Path
import sys
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Report/ScaleWave_4h_Simple_Results_and_Rankings_2026-11-05.pdf'
TMP=ROOT/'tmp/pdfs/4h_beginner_report';TMP.mkdir(parents=True,exist_ok=True)
E=ROOT/'Code/experiments';r=pd.read_csv(E/'individual_4h_additions_v1/individual_results.csv');base=pd.read_csv(E/'two_feature_4h_v1/scores.csv')
labels={'add_1':'Liquidation historical activity - 1h','add_2':'Volatility ratio - 4h/24h','add_3':'Realized volatility - 168h','add_4':'Liquidation venue breadth - 1h','add_5':'Funding persistence - 5m','add_6':'Basis level - 1h'}
pair=['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value']
W,H=landscape(A4);width=W-72;navy=colors.HexColor('#17384C');teal=colors.HexColor('#087E8B')
styles={k:ParagraphStyle(k,fontName=f,fontSize=z,leading=l,textColor=c,spaceAfter=sp) for k,f,z,l,c,sp in [('title','Helvetica-Bold',23,27,navy,10),('head','Helvetica-Bold',12,15,teal,7),('body','Helvetica',10,13,navy,8),('small','Helvetica',8.5,11,navy,6),('code','Courier',7.8,10,navy,3),('cell','Helvetica',9,11,navy,0),('th','Helvetica-Bold',9,11,colors.white,0)]}
story=[]
def p(v,k='body'):return Paragraph(v,styles[k])
def add(v,k='body'):story.append(p(v,k))
def page(title,sub):
 if story:story.append(PageBreak())
 add('SCALEWAVE | 5 NOVEMBER 2026','small');add(title,'title');add(sub,'small')
def table(headers,rows,widths):
 data=[[p(escape(str(v)),'th') for v in headers]]+[[p(escape(str(v)),'cell') for v in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,8)])
page('4-hour prediction: simple report','Completed hybrid and MLP experiments | Same 18,481 historical evaluation observations')
add('<b>Question:</b> Can our models predict the size of Bitcoin\'s move over the next 4 hours better than the original Random Forest baseline? We predict absolute start-to-end log return, not price direction or the largest move during those hours.')
table(['Term','Meaning'],[['MAE','Average size of the prediction error, in log basis points. Lower is better.'],['IC','Spearman rank correlation: do larger predicted moves tend to be larger in reality? Higher is better. It is not an accuracy percentage.'],['Hybrid','Median regression (linear quantile regression at 0.5) combined with Random Forest. Validation chooses their weights.'],['MLP','A small neural network that starts with median regression and learns an additional nonlinear correction.']],[80,width-80])
add('Starting features: the original pair','head')
for name in pair:add(escape(name),'code')
add('These measure historical liquidation-activity frequency in peer assets using 5-minute and 15-minute underlying windows. Pair + one extra feature means three inputs in total.','small')
add('What we did','head')
add('1. Built 4-hour targets by summing 48 consecutive 5-minute log returns, then taking the absolute value.<br/>2. Trained the hybrid and MLP using the original pair.<br/>3. Added each of the six remaining features separately to that pair.<br/>4. Tested two configurations per model. For the MLP, this included 8/4 versus 4 hidden units and dropout 0.05 versus 0.15. Earlier validation data selected settings; final MLP predictions averaged three random seeds.<br/>5. Evaluated four successive time windows and compared MAE and IC on the same observations. Training outcomes ended before each cutoff; saved predictions were checked.')
add('<b>Scope:</b> the rankings cover this two-feature and six-individual-addition study. We did not test combinations of the extra features. These are historical evaluation results.','small')

page('Model results: ranked by MAE','Each addition row uses the original pair plus exactly one extra feature. Rank 1 has the lowest MAE in this study.')
rankrows=[]
for row in base.to_dict('records'):
 rankrows.append(dict(model={'rf_baseline':'Original RF baseline','hybrid':'Hybrid','mlp':'MLP'}[row['model']],extra='None (original pair)',mae=row['mae'],ic=row['rank_ic']))
for row in r.to_dict('records'):rankrows.append(dict(model='Hybrid' if row['model']=='hybrid' else 'MLP',extra=labels[row['pack']],mae=row['mae'],ic=row['rank_ic']))
rankrows.sort(key=lambda v:v['mae'])
table(['Rank','Model','Extra feature','MAE (lower better)','IC (higher better)'],[[i+1,z['model'],z['extra'],f"{z['mae']:.4f}",f"{z['ic']:.5f}"] for i,z in enumerate(rankrows)],[38,125,365,120,width-648])
add('<b>Best MLP:</b> pair + venue breadth, MAE <b>50.9081</b>, IC <b>0.17635</b>. <b>Best hybrid:</b> pair + volatility ratio, MAE <b>51.1324</b>, IC <b>0.16656</b>.','small')
add('All 12 addition results beat the original RF baseline on MAE and IC. But beating RF does not always mean the extra feature helped: the two-feature hybrid and MLP already beat RF. Compare additions with their own model\'s two-feature control on the next page.','small')

page('Feature rankings: what actually helped','Ranked separately for each model by MAE change versus its own two-feature control. Negative change means improvement.')
hy=r[r.model=='hybrid'].set_index('pack');ml=r[r.model=='mlp'].set_index('pack')
hr=hy.mae_change_vs_pair.rank(method='min').astype(int);mr=ml.mae_change_vs_pair.rank(method='min').astype(int)
table(['Extra feature','Hybrid rank','Hybrid MAE change','MLP rank','MLP MAE change'],[[labels[k],hr[k],f'{hy.loc[k,"mae_change_vs_pair"]:+.4f}',mr[k],f'{ml.loc[k,"mae_change_vs_pair"]:+.4f}'] for k in hr.sort_values().index],[310,85,140,85,width-620])
add('Conclusion','head')
add('<b>Volatility ratio, venue breadth and 168h realized volatility improved both MAE and IC in both models.</b> The other three additions slightly lowered hybrid MAE but increased MLP MAE. The original two inputs were used together, so these tests do not give their individual ranks.','small')
add('Each addition was retuned using earlier validation data. The changes therefore measure adding a feature with retuning, not its effect at fixed settings.','small')
add('Exact original names of the six additions','head')
for pack in ['add_2','add_4','add_3','add_6','add_1','add_5']:
 add(escape(hy.loc[pack,'added_feature']),'code')
add('Sources: Code/experiments/two_feature_4h_v1/scores.csv and Code/experiments/individual_4h_additions_v1/individual_results.csv. Training runners: Code/two_feature_4h.py and Code/individual_4h_additions.py.','small')

def footer(c,d):
 c.setFillColor(navy);c.setFont('Helvetica',8);c.drawString(36,19,'ScaleWave | 4-hour magnitude prediction | Recorded results');c.drawRightString(W-36,19,str(d.page))
SimpleDocTemplate(str(OUT),pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=26,bottomMargin=33).build(story,onFirstPage=footer,onLaterPages=footer)
doc=PdfReader(OUT);assert len(doc.pages)==3,len(doc.pages)
text='\n'.join(pg.extract_text() for pg in doc.pages);compact=''.join(text.split())
for name in pair+list(r.added_feature.unique()):assert name in compact,name
for row in rankrows:assert f"{row['mae']:.4f}" in text and f"{row['ic']:.5f}" in text
(TMP/'text.txt').write_text(text,encoding='utf-8');print('Saved and checked',OUT,flush=True)
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
pdf=pdfium.PdfDocument(str(OUT))
for i in range(len(pdf)):
 pg=pdf[i];pg.render(scale=1.4).to_pil().save(TMP/f'page_{i+1}.png');pg.close()
pdf.close();print('Rendered all3 pages',flush=True)
