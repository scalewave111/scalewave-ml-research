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
OUT=ROOT/'Report/ScaleWave_4h_Todays_Work_2026-11-06_Ranked.pdf'
TMP=ROOT/'tmp/pdfs/4h_today_report_ranked';TMP.mkdir(parents=True,exist_ok=True)
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
 add('SCALEWAVE | 6 NOVEMBER 2026','small');add(title,'title');add(sub,'small')
def table(headers,rows,widths):
 data=[[p(escape(str(v)),'th') for v in headers]]+[[p(escape(str(v)),'cell') for v in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,8)])

four=pd.read_csv(E/'four_feature_4h_v1/scores.csv').set_index('model')
main=pd.read_csv(E/'seven_eight_feature_4h_v1/scores.csv').set_index('model')
five=pd.read_csv(E/'five_feature_4h_v1/scores.csv').set_index('model')
focus=pd.read_csv(E/'focused_four_feature_4h_v1/scores.csv')
used=[]
def result(label,inputs,row):
 used.append(row)
 return [label,inputs,f"{row['mae']:.4f}",f"{row['rank_ic']:.5f}"]
page('Today\'s 4-hour experiments','6 November 2026 | Completed work | 18,481 historical test observations per result')
add('We predicted the <b>size of Bitcoin\'s start-to-end move over the next 4 hours</b>, using absolute log return. The target measures movement size, not whether price rises or falls. Results combine four successive time windows. Earlier validation data selected settings; final MLP predictions averaged three random seeds.','small')
table(['Term','Simple meaning'],[['Hybrid','Median regression (a linear model) blended with Random Forest; earlier validation chooses the blend.'],['MLP','A small neural network that adds a nonlinear correction to a fitted median-regression model.'],['MAE','Average prediction error in log basis points. Lower is better.'],['IC','Spearman rank correlation: whether larger predicted moves tend to be larger actual moves. Higher is better; it is not an accuracy percentage.']],[75,width-75])
add('Model ranking: fixed feature combinations','head')
rows=[result('RF baseline (earlier reference)','A, B',main.loc['rf_baseline'])]
for n,df in [('four',main),('five',five),('seven',main),('eight',main)]:
 keys={'four':'A, B, D, F','five':'A, B, D, E, F','seven':'A, B, D, E, F, G, H','eight':'A, B, C, D, E, F, G, H'}[n]
 for model in ['hybrid','mlp']:rows.append(result(model.upper() if model=='mlp' else 'Hybrid',keys,df.loc[model+'_'+n]))
rows.sort(key=lambda x:float(x[2]))
ic_order=sorted(range(len(rows)),key=lambda i:float(rows[i][3]),reverse=True)
ic_ranks={i:k+1 for k,i in enumerate(ic_order)}
table(['MAE rank','Model','Features (page 2)','MAE','IC','IC rank'],[[i+1]+row+[ic_ranks[i]] for i,row in enumerate(rows)],[55,165,260,100,100,width-680])
add('<b>Recorded results:</b> Eight-feature hybrid had the lowest MAE in this table (50.5001). Five-feature MLP had the highest IC (0.21590). Five-feature MLP scored better than five-feature hybrid on both metrics.','small')
add('Today we tested four-feature tuning, a focused four-feature combination search, and fixed five-, seven- and eight-feature models. Saved prediction checks passed.','small')

page('Features used','Letters identify exact inputs in the results tables. A and B are the original two features.')
names=pair+[ 'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h','F4Z__liquidation__venue_breadth_percentile__1h__value','F4Z__funding__funding_persistence_percentile__5m__value','F4Z__basis__basis_level_rz__1h__value']
short=['Peer liquidation historical activity - 5m','Peer liquidation historical activity - 15m','Peer liquidation historical activity - 1h','Volatility ratio - 4h / 24h','Realized volatility - 168h (7 days)','Liquidation venue breadth percentile - 1h','Funding persistence percentile - 5m','Basis level, robust standardized - 1h']
for letter,label,name in zip('ABCDEFGH',short,names):
 add(letter+' | '+label,'head');add(escape(name),'code');story.append(Spacer(1,5))
add('Feature combinations tested','head')
add('<b>4 features:</b> A, B, D, F. <b>5 features:</b> A, B, D, E, F.<br/><b>7 features:</b> all except C. <b>8 features:</b> all A-H. The focused four-feature search kept A and B and screened pairs from C-H.','small')
add('The five-feature run added volatility ratio, venue breadth and seven-day realized volatility to the original pair. These were the top three additions in the earlier individual-feature study; that earlier ranking was used to choose this run.','small')

page('Tuning and combination search','Completed four-feature tests')
rr=[result('Hybrid: validation-gated tuning','A, B, D, F',four.loc['hybrid_four_selected']),result('MLP: validation-gated architecture','A, B, D, F',four.loc['mlp_four_selected'])]
for m in ['hybrid','mlp']:
 z=focus[(focus.model==m)&(focus.strategy=='selected')].iloc[0]
 rr.append(result(('Hybrid' if m=='hybrid' else 'MLP')+': focused combination search','Changes by window; see below',z))
table(['Test','Features','MAE','IC'],rr,[270,260,120,width-650])
add('Tuning changes required an earlier-validation improvement. The selected hybrid scored worse than its default result on page 1. No MLP architecture change passed the gate; its result stayed unchanged. Subsequent fixed 5/7/8-feature runs used the original hybrid settings and an MLP with 8 and 4 hidden units, dropout 0.05.','small')
add('The focused search screened 15 combinations per model/window, fully validated the top three plus A+B+D+F, then tested the validation winner. Selected inputs: <b>January:</b> A+B+E+F (both); <b>April:</b> A+B+D+F (both); <b>July:</b> A+B+D+E (both); <b>October:</b> A+B+F+H (hybrid), A+B+D+F (MLP). Both combined search results had higher MAE and lower IC than the fixed four-feature results on page 1.','small')
add('Sources: scores.csv in Code/experiments/{four_feature_4h_v1, focused_four_feature_4h_v1, seven_eight_feature_4h_v1, five_feature_4h_v1}; corresponding saved audits and selection files.','small')

page('Feature ranking and conclusions','Rankings and conclusions below come from saved experiment results.')
add('Feature ranking: earlier individual-addition tests','head')
add('Each feature was added separately to the original pair A+B, with settings selected using earlier validation. Rank 1 gave the largest MAE reduction for that model. Negative MAE change means improvement; positive means worse. These are the earlier tests used to choose today\'s five-feature model.','small')
hy=r[r.model=='hybrid'].set_index('pack');ml=r[r.model=='mlp'].set_index('pack')
hr=hy.mae_change_vs_pair.rank(method='min').astype(int);mr=ml.mae_change_vs_pair.rank(method='min').astype(int)
letters={'add_1':'C','add_2':'D','add_3':'E','add_4':'F','add_5':'G','add_6':'H'}
table(['Feature','Hybrid rank','Hybrid MAE change','MLP rank','MLP MAE change'],[[letters[k]+' | '+labels[k],hr[k],f'{hy.loc[k,"mae_change_vs_pair"]:+.4f}',mr[k],f'{ml.loc[k,"mae_change_vs_pair"]:+.4f}'] for k in hr.sort_values().index],[300,85,145,85,width-615])
add('A and B were always used together, so these tests do not establish their separate ranks. The table measures adding a feature with retuning; it is not an importance ranking inside the eight-feature model.','small')
add('What we conclude from the recorded results','head')
add('1. <b>Lowest MAE:</b> the eight-feature hybrid scored 50.5001, with IC 0.21433. It had the lowest MAE among the models and strategies in this report.<br/>2. <b>Highest IC:</b> the five-feature MLP scored 0.21590, with MAE 50.5048. It had the highest IC in this report. The lowest-MAE model and highest-IC model were different.<br/>3. <b>Feature additions:</b> volatility ratio, venue breadth and seven-day realized volatility improved both MAE and IC for both models in the earlier individual-addition tests. Today\'s five-feature models combined these three with A+B and improved both metrics over their four-feature versions.<br/>4. <b>Feature effects depend on the combination:</b> adding C to the seven-feature set lowered MAE for both models in the eight-feature run, although C was a weak individual addition.<br/>5. <b>Tuning and search:</b> the tuned hybrid and focused combination-search strategies did not improve the fixed four-feature test results. No MLP architecture change passed the validation gate.','small')
add('These conclusions describe the recorded historical tests. The model ranking on page 1 covers fixed feature combinations; other tested strategies are listed on page 3.','small')
add('Feature-ranking source: Code/experiments/individual_4h_additions_v1/individual_results.csv. Model results: the saved score files listed on page 3.','small')

def footer(c,d):
 c.setFillColor(navy);c.setFont('Helvetica',8);c.drawString(36,19,'ScaleWave | 4-hour magnitude prediction | Recorded results');c.drawRightString(W-36,19,str(d.page))
SimpleDocTemplate(str(OUT),pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=26,bottomMargin=33).build(story,onFirstPage=footer,onLaterPages=footer)
doc=PdfReader(OUT);assert len(doc.pages)==4,len(doc.pages)
text='\n'.join(pg.extract_text() for pg in doc.pages);compact=''.join(text.split())

for name in names:assert name in compact,name
for row in used:
 assert f"{row['mae']:.4f}" in text and f"{row['rank_ic']:.5f}" in text
assert 'lightgbm' not in text.lower()
assert 'activation' not in text.lower()
(TMP/'text.txt').write_text(text,encoding='utf-8')
print('Saved and checked',OUT)
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
pdf=pdfium.PdfDocument(str(OUT))
for i in range(len(pdf)):
 pg=pdf[i];pg.render(scale=1.4).to_pil().save(TMP/f'page_{i+1}.png');pg.close()
pdf.close()
print('Rendered',len(doc.pages),'pages')
