from pathlib import Path
import csv,sys
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
W1=Path(r'C:/Users/Siddh/Desktop/ML/November/Week 1')
W2=W1.parent/'Week 2'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else W2/'Report'
OUT.mkdir(parents=True,exist_ok=True)
OLD=W1/'Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/scores.csv'
NEW=W2/'Code/experiments/conditional_48h_mean_feature_comparison_v1/scores.csv'
old=list(csv.DictReader(OLD.open()));new=list(csv.DictReader(NEW.open()))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#17334b'),spaceAfter=12))
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=6))
styles.add(ParagraphStyle(name='Head2',fontName='Helvetica-Bold',fontSize=12,leading=15,textColor=colors.HexColor('#17334b'),spaceBefore=8,spaceAfter=7))
story=[]
def p(t,style='Body2'):story.append(Paragraph(t,styles[style]))
def table(headers,rows,widths):
 data=[[Paragraph(str(v),styles['Small2']) for v in headers]]+[[Paragraph(str(v),styles['Small2']) for v in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9db0bf')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')])]))
 story.append(t);story.append(Spacer(1,8))
def f(r,k):return f'{float(r[k]):.2f}'
p('BTC 48-hour move-size models','Title2')
p('November Week 2 | Results explained simply | 10 November 2026','Small2')
p('<b>Main finding:</b> Earlier models slightly reduced average absolute error. New mean-targeting models predict average sizes more closely, but show only a small UP improvement in squared error and no DOWN improvement.')
p('What are we predicting?','Head2')
p('For every timestamp, each model predicts two sizes: the rise if BTC ends UP after 48 hours, and the fall if BTC ends DOWN. Both sizes are positive. These are end-to-end moves, not the largest move during the 48 hours. <b>100 basis points (bps) = 1%.</b>')
p('We tested four chronological folds, training on older data and evaluating later data. A 48-hour label purge keeps training outcomes from overlapping the next forecast period. Results include 10,191 actual UP rows and 8,290 actual DOWN rows.')
p('How to read the numbers','Head2')
table(['Measure','Plain-language meaning'],[
['MAE','Average absolute miss in move size. Lower is better. MAE of 193 bps means a miss of 1.93 percentage points on average.'],
['RMSE / MSE','Measures error while giving large misses more weight. Lower is better. RMSE is in bps; MSE is in squared bps.'],
['Rank IC','Whether larger forecasts correspond to larger actual moves. Near zero is weak; higher positive values are better. It is not an accuracy percentage.'],
['Mean bias','Average prediction minus average actual size. Negative means average underprediction.'],
['Mean baseline','A simple forecast: use the average move size from training data for that side. It is recalculated in each fold.']],[100,411])
p('Earlier approach: predict a typical size','Head2')
p('The MLP (small neural network) and RF-plus-median-regression hybrid were trained to minimize MAE. This favors median-like, or typical, sizes. Shown below: the lowest-MAE raw model for each side in the calibration experiment.')
rows=[]
for side,name in [('UP','MLP raw'),('DOWN','Hybrid raw')]:
 r=next(r for r in old if r['side']==side and r['model']==name)
 b=next(r for r in old if r['side']==side and r['model']=='Training mean baseline')
 rows.append([side,name,f(r,'mae_ordinary_bps'),f(b,'mae_ordinary_bps'),f(r,'mean_predicted_magnitude_bps'),f(r,'mean_actual_magnitude_bps')])
table(['Side','Model','MAE','Baseline MAE','Avg. forecast','Avg. actual'],rows,[42,105,70,90,102,102])
p('UP error improved by about 1.2%; DOWN by 3.6%. Forecast averages were too low. Scaling them closer to actual averages increased pooled test MAE. Diagnostics did not establish consistent severe overfitting.','Small2')
story.append(PageBreak())
p('New approach: predict a mean size','Title2')
p('Squared-error training targets the conditional mean needed for an expected-return calculation. The new hybrid combines Ridge (regularized linear regression) with a squared-error random forest. The MLP also uses a Ridge linear component.')
p('<b>Feature sets:</b> 4 = liquidation activity 5m/15m + volatility ratio + venue breadth. 5 = those four + realized volatility 168h. 8 = those five + liquidation activity 1h + funding persistence + basis level. These were predefined candidates informed by earlier overall-magnitude work, not proven optimal conditional feature sets.','Small2')
p('All new fixed-feature results','Head2')
rows=[]
for r in new:
 if r['model'].startswith('baseline') or r['model'].endswith('selected'):continue
 label=r['model'].replace('mlp_','MLP / ').replace('hybrid_','Hybrid / ').replace('four','4').replace('five','5').replace('eight','8')
 rows.append([r['side'],label,f(r,'mae_bps'),f(r,'rmse_bps'),f'{float(r["rank_ic"]):.3f}',f(r,'mean_bias_bps')])
table(['Side','Model / features','MAE','RMSE','Rank IC','Mean bias'],rows,[42,139,80,80,80,90])
p('What happened when validation chose the features?','Head2')
p('For each side and model, earlier validation data chose among 4, 5 and 8 features using MSE. The following RMSE scores measure that selection strategy on later data.','Small2')
rows=[]
for name,label in [('baseline_mean','Training-mean baseline'),('mlp_selected','Validation-selected MLP'),('hybrid_selected','Validation-selected hybrid')]:
 rows.append([label]+[f(next(r for r in new if r['side']==side and r['model']==name),'rmse_bps') for side in ['UP','DOWN']])
table(['Strategy','UP RMSE','DOWN RMSE'],rows,[291,110,110])
p('What can we conclude?','Head2')
p('<b>1. UP:</b> the 4-feature hybrid had the lowest observed RMSE (257.40 versus baseline 259.52), about 1.6% lower MSE. The validation-selected hybrid also slightly improved MSE.<br/><b>2. DOWN:</b> every new model had worse RMSE than the mean baseline.<br/><b>3. Trade-off:</b> all new models had worse MAE than the mean baseline. Better average sizes did not mean better individual forecasts.<br/><b>4. Limits:</b> rank IC remains weak. Matching averages does not prove accurate means at each timestamp. These historical results are exploratory; no direction probabilities, combined signed-return performance or trading profit has been established.','Small2')
p('Next: inspect performance by fold and calibration by forecast group, then evaluate a frozen approach on unused later data. No new training was performed to create this report.','Small2')
p('Sources: Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/{scores.csv,fit_diagnostics.csv}; Week 2/Code/experiments/conditional_48h_mean_feature_comparison_v1/scores.csv.','Small2')
story.append(PageBreak())
p('Features used by the models','Title2')
p('A feature is an input the model uses to make a forecast. Both UP and DOWN models use the same feature sets. The earlier MAE-trained models used all eight inputs.')
p('<b>4-feature set:</b> A, B, C, D<br/><b>5-feature set:</b> A, B, C, D, E<br/><b>8-feature set:</b> A, B, C, D, E, F, G, H')
import json
contract=json.loads((W2/'Code/experiments/conditional_48h_mean_feature_comparison_v1/CONTRACT.json').read_text())
cols=contract['features']['eight']
ordered=[cols[i] for i in [1,0,3,5,4,2,6,7]]
assert set(ordered[:4])==set(contract['features']['four'])
assert set(ordered[:5])==set(contract['features']['five'])
feature_info=[
('A','Peer liquidation activity: 5 minutes','Historical frequency of liquidation activity in peer assets, using 5-minute liquidation windows. Measures activity frequency, not liquidation size.'),
('B','Peer liquidation activity: 15 minutes','The same historical activity-frequency measure, using 15-minute liquidation windows.'),
('C','Price volatility ratio: 4h / 24h','Compares recent 4-hour price volatility with 24-hour volatility, adjusted for window length. Shows whether recent movement is unusually strong or quiet.'),
('D','Liquidation venue breadth: 1 hour','How broadly liquidation activity is spread across eligible exchanges, ranked against its own history.'),
('E','Realized price volatility: 168 hours','Price variability over the past 7 days, calculated from squared 5-minute log returns.'),
('F','Peer liquidation activity: 1 hour','Historical frequency of liquidation activity in peer assets, using 1-hour liquidation windows.'),
('G','Funding persistence: 5 minutes','How much of the funding path stayed on the same side of zero as its ending value, ranked against history.'),
('H','Basis level: 1 hour','The mark-price versus index-price gap, expressed relative to its historical median using a robust scale.')]
for (letter,label,meaning),column in zip(feature_info,ordered):
 p(f'<b>{letter}. {label}</b>','Body2')
 p(column,'Small2')
 p(meaning,'Small2')
p('<b>Why these groups?</b> The smaller groups were informed by earlier overall-magnitude experiments, particularly the 4-hour work. They are comparison candidates, not proven best features for separate 48-hour UP/DOWN forecasting.','Small2')
p('Exact membership: Week 2/Code/experiments/conditional_48h_mean_feature_comparison_v1/CONTRACT.json. Meanings follow the previously documented feature definitions and price-feature implementation.','Small2')

def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536775'));c.drawString(42,23,'ScaleWave | Conditional magnitude research');c.drawRightString(553,23,str(d.page))
file=OUT/'BTC_48h_Magnitude_Results_Simple.pdf'
SimpleDocTemplate(str(file),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
pdf=PdfReader(file);assert len(pdf.pages)==3,len(pdf.pages)
text='\n'.join(pg.extract_text() for pg in pdf.pages)
assert '193.07' in text and '167.92' in text and '257.40' in text
sys.path.insert(0,str(W1/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
render=pdfium.PdfDocument(str(file))
for i in range(len(render)):render[i].render(scale=1.3).to_pil().save(OUT/f'page_{i+1}.png')
print(file)
