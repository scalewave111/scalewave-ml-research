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
p('Raw MLP and raw hybrid','Title2')
p('BTC 48-hour UP/DOWN magnitude | Beginner report | 10 November 2026','Small2')
p('<b>Raw</b> means the original model forecast before extra mean scaling. Inputs are still cleaned and scaled. Each model predicts an UP size and a DOWN size; both are positive ordinary bps. <b>100 bps = 1%.</b>')
p('Techniques used','Head2')
table(['Technique','How it works'],[
['Raw MLP','A small neural network adds a learned correction to a fixed median-regression prediction. Two hidden layers (8 and 4 units), SiLU activation, 5% dropout, MAE loss; average forecasts from three random seeds. Earlier validation chooses the training duration.'],
['Raw hybrid','Combines median regression and an absolute-error random forest. Earlier validation chooses the RF weight from 0%, 10%, 25% or 50%; the rest is median regression.'],
['Chronological evaluation','Four expanding folds starting January, April, July and October 2024. Separate UP/DOWN training rows; strict 48-hour label-end purges. In this run, the first two validation months choose settings; the last month fits scaling, which is excluded from these raw forecasts.']],[102,409])
p('Results against simple baselines','Head2')
p('MAE = average absolute size error (lower is better). Rank IC = ability to rank smaller/larger actual moves (higher positive values are better). Avg. forecast and Avg. actual are mean sizes, not probabilities. All size/error columns below are bps.','Small2')
rows=[]
for side in ['UP','DOWN']:
 for model in ['Training mean baseline','Training median baseline','MLP raw','Hybrid raw']:
  r=next(r for r in old if r['side']==side and r['model']==model)
  label=model.replace('Training mean baseline','Mean baseline').replace('Training median baseline','Median baseline')
  rows.append([side,label,f(r,'mae_ordinary_bps'),f'{float(r["rank_ic"]):.3f}',f(r,'mean_predicted_magnitude_bps'),f(r,'mean_actual_magnitude_bps')])
table(['Side','Model','MAE','Rank IC','Avg. forecast','Avg. actual'],rows,[40,129,70,70,101,101])
p('Mean baseline = training-period average size. Median baseline = training-period middle size. Both are recalculated for each side and fold. Evaluation: 10,191 UP rows and 8,290 DOWN rows.','Small2')
p('How much did MAE improve?','Head2')
rows=[]
for side in ['UP','DOWN']:
 for model in ['MLP raw','Hybrid raw']:
  r=next(r for r in old if r['side']==side and r['model']==model)
  values=[]
  for base in ['Training mean baseline','Training median baseline']:
   b=float(next(r for r in old if r['side']==side and r['model']==base)['mae_ordinary_bps']);gain=b-float(r['mae_ordinary_bps'])
   values.append(f'{gain:.2f} bps ({100*gain/b:.2f}%)')
  rows.append([side,model,*values])
table(['Side','Model','Reduction vs mean','Reduction vs median'],rows,[40,129,171,171])
p('<b>Conclusion:</b> all four raw models beat both baselines on pooled MAE. UP MLP and DOWN hybrid have the lowest errors for their sides. Gains are modest and ranking is weak. Predictions average below actual sizes; MAE training targets median-like sizes, not verified conditional means.','Small2')
p('Results are historical and exploratory. No direction probability, combined expected-return accuracy or trading profit has been established. Sources: Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/{scores.csv,CONTRACT.json}; Code/fit_conditional_48h_magnitude_calibrated.py.','Small2')
story.append(PageBreak())
p('Features used by the models','Title2')
p('A feature is an input the model uses to make a forecast. All four raw models use all eight inputs listed below.')
import json
contract=json.loads((OLD.parent/'CONTRACT.json').read_text())
cols=contract['features']
ordered=[cols[i] for i in [1,0,3,5,4,2,6,7]]
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
p('Source: Week 1/Code/experiments/conditional_48h_magnitude_calibration_diagnostics_v1/CONTRACT.json. The historical-active-rate features measure frequency, not liquidation amounts. All input preprocessing is fitted using training data.','Small2')

def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536775'));c.drawString(42,23,'ScaleWave | Conditional magnitude research');c.drawRightString(553,23,str(d.page))
file=OUT/'BTC_48h_Raw_MLP_Hybrid_Report.pdf'
SimpleDocTemplate(str(file),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
pdf=PdfReader(file);assert len(pdf.pages)==2,len(pdf.pages)
text='\n'.join(pg.extract_text() for pg in pdf.pages)
assert '193.07' in text and '167.92' in text
for column in cols: assert column in ''.join(text.split())
sys.path.insert(0,str(W1/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
render=pdfium.PdfDocument(str(file))
for i in range(len(render)):render[i].render(scale=1.3).to_pil().save(OUT/f'page_{i+1}.png')
print(file)
