from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from pypdf import PdfReader
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'Code/experiments';OUT=ROOT/'Report/ScaleWave_Direction_and_Magnitude_Simple_Forecast_Report.pdf'
a=pd.read_csv(E/'two_group_tuning_v1/scores.csv').set_index('model');b=pd.read_csv(E/'five_feature_4h_v1/scores.csv').set_index('model');c=pd.read_csv(E/'seven_eight_feature_4h_v1/scores.csv').set_index('model');d=pd.read_csv(E/'individual_hybrid_additions_v1/individual_results.csv')
s=getSampleStyleSheet();s.add(ParagraphStyle(name='T',parent=s['Title'],fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#17384C'),spaceAfter=6));s.add(ParagraphStyle(name='Sub',parent=s['Normal'],fontSize=9,textColor=colors.HexColor('#567080'),spaceAfter=10));s.add(ParagraphStyle(name='B',parent=s['BodyText'],fontSize=9.5,leading=13,spaceAfter=6));s.add(ParagraphStyle(name='H',parent=s['Heading2'],fontSize=13,leading=16,textColor=colors.HexColor('#087E8B'),spaceBefore=6,spaceAfter=4));s.add(ParagraphStyle(name='Tiny',parent=s['BodyText'],fontSize=8,leading=10,spaceAfter=5))
story=[]
def p(x,sty='B'):story.append(Paragraph(x,s[sty]))
def tab(headers,rows,widths):
 data=[[Paragraph('<font color="white">'+str(x)+'</font>',s['Tiny']) for x in headers]]+[[Paragraph(str(x),s['Tiny']) for x in row] for row in rows];t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17384C')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F6'),colors.white]),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CDD9DE')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([t,Spacer(1,6)])
def foot(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#567080'));c.drawString(16*mm,10*mm,'ScaleWave | Direction and Magnitude Forecasting');c.drawRightString(194*mm,10*mm,str(d.page))





p('How can our models predict BTC move size and direction?','T')
p('Combining direction probabilities with magnitude estimates to forecast a signed return','Sub')
p('What we already have','H')
tab(['Model work','What it predicts','Horizon'],[
['Direction model','Probability of the endpoint finishing UP or DOWN. Reported model: v45.','48 hours'],
['Magnitude models','Size of the absolute start-to-end endpoint return.','4 hours and 48 hours']],[42*mm,95*mm,38*mm])
p('To forecast an actual signed move, we need both the likely direction and the move size for each direction. “Actual move” here means a forecast of signed endpoint return; the realized move is only known after the horizon ends.','B')
p('The calculation needed','H')
p('For one timestamp and one horizon, estimate average signed return separately for outcomes that finish UP and DOWN. Then weight each by its direction probability:','B')
tab(['Step','Calculation'],[
['1. Direction chances','P(UP) and P(DOWN), where P(DOWN) = 1 - P(UP).'],
['2. Conditional move sizes','Average UP return in ordinary bps; average DOWN return in ordinary bps (negative).'],
['3. Expected signed return','P(UP) x average UP bps + P(DOWN) x average DOWN bps.']],[48*mm,127*mm])
p('This gives one probability-weighted signed endpoint-return forecast in ordinary bps. Positive means the forecast favors an upward return; negative means it favors a downward return. It is an average forecast, not a guarantee of the realized move.')
p('What the current reports provide','H')
p('The direction report gives a historical 48-hour v45 example: <b>P(UP) 53.86%, P(DOWN) 46.14%</b> at 22 January 2025, 00:00 UTC. It labels v45 an exploratory candidate; fresh probability calibration and feature timing are unverified.')
p('The magnitude experiments provide candidate models for absolute move size. At 48 hours, a hybrid with the original liquidation pair, volatility ratio and venue breadth had MAE 181.05 log bps. At 4 hours, the eight-feature hybrid had the lowest MAE among recent fixed combinations (50.50 log bps). Those are historical average errors, not predicted move sizes for the direction example timestamp.')
p('The missing piece','H')
p('The current magnitude model predicts overall absolute size. It does not estimate separate typical UP and DOWN returns. Also, the reports do not provide a matched magnitude prediction for the historical 48-hour direction example. Therefore we cannot calculate a signed-bps forecast from the existing reported numbers alone.')
p('Do not multiply P(UP) by overall absolute magnitude and call it expected return. That assumes UP and DOWN move sizes are the same and that the overall magnitude is the right conditional size. The experiments have not established those assumptions.','Tiny')
story.append(PageBreak())
p('Roadmap to a signed bps forecast','T')
p('Use the existing work as the starting point; train and validate the missing conditional estimates','Sub')
p('1. Match the forecast rows','H');p('For each forecast timestamp, match the direction and magnitude inputs, target and predictions for the same asset and horizon. Start with 48 hours, where both direction and magnitude work exist. Treat 4-hour magnitude as a separate estimate until a 4-hour direction model is available.')
p('2. Estimate size by direction','H');p('Using training data only, estimate expected endpoint return when BTC finishes UP and expected endpoint return when it finishes DOWN. These are conditional signed returns. Convert each to ordinary bps per observation, then estimate the UP and DOWN averages. Alternatively train a model to predict signed return directly, while retaining direction probability as a separate output.')
p('3. Form the signed forecast','H');p('<b>Expected bps = P(UP) x expected UP bps + P(DOWN) x expected DOWN bps.</b> Keep the UP and DOWN contributions visible so the forecast shows where positive and negative expected return comes from.')
p('4. Test on later data','H');p('Freeze the model choices, then compare signed-bps forecasts with realized endpoint returns on fresh chronological data. Compare with a zero-return forecast and the current best. Check average error, bias, probability calibration, and performance across periods. The existing history has been repeatedly examined, so it cannot provide fresh confirmation.')
p('5. Evaluate trading separately','H');p('Apply entry, exit and position-size rules to the signed forecasts. Include fees, spread, slippage and risk limits, and test whether stops or targets are reached in the actual price path. A good endpoint-return forecast does not by itself prove a profitable trading rule.')
p('Example of the required inputs (not a result)','H');tab(['Input','Example placeholder'],[['P(UP) / P(DOWN)','53.86% / 46.14% (historical v45 example)'],['Expected UP return','+X ordinary bps (must be estimated)'],['Expected DOWN return','-Y ordinary bps (must be estimated)'],['Expected signed return','0.5386 x X + 0.4614 x (-Y) ordinary bps']],[68*mm,107*mm])
p('X and Y are intentionally left blank: the magnitude report does not provide direction-conditional sizes for the same timestamp. 53.86% / 46.14% is also historical, not live. Once the missing sizes are measured and validated, this calculation turns the two model outputs into a signed ordinary-bps forecast.','Tiny')
p('Current status: direction probabilities and absolute-magnitude models have been studied separately. A matched signed-bps forecast has not yet been calculated or validated.','B')
p('Sources: BTC_48H_Direction_Research_Report_Repaired.pdf; saved 48-hour and 4-hour magnitude experiment results. Direction details are taken from the supplied report; no new training was performed.','Tiny')
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=13*mm,bottomMargin=16*mm,title='What can we forecast for BTC?').build(story,onFirstPage=foot,onLaterPages=foot)
rdr=PdfReader(OUT);assert len(rdr.pages)==2,len(rdr.pages)
import sys
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
tmp=ROOT/'tmp/pdfs/direction_magnitude_simple';tmp.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(OUT))
for i in range(len(pdf)):pdf[i].render(scale=1.5).to_pil().save(tmp/f'page_{i+1}.png')
print(OUT)
