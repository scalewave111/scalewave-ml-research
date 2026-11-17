from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from pypdf import PdfReader
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'Code/experiments';OUT=ROOT/'Report/ScaleWave_Magnitude_Forecasting_48h_4h_Revised.pdf'
a=pd.read_csv(E/'two_group_tuning_v1/scores.csv').set_index('model');b=pd.read_csv(E/'five_feature_4h_v1/scores.csv').set_index('model');c=pd.read_csv(E/'seven_eight_feature_4h_v1/scores.csv').set_index('model');d=pd.read_csv(E/'individual_hybrid_additions_v1/individual_results.csv')
s=getSampleStyleSheet();s.add(ParagraphStyle(name='T',parent=s['Title'],fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#17384C'),spaceAfter=6));s.add(ParagraphStyle(name='Sub',parent=s['Normal'],fontSize=9,textColor=colors.HexColor('#567080'),spaceAfter=10));s.add(ParagraphStyle(name='B',parent=s['BodyText'],fontSize=9.5,leading=13,spaceAfter=6));s.add(ParagraphStyle(name='H',parent=s['Heading2'],fontSize=13,leading=16,textColor=colors.HexColor('#087E8B'),spaceBefore=6,spaceAfter=4));s.add(ParagraphStyle(name='Tiny',parent=s['BodyText'],fontSize=8,leading=10,spaceAfter=5))
story=[]
def p(x,sty='B'):story.append(Paragraph(x,s[sty]))
def tab(headers,rows,widths):
 data=[[Paragraph('<font color="white">'+str(x)+'</font>',s['Tiny']) for x in headers]]+[[Paragraph(str(x),s['Tiny']) for x in row] for row in rows];t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17384C')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F6'),colors.white]),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CDD9DE')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([t,Spacer(1,6)])
def foot(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#567080'));c.drawString(16*mm,10*mm,'ScaleWave | magnitude forecasting');c.drawRightString(194*mm,10*mm,str(d.page))

p('BTC forecasts: from research to decisions','T')
p('A practical plan using the completed 4-hour and 48-hour magnitude experiments','Sub')
p('The answer a decision-maker needs','H')
p('At a specific time, for a specific horizon: <b>How likely is BTC to rise or fall? How large might the endpoint move be? What is the chance of an unusually large move? In which market conditions is the forecast reliable?</b> Each answer needs its own measured evidence.')
tab(['Forecast question','What should be reported','Status in these experiments'],[
['Direction','P(up) and P(down) over 4h or 48h.','Not established by the magnitude experiments.'],
['Move size','A prediction of absolute endpoint movement, with a horizon and units.','Historical magnitude predictions exist for both horizons.'],
['Extreme move','Probability of exceeding a defined absolute-return threshold.','Not established by the reviewed magnitude results.'],
['Context and reliability','Current market state; performance in similar conditions and across time.','Feature comparisons exist. Aggregate scores do not establish reliable probabilities in each market state.']],[43*mm,68*mm,64*mm])
p('What our existing models contribute','H')
p('The hybrid combines median regression with Random Forest. The MLP adds a neural correction to a fitted median-regression model. Both were used to estimate <b>absolute BTC start-to-end movement</b>. This answers the size part of the forecast. A large magnitude prediction can describe either an upward or a downward endpoint move.')
p('The 4-hour model concerns the next four hours; the 48-hour model concerns the next two days. Their outputs can be displayed side by side. Combining them into a trading rule has not been tested. Neither predicts the maximum excursion within the horizon or the path taken by price.')
p('What the current results do not tell us','H')
p('An error score or ranking score is not a probability of winning a trade. A magnitude estimate is not a probability that a threshold will be crossed. Direction, extreme-event probability, trading costs and the timing of exits need separate evaluation. No current-market forecast or live trading probability is generated in this report.')
p('Current evidence status','H')
p('Treat these models as exploratory historical candidates. The evaluation period has already been examined. The saved results support further forecast development; they do not establish confirmed live performance.','B')
story.append(PageBreak())
p('What the completed experiments tell us','T')
p('Selected hybrid and MLP results relevant to building a magnitude forecast','Sub')
p('48-hour magnitude: a small historical improvement','H')
p('Among the candidates shown here, the original-setting hybrid with <b>liquidation activity 5m and 15m, volatility ratio 4h/24h, and venue breadth 1h</b> had the lowest average size error. It improved on both the original reference and the three-feature hybrid that used the pair plus volatility ratio. This provides a candidate for the size field of a 48-hour forecast; it does not supply P(up) or P(down).')
p('Adding more features did not consistently improve the 48-hour results. The eight-feature MLP slightly improved average error over the original reference, while the seven- and eight-feature hybrids did not. The reported advantage is small and remains historical.')
p('4-hour magnitude: two useful candidate comparisons','H')
p('The <b>eight-feature hybrid</b> had the lowest average error among the recent fixed 4/5/7/8-feature combinations. The <b>five-feature MLP</b> had the strongest ranking of larger versus smaller moves. Both improved over the earlier RF reference. These are two candidates to evaluate further, rather than evidence of a known trading win rate.')
p('The five-feature MLP uses liquidation activity 5m and 15m, volatility ratio 4h/24h, venue breadth 1h, and realized volatility 168h. The eight-feature hybrid adds liquidation activity 1h, funding persistence 5m, and basis level 1h. In earlier individual-addition tests, ratio, breadth and 168h volatility improved both models. These findings depend on the tested feature combinations.')
p('Compact supporting evidence','H')
z=d.set_index('pack').loc['add_3']
tab(['Horizon and model','Average error (log bps)','Ranking score'],[
['48h: original reference',f"{a.loc['baseline','mae']:.4f}",f"{a.loc['baseline','rank_ic']:.5f}"],
['48h: pair + ratio hybrid',f"{a.loc['yesterday_best','mae']:.4f}",f"{a.loc['yesterday_best','rank_ic']:.5f}"],
['48h: pair + ratio + breadth hybrid',f'{z.untuned_mae:.4f}',f'{z.untuned_ic:.5f}'],
['4h: earlier RF reference',f"{c.loc['rf_baseline','mae']:.4f}",f"{c.loc['rf_baseline','rank_ic']:.5f}"],
['4h: eight-feature hybrid',f"{c.loc['hybrid_eight','mae']:.4f}",f"{c.loc['hybrid_eight','rank_ic']:.5f}"],
['4h: five-feature MLP',f"{b.loc['mlp_five','mae']:.4f}",f"{b.loc['mlp_five','rank_ic']:.5f}"]],[95*mm,43*mm,37*mm])
p('Average error is MAE (lower is better); ranking score is Spearman IC (higher is better). These are not forecast values or percentages of correct trades. Each listed result uses 18,481 historical observations. Compare models within the same horizon. The reference models use features; they are not constant, no-feature benchmarks.','Tiny')
p('What is not yet established','H')
p('These aggregate results do not identify a market condition in which a trade is reliably profitable. Time stability, conditional performance and uncertainty should be assessed from the saved predictions before making such a claim.','B')
story.append(PageBreak())
p('The forecast format and next steps','T')
p('Proposed work: a plan, not a claim that the missing forecasts are already available','Sub')
p('1. Complete and check the magnitude forecast','H')
p('Freeze a candidate and its preprocessing separately for 4h and 48h. Compare it with a training-median constant baseline and the best existing model on identical observations. Review performance by period and market condition. Then evaluate the frozen choices on fresh data. A range around the forecast needs its own coverage checks.')
p('2. Build direction and extreme-event probabilities','H')
p('Direction needs a separate target for upward versus downward endpoint returns. Extreme-move probability needs a defined horizon and threshold. Evaluate each probability model against its simple historical-rate baseline. Check calibration: among forecasts near 60%, the event should occur about 60% of the time in later data. This is a check to perform, not a result already obtained.')
p('3. Publish one record per forecast time','H')
tab(['Field','Proposed content'],[
['Time and horizon','Forecast timestamp, data cutoff, BTC, and 4h or 48h.'],
['Direction','P(up), P(down): unavailable until a probability model is evaluated.'],
['Magnitude','Predicted absolute endpoint move in log bps, from the selected model and current inputs. No live number is calculated here.'],
['Extreme move','P(absolute endpoint return > threshold), with threshold and units stated; unavailable until evaluated.'],
['Market context','Current feature values or defined regime, with evidence for performance in that state.'],
['Evidence status','Exploratory / development candidate / frozen for fresh confirmation / confirmed. Include model version and evaluation period.']],[42*mm,133*mm])
p('4. Evaluate a separate trading decision process','H')
p('Use the forecast together with fees, spread, slippage and risk limits to test long, short or no-trade rules, position size and execution. A high chance of being right does not by itself establish profitability: payoff sizes and costs also matter. Endpoint forecasts do not establish whether a stop or target is reached first. Test those rules on later data before describing them as useful trading rules.')
p('What this work adds','H')
p('The completed experiments provide candidate estimates of move size at two horizons. The next deliverable is a timestamped forecast with clear evidence and explicitly unavailable probability fields. The full trading forecast requires separately evaluated direction, extreme-event and decision components.')
p('Evidence files: Code/experiments/two_group_tuning_v1/scores.csv; individual_hybrid_additions_v1/individual_results.csv; five_feature_4h_v1/scores.csv; seven_eight_feature_4h_v1/scores.csv; individual_4h_additions_v1/individual_results.csv (all under Code/experiments).','Tiny')
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=13*mm,bottomMargin=16*mm,title='BTC Forecasts: Research and Next Steps').build(story,onFirstPage=foot,onLaterPages=foot)
rdr=PdfReader(OUT);assert len(rdr.pages)==3,len(rdr.pages)
txt=' '.join(x.extract_text() for x in rdr.pages)
assert '181.0547' in txt and '50.5001' in txt
assert 'photo' not in txt.lower() and 'LightGBM' not in txt
print(OUT)
import sys
sys.path.insert(0,str(ROOT/'Code/tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
tmp=ROOT/'tmp/pdfs/magnitude_revised';tmp.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(OUT))
for i in range(len(pdf)):pdf[i].render(scale=1.3).to_pil().save(tmp/f'page_{i+1}.png')
