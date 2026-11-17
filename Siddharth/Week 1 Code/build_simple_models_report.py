from pathlib import Path
import json,sys
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
R=Path(__file__).resolve().parent
P=R/'output/pdf/ScaleWave_Simple_Model_Results_2026-11-04.pdf'
T=R/'tmp/pdfs/simple_models';T.mkdir(parents=True,exist_ok=True)
navy=colors.HexColor('#16324F');teal=colors.HexColor('#087E8B')
styles={k:ParagraphStyle(k,fontName=f,fontSize=s,leading=l,textColor=c,spaceAfter=a) for k,f,s,l,c,a in [('title','Helvetica-Bold',23,27,navy,12),('h','Helvetica-Bold',14,18,teal,10),('body','Helvetica',11,16,navy,12),('small','Helvetica',9,12,navy,9),('cell','Helvetica',9,12,navy,0),('head','Helvetica-Bold',9,12,colors.white,0)]}
def p(t,s='body'):return Paragraph(t,styles[s])
pair='5-minute + 15-minute liquidation active rates'
three=pair+' + 4-hour / 24-hour volatility ratio'
four=three+' + 1-hour liquidation active rate'
spec=[
('baseline_single_feature_v6','baseline','Original baseline','Median regression + random forest',pair),
('baseline_single_feature_v6','vol_ratio','RF with volatility ratio','Median regression + random forest',three),
('baseline_single_feature_v6','rv7d','RF with 7-day volatility','Median regression + random forest',pair+' + 7-day realized volatility'),
('mlp_fixed_liquidation_1h_v1','fixed_four_feature_mlp','MLP with 4 features','Linear model + small MLP neural network',four),
('linear_initialized_mlp_v1','initialized_mlp','MLP with 3 features','Linear model + small MLP neural network',three),
('rf_fixed_liquidation_1h_v1','rf_plus_liq1h','RF with 4 features','Median regression + random forest',four),
('rf_four_feature_tuning_v1','tuned_blend','Tuned RF with 4 features','Median regression + random forest; tuned settings',four),
('mlp_two_liquidation_v1','mlp','Original small MLP','MLP neural network',pair),
('neural_snapshot_vs_sequence_v1','patch_transformer','Time-series transformer','Small PatchTST-style transformer',three),
('lstm_walk_forward_v1','lstm','LSTM','Long short-term memory neural network','Selected inputs from liquidation, flow, volatility, funding, basis and past returns; varied by test window')]
rows=[]
for run,key,label,algo,features in spec:
 f=R/'experiments'/run/'scores.csv';assert (f.parent/'COMPLETE.json').exists()
 d=pd.read_csv(f);s=d[(d.split=='test')&(d.model==key)].iloc[0];assert s.rows==18481
 rows.append(dict(model=label,algorithm=algo,features=features,mae=float(s.mae),ic=float(s.rank_ic),source=str(f)))
rows.sort(key=lambda x:x['mae'])
story=[p('Models we tested','title'),p('ScaleWave | 4 November 2026','small'),p('<b>Goal:</b> predict how large Bitcoin\'s price move will be over the next 48 hours.'),p('<b>Ranked by prediction error (MAE): lower is better.</b><br/>IC shows how well predictions rank small and large moves: higher is better.','small')]
data=[[p(t,'head') for t in ['Rank / model','Algorithm','Features used','MAE','IC']]]
for i,r in enumerate(rows,1):data.append([p(f"<b>{i}. {r['model']}</b>",'cell'),p(r['algorithm'],'cell'),p(r['features'],'cell'),p(f"{r['mae']:.3f}",'cell'),p(f"{r['ic']:.4f}",'cell')])
table=Table(data,colWidths=[110,110,182,56,49],repeatRows=1)
table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF4F8'),colors.white]),('BACKGROUND',(0,1),(-1,1),colors.HexColor('#DDF1ED')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
story +=[table,Spacer(1,10),p('Main model comparisons only; repeated tuning and calibration variants are omitted. All rows use the same 18,481 test observations. MAE is in log basis points; IC is Spearman correlation.','small'),PageBreak(),p('What we achieved','title'),p('What we did','h'),p('We tested random forest models and three types of neural network: MLP, LSTM and a time-series transformer. We tried new input features, selected features and tuned model settings. We screened 520 additional feature fields.'),p('We used the same walk-forward method throughout: train on earlier data and test on later data.'),p('The features we found useful','h'),p('<b>1. 4-hour / 24-hour volatility ratio - best addition.</b><br/>It compares recent price fluctuations with those over the past day. Adding it to the original two features gave our best model.'),p('<b>2. 7-day realized volatility - small improvement.</b><br/>It measures how much prices fluctuated over the previous week. It also helped the original RF model, but less than the volatility ratio.'),p('<b>3. 1-hour liquidation historical active rate - helped MLP error only.</b><br/>It measures how often liquidation activity occurred over a longer lookback. Adding it reduced MLP error from <b>181.731 to 181.615</b>, but IC fell from <b>0.0616 to 0.0540</b>.'),p('The original features','h'),p('The <b>5-minute and 15-minute liquidation historical active rates</b> were our starting inputs. They measure the frequency of liquidation activity, not its dollar amount.'),p('Conclusion','h'),p('<b>We slightly beat the baseline using new features.</b> The winning algorithm was <b>median regression + random forest</b>, using the two original liquidation features and the <b>4-hour / 24-hour volatility ratio</b>.'),p('Its error fell from <b>181.526 to 181.248</b>, an improvement of about <b>0.15%</b>. IC increased from <b>0.0774 to 0.1068</b>.')]
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(navy);c.drawString(44,23,'ScaleWave | Simple model results');c.drawRightString(A4[0]-44,23,str(d.page))
SimpleDocTemplate(str(P),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=32,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(P);assert len(reader.pages)==2,len(reader.pages)
text=' '.join(x.extract_text() for x in reader.pages)
for r in rows:assert f"{r['mae']:.3f}" in text and f"{r['ic']:.4f}" in text
sys.path.insert(0,str(R/'tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
doc=pdfium.PdfDocument(str(P))
for i in range(len(doc)):
 pg=doc[i];pg.render(scale=1.5).to_pil().save(T/f'page_{i+1}.png');pg.close()
doc.close()
P.with_suffix('.json').write_text(json.dumps(dict(rows=rows,pages=2,text_checks='PASS'),indent=2),encoding='utf-8')
print(P)
