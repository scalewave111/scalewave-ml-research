"""One-page beginner-friendly report from verified experiment scores."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
OUT=HERE/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
TMP=HERE/'tmp/pdfs';TMP.mkdir(parents=True,exist_ok=True)
PDF=OUT/'ScaleWave_Daily_Research_Report_2026-11-04.pdf'
items=[
 ('Original baseline','baseline_single_feature_v6','baseline'),
 ('Best RF blend - 3 inputs','baseline_single_feature_v6','vol_ratio'),
 ('MLP - 3 inputs, linear-initialized','linear_initialized_mlp_v1','initialized_mlp'),
 ('MLP - 4 inputs, linear-initialized','mlp_fixed_liquidation_1h_v1','fixed_four_feature_mlp'),
 ('RF blend - 4 inputs','rf_fixed_liquidation_1h_v1','rf_plus_liq1h'),
 ('Tuned RF blend - 4 inputs','rf_four_feature_tuning_v1','tuned_blend'),
 ('Small time-series transformer - 3 inputs','neural_snapshot_vs_sequence_v1','patch_transformer'),
 ('LSTM','lstm_walk_forward_v1','lstm')]
rows=[];sources={}
for label,run,model in items:
    folder=HERE/'experiments'/run
    assert (folder/'COMPLETE.json').exists()
    path=folder/'scores.csv';data=pd.read_csv(path)
    row=data[(data['split']=='test')&(data.model==model)].iloc[0]
    assert row.rows==18481
    rows.append(dict(label=label,mae=float(row.mae),ic=float(row.rank_ic)))
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()

navy=colors.HexColor('#16324F');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#52606D')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=22,leading=25,textColor=navy,spaceAfter=4),
 'date':ParagraphStyle('date',fontName='Helvetica',fontSize=9.5,leading=12,textColor=gray,spaceAfter=10),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.5,leading=12.5,textColor=navy,spaceAfter=5),
 'section':ParagraphStyle('section',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=teal,spaceBefore=9,spaceAfter=5),
 'bullet':ParagraphStyle('bullet',fontName='Helvetica',fontSize=9.5,leading=12.3,textColor=navy,leftIndent=10,firstLineIndent=-8,spaceAfter=4),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8,leading=10.5,textColor=gray,spaceAfter=4),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9.2,leading=11.5,textColor=navy),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=9.2,leading=11.5,textColor=colors.white)}
def p(text,style='body'):return Paragraph(text,styles[style])
story=[p('Today\'s model research','title'),p('ScaleWave | 4 November 2026','date'),
 p('<b>Goal:</b> predict the size of Bitcoin\'s move over the next 48 hours and reduce prediction error. The target is the start-to-end move, not its direction.'),
 p('What we did','section')]
for text in [
 'Compared neural networks (MLP, LSTM and a small time-series transformer) with the original model.',
 'Tested new inputs, screened 520 additional feature fields, and checked whether selected features helped the MLP.',
 'Tested a fourth input in the RF blend and tuned eight RF configurations with six blend weights.',
 'Used time-ordered walk-forward tests: train on earlier data, evaluate on later data. All results below use the same 18,481 test observations.']:
    story.append(p('- '+text,'bullet'))
story+=[p('Main results','section'),p('<b>MAE:</b> average prediction error in log basis points (bps); lower is better. <b>IC:</b> how well predictions rank small and large moves; higher is better.','small')]
tabledata=[[p('Model','head'),p('MAE (bps)','head'),p('IC (Spearman)','head')]]
for r in rows:tabledata.append([p(r['label'],'cell'),p(f"{r['mae']:.3f}",'cell'),p(f"{r['ic']:.4f}",'cell')])
table=Table(tabledata,colWidths=[310,78,119],repeatRows=1,hAlign='LEFT')
table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F1F5F9'),colors.white]),('BACKGROUND',(0,2),(-1,2),colors.HexColor('#DDF1ED')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D5DEE6'))]))
story+=[table,Spacer(1,6),p('<b>RF blend</b> = median regression + random forest. <b>Linear-initialized MLP</b> = a fitted linear prediction plus a small neural correction.','small'),
 p('<b>Three inputs:</b> 5-minute and 15-minute liquidation historical active rates, plus the 4-hour/24-hour volatility ratio. <b>Fourth input:</b> the 1-hour liquidation historical active rate.','small'),
 p('What we concluded','section')]
for text in [
 '<b>Best overall:</b> the three-input RF blend. It reduced MAE from 181.526 to 181.248 (about 0.15%) and improved IC.',
 '<b>Best neural MAE:</b> the four-input MLP. The fourth input reduced MAE from 181.731 to 181.615, but IC fell and it still did not beat the original baseline.',
 '<b>Feature benefits depend on the model.</b> The fourth input slightly helped MLP error, but worsened the RF blend overall. RF tuning did not beat the baseline.',
 '<b>Validation gains did not always carry forward.</b> Several features and settings looked better on earlier validation data but performed worse in later test periods.']:
    story.append(p('- '+text,'bullet'))
story+=[Spacer(1,5),p('Scope: the historical data were repeatedly examined during research. These results are not confirmation on new, unseen data. The small fourth-input MLP gain was not supported by an aggregate improvement when its previously successful final window was excluded.','small')]
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#D5DEE6'));c.line(44,31,A4[0]-44,31)
    c.setFillColor(gray);c.setFont('Helvetica',7.5)
    c.drawString(44,20,'Source: saved experiment scores and completed-run audits | No new training for this report')
    c.drawRightString(A4[0]-44,20,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=34,bottomMargin=42,title='ScaleWave daily research report - 4 November 2026',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF);assert len(reader.pages)==1,f'Expected one page, got {len(reader.pages)}'
text='\n'.join(page.extract_text() for page in reader.pages)
for r in rows:assert f"{r['mae']:.3f}" in text and f"{r['ic']:.4f}" in text
assert 'What we did' in text and 'What we concluded' in text
(TMP/'report_text.txt').write_text(text,encoding='utf-8')
sys.path.insert(0,str(HERE/'tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
pdf=pdfium.PdfDocument(str(PDF));page=pdf[0];im=page.render(scale=1.6).to_pil();im.save(TMP/'daily_report_page1.png');page.close();pdf.close()
(OUT/'ScaleWave_Daily_Research_Report_2026-11-04_manifest.json').write_text(json.dumps(dict(report=str(PDF),pages=len(reader.pages),sources=sources,rows=rows,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),text_checks='PASS',visual_review='pending'),indent=2),encoding='utf-8')
print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),preview=str(TMP/'daily_report_page1.png'))))
