"""Typeset the verified 72h report into a readable paginated PDF."""
from pathlib import Path
import html
import json
from bs4 import BeautifulSoup
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'Deliverables/report_72h_20261114/report_72h.html'
OUTPUT=ROOT/'Report/BTC_72h_Models_Features_Rankings_2026-11-14.pdf'
pdfmetrics.registerFont(TTFont('Report','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='Report',boldItalic='ReportBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Report',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='CellReport',fontName='Report',fontSize=8.4,leading=11,wordWrap='CJK'))
styles.add(ParagraphStyle(name='HeadReport',fontName='ReportBold',fontSize=8.4,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='SectionReport',fontName='ReportBold',fontSize=16,leading=20,spaceBefore=14,spaceAfter=10,keepWithNext=True))
styles.add(ParagraphStyle(name='TitleReport',fontName='ReportBold',fontSize=25,leading=30,spaceAfter=12))
W,H=landscape(A4);width=W-88

def clean(s):return str(s).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
def p(s,style='BodyReport'):return Paragraph(html.escape(clean(s)),styles[style])
def make_table(rows):
    n=len(rows[0])
    if n==7: fractions=[.045,.285,.105,.13,.13,.13,.175];total=sum(fractions);fractions=[x/total for x in fractions]
    elif n==2:fractions=[.37,.63]
    elif n==1:fractions=[1]
    else:fractions=[1/n]*n
    cells=[[p(c,'HeadReport' if i==0 else 'CellReport') for c in row] for i,row in enumerate(rows)]
    table=LongTable(cells,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17354b')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f8')]),
        ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d0d7df')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return table

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#52606c'))
    canvas.drawString(44,22,'72 hour excursion models | Historical research | 14 November 2026')
    canvas.drawRightString(W-44,22,f'Page {doc.page}');canvas.restoreState()

def best_model_summary():
    import pandas as pd
    rows=[['Target','Forecast','MAE','RMSE','Spearman IC']]
    for target,relative in [('UP','excursion_neural_72h_v1/20261113T123304519186Z'),
                            ('DOWN','neural_market_down_72h_v1/20261114T123111032438Z')]:
        scores=pd.read_csv(ROOT/'Code/experiments'/relative/'metrics.csv')
        scores=scores[scores.target.eq(target.lower()+'_magnitude_log_bps') & scores.period.eq('ALL')]
        for key,label in [('selected','Best MLP' if target=='UP' else 'Best GRU16'),('training_median','Median baseline'),('training_mean','Mean baseline')]:
            s=scores[scores.model.eq(key)].iloc[0]
            rows.append([target,label,f'{s.mae:.4f}',f'{s.rmse:.4f}',f'{s.ic:.4f}'])
    story=[p('Best models and their scores','SectionReport'),
        p('Best means lowest observed MAE among the completed 72-hour experiments reviewed here. Use the original regularized MLP for UP and the 16-unit market GRU for DOWN. Scores below use the same 18,385 historical forecast origins per target.'),
        make_table(rows),Spacer(1,10),
        p('Both best models beat both mean and median on MAE. On RMSE, both beat the median but neither beats the mean. UP improves median MAE by 1.3753%; DOWN improves it by 1.0298%. These are historical research results, not untouched-data confirmation.'),
        p('The following two model cards connect the winning architectures to their prediction equations and feature construction. Full exact quarterly UP feature identifiers, the 30 DOWN feature identifiers, architectures and the complete generated-feature dictionary follow in the main report.'),PageBreak(),
        p('Best UP model features and prediction formula','SectionReport'),
        p('Architecture: standardized inputs and missingness flags → 32 ReLU units → 1 linear output. AdamW learning rate 0.001; weight decay 0.1; MAE loss; 60 epochs; batch 256; seed 48. Feature sets change by quarterly issue, so the best UP policy is four fitted models.'),
        make_table([['Issue','Selected feature groups','Count / model weight'],
            ['January','Five F4Z basis/funding/flow/price-retention inputs + eight price measurements at each of 1h and 4h.','21 / 50%'],
            ['April','Three F4Z OI-recency/basis/flow inputs + eight price measurements at each of 1h and 4h.','19 / 50%'],
            ['July','Eight price measurements at 1h,4h,12h,24h,48h,168h + RV ratios 1h/24h,4h/24h,24h/168h.','51 / 50%'],
            ['October onward','F4Z liquidation historical activity rate at 5m and realized-variance robust score at 15m.','2 / 100%']]),Spacer(1,8),
        p('Prediction: z = clip(training-standardized [median-imputed x, missing flags], -8, 8); h = ReLU(W1 z + b1); f = max(0, u × (W2 h + b2)); forecast = w f + (1-w)m. Here m is the mature training-target median, u=max(m,1), and w is the model weight above. W and b are learned parameters.'),
        p('Eight price measurements: RV=sqrt(sum r²); upside RV=sqrt(sum max(r,0)²); downside RV=sqrt(sum min(r,0)²); net=sum r; range=H-L; distance from high=H-C; distance from low=C-L; largest move=max |r|. r denotes strictly earlier 5-minute log-bps returns; H,L,C are local high, low and current cumulative log-price levels. RV ratio=(RV_short/sqrt(short hours))/(RV_long/sqrt(long hours)), with denominator floor 1e-6.'),
        p('F4Z fields are inherited masked inputs. Their exact names are listed later; their upstream percentile/robust-score formulas were not independently reconstructed and are not invented here.'),PageBreak(),
        p('Best DOWN model features and prediction formula','SectionReport'),
        p('Architecture: four snapshots at t-18h,t-12h,t-6h,t; each contains 30 engineered values and 30 missingness flags → one-layer GRU16 → linear output head. AdamW learning rate 0.001; weight decay 0.03; MAE loss; 24 epochs; batch 512; at most 4096 training rows.'),
        make_table([['Inputs','Formula and lookback'],
            ['72h hourly volatility','sqrt(sum R² / 72)'],
            ['72h normalized momentum','sum R / max(sqrt(sum R²),1e-8)'],
            ['72h range position','(C-L) / max(H-L,1e-8)'],
            ['72h upside variance share','sum max(R,0)² / max(sum R²,1e-16)'],
            ['168h drawdown; 336h jump share','(H-C)/max(sqrt(sum R²),1e-8); max R²/max(sum R²,1e-16)'],
            ['20 market summaries','Five sources × mean/deviation × 24h/72h. Sources: flow pressure, flow activity, local and peer historical liquidation activity rates, liquidation breadth. Deviation = source(t-30min) - window mean.'],
            ['4 interactions','For h=24h,72h: liquidation-rate mean × hourly volatility; (flow-pressure mean - 0.5) × normalized momentum.']]),Spacer(1,8),
        p('R is an hourly log-bps return formed from exactly 12 earlier finite 5-minute returns. H,L,C refer to the local cumulative hourly price path. Market means use [t-h,t), exclude the origin, and require at least 80% of expected half-hour snapshots. The same 30 formulas are evaluated at each of the four sequence timestamps.'),
        p('Prediction: h = final hidden state of GRU16(sequence); a = Wout h + bout; forecast = max(0, m + w × 0.75u × tanh(a)), where u=max(m,1). w=0.5 for January/April/July fits and 0.25 from October. If less than 80% of sequence inputs are finite, w=0 and the forecast is m. Preprocessing uses training data only.'),PageBreak()]
    return story

def main():
    soup=BeautifulSoup(SOURCE.read_text(encoding='utf-8'),'html.parser')
    story=[p('72 hour model research report','TitleReport'),p('BTC canonical UP and DOWN excursions | Models, features, hyperparameters and verified results')]
    story.extend(best_model_summary())
    for section in soup.find_all('section'):
        title=section.h2.get_text(' ',strip=True)
        if title=='Source linked technical appendix':continue
        if title in ['UP model ranking','DOWN model ranking','Best UP model architecture and hyperparameters','Best DOWN GRU architecture and hyperparameters','How the features were created','Other model architectures and hyperparameters']:
            story.append(PageBreak())
        story.append(p(title,'SectionReport'))
        for node in section.children:
            if getattr(node,'name',None)=='p':story.append(p(node.get_text(' ',strip=True)))
            elif getattr(node,'name',None)=='table':
                rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in node.find_all('tr')]
                story.extend([make_table(rows),Spacer(1,9)])
    story.extend([PageBreak(),p('Source runs and audit evidence','SectionReport'),p('All 14 prediction hashes and aligned target/origin keys were verified. Pooled scores were recomputed without training or inference. Paths below are relative to Code/experiments in the Week 2 workspace. The accompanying HTML report retains expandable exact contracts and selection plans; model_rankings.csv and period_metrics.csv contain the numerical tables.')])
    audit=json.loads((SOURCE.parent/'audit.json').read_text())
    story.append(make_table([['Experiment','Run path']]+[[v['experiment'],v['path']] for v in audit]))
    story.extend([PageBreak(),p('Complete generated feature dictionary','SectionReport'),p('All 325 generated feature definitions are listed here. They describe the candidate universe, not 325 inputs used by each model. The exact winning UP and DOWN inventories appear earlier.')])
    dictionary=soup.find('summary',string='All generated price and market formulas').parent.find('table')
    head=[c.get_text(' ',strip=True) for c in dictionary.find('tr').find_all(['th','td'])]
    wanted=[head.index('feature'),head.index('formula')]
    rows=[['Feature','Formula']]
    for tr in dictionary.find_all('tr')[1:]:
        vals=[c.get_text(' ',strip=True) for c in tr.find_all('td')]
        rows.append([vals[i] for i in wanted])
    story.append(make_table(rows))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=(W,H),leftMargin=44,rightMargin=44,topMargin=35,bottomMargin=40,
        title='72 hour model research report',author='ML research workspace')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(OUTPUT);text='\n'.join(page.extract_text() or '' for page in reader.pages)
    for value in ['234.3441','232.3157','237.6120','234.7329','GRU','325']:
        assert value in text,value
    assert len(rows)==326
    print(json.dumps(dict(pdf=str(OUTPUT),pages=len(reader.pages),bytes=OUTPUT.stat().st_size)))

if __name__=='__main__':main()
