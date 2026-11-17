"""Complete ranked daily research report; values read from saved results."""
from pathlib import Path
import json,hashlib,sys
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs/ranked_daily';TMP.mkdir(parents=True,exist_ok=True)
PDF=OUT/'ScaleWave_Full_Model_Ranking_2026-11-04.pdf'
catalog=[]
def add(run,model,label,features):catalog.append((run,model,label,features))
run='baseline_single_feature_v6'
for model,label,features in [
 ('baseline','Original RF blend baseline','A+B'),
 ('vol_ratio','RF blend + volatility ratio','A+B+C'),
 ('rv7d','RF blend + 7-day realized volatility','A+B+G'),
 ('rv4h','RF blend + 4-hour realized volatility','A+B+E'),
 ('rv24h','RF blend + 24-hour realized volatility','A+B+F'),
 ('past48h_move','RF blend + past 48-hour absolute move','A+B+H'),
 ('funding_persistence','RF blend + 15-minute funding persistence','A+B+I'),
 ('selected','RF blend: validation-selected added feature','Varies')]:add(run,model,label,features)
for run,model,label,features in [
 ('lstm_walk_forward_v1','lstm','LSTM: validation-selected sequence model','LSTM pool'),
 ('lstm_walk_forward_v1','lstm_hybrid','LSTM + median-regression hybrid','LSTM pool'),
 ('lstm_feature_architecture_v3','selected_lstm','LSTM after feature and architecture selection','LSTM pool'),
 ('mlp_two_liquidation_v1','mlp','Standard two-input MLP','A+B'),
 ('mlp_two_liquidation_warmstart_v2','mlp','Two-input MLP with linear initialization','A+B'),
 ('mlp_two_liquidation_recency_v3','mlp','Recency-weighted two-input MLP','A+B'),
 ('mlp_hyperparameter_tuning_v4','mlp','Hyperparameter-tuned two-input MLP','A+B'),
 ('mlp_baseline_correction_v5','corrected','Baseline + MLP error correction','A+B'),
 ('vol_ratio_plus_level_v1','ratio_plus_level','RF blend: ratio + 24-hour volatility level','A+B+C+F'),
 ('neural_snapshot_vs_sequence_v1','residual_mlp','Three-input residual MLP','A+B+C'),
 ('neural_snapshot_vs_sequence_v1','patch_transformer','Small PatchTST-style transformer','A+B+C'),
 ('linear_initialized_mlp_v1','linear_only','Three-input linear model alone (control)','A+B+C'),
 ('linear_initialized_mlp_v1','initialized_mlp','Three-input linear-initialized residual MLP','A+B+C'),
 ('mlp_feature_discovery_v1','selected_mlp','MLP: validation-selected fourth feature','Varies'),
 ('mlp_fixed_liquidation_1h_v1','linear_4features','Four-input linear model alone (control)','A+B+C+D'),
 ('mlp_fixed_liquidation_1h_v1','fixed_four_feature_mlp','Four-input linear-initialized residual MLP','A+B+C+D'),
 ('rf_fixed_liquidation_1h_v1','rf_plus_liq1h','Four-input RF blend, original settings','A+B+C+D'),
 ('rf_four_feature_tuning_v1','nested_original_control','Four-input RF: longer-validation control','A+B+C+D'),
 ('rf_four_feature_tuning_v1','tuned_blend','Four-input hyperparameter-tuned RF blend','A+B+C+D'),
 ('rf_four_feature_tuning_v1','selected_rf_alone','Selected RF component alone (diagnostic)','A+B+C+D'),
 ('vol_ratio_calibration_diagnostics_v1','half_offset','Three-input RF blend + half-offset calibration','A+B+C'),
 ('vol_ratio_calibration_diagnostics_v1','full_offset','Three-input RF blend + full-offset calibration','A+B+C'),
 ('vol_ratio_calibration_diagnostics_v1','scale_only','Three-input RF blend + scale-only calibration','A+B+C')]:add(run,model,label,features)
assert len(catalog)==31
records=[];sources={}
for run,model,label,features in catalog:
    folder=ROOT/'experiments'/run;assert (folder/'COMPLETE.json').exists()
    path=folder/'scores.csv';df=pd.read_csv(path)
    if 'split' in df:df=df[df['split']=='test']
    row=df[df.model==model].iloc[0];assert row.rows==18481
    records.append(dict(model=label,features=features,mae=float(row.mae),spearman_ic=float(row.rank_ic),pearson_ic=float(row.pearson_ic),run=run,model_key=model,source=str(path)))
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
ranked=pd.DataFrame(records).sort_values(['mae','model']).reset_index(drop=True);ranked.insert(0,'rank',range(1,len(ranked)+1))
ranked.to_csv(OUT/'ScaleWave_Full_Model_Ranking_2026-11-04.csv',index=False)

navy=colors.HexColor('#16324F');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#52606D');pale=colors.HexColor('#EEF4F8');green=colors.HexColor('#DDF1ED')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=navy,spaceAfter=7),
 'subtitle':ParagraphStyle('subtitle',fontName='Helvetica',fontSize=10,leading=13,textColor=gray,spaceAfter=13),
 'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=12,leading=15,textColor=teal,spaceBefore=12,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,textColor=navy,spaceAfter=7),
 'bullet':ParagraphStyle('bullet',fontName='Helvetica',fontSize=10,leading=14,textColor=navy,leftIndent=10,firstLineIndent=-8,spaceAfter=6),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=11.5,textColor=gray,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.2,leading=10.1,textColor=navy),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8.2,leading=10.1,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8.1,leading=10.5,textColor=navy,spaceAfter=5),
 'feature':ParagraphStyle('feature',fontName='Helvetica-Bold',fontSize=10.4,leading=13,textColor=teal,spaceAfter=5)}
def P(text,style='body'):return Paragraph(text,styles[style])
def B(text):return P('- '+text,'bullet')
story=[]
story+=[P('All tested models: results and findings','title'),P('ScaleWave | Daily research report | 4 November 2026','subtitle'),
 P('<b>Objective:</b> predict the size of Bitcoin\'s start-to-end price move over the next 48 hours and reduce prediction error. This predicts move size, not direction.'),
 P('The results below use the same <b>18,481 historical test observations</b>, from 1 January 2024 to 22 January 2025. Training used earlier observations; testing used later observations in four walk-forward windows.'),
 P('What we did','h'),
 B('Compared LSTM networks, small MLPs and a small time-series transformer with the original regression/random-forest model.'),
 B('Tested feature selection, network architectures, linear initialization, recency weighting, error correction and MLP hyperparameter tuning.'),
 B('Added six individual features to the original RF blend. Tested calibration and a combination of volatility ratio plus absolute volatility level.'),
 B('Screened 520 additional feature fields. Tested selected additions in the MLP, then tested the 1-hour liquidation feature in every window.'),
 B('Tested that fourth feature in the RF blend. Tuned eight RF configurations and six blend weights using three earlier validation quarters per window.'),
 P('What the measured results show','h'),
 B('<b>Best overall:</b> median regression + RF using the two original liquidation rates and volatility ratio. MAE <b>181.248</b>, IC <b>0.1068</b>. Original baseline: MAE 181.526, IC 0.0774. The MAE reduction is about 0.15%.'),
 B('<b>Lowest neural MAE:</b> the four-input linear-initialized MLP, at <b>181.615</b>. Adding the fourth feature reduced MLP MAE from 181.731, but IC fell from 0.0616 to 0.0540. It did not beat the original baseline.'),
 B('<b>The same feature behaved differently across models.</b> The fourth input slightly reduced overall MLP error but worsened the RF blend. RF tuning did not beat the baseline.'),
 B('<b>Better validation results did not always carry into later test data.</b> The selected-feature MLP and tuned RF are examples. This is an observed result, not a diagnosis of a single cause.'),
 P('How to read this report','h'),
 P('<b>MAE:</b> average prediction error in log basis points (bps); lower is better. <b>IC:</b> Spearman correlation, measuring how well predictions rank smaller versus larger actual moves; higher is better. <b>RF blend:</b> median regression plus random forest. <b>MLP:</b> a small neural network. <b>Linear-initialized residual MLP:</b> a fitted linear prediction plus a neural correction.'),
 P('Scope: the ranking contains 31 distinct completed full-period model, control and calibration comparisons from this day\'s work. Repeated baseline rows and identical prediction aliases appear once. Individual tuning trials with validation-only scores are not ranked as full-test models. GARCH was discussed but not trained.','small'),
 P('The historical data were repeatedly examined during research. These results are not confirmation on new, unseen data.','small'),PageBreak()]

story+=[P('Complete ranking by test MAE','title'),P('All 31 comparisons | Same test rows | Lower MAE is better','subtitle')]
tabledata=[[P(x,'head') for x in ['#','Model / tested procedure','Inputs','MAE (bps)','IC']]]
for _,r in ranked.iterrows():tabledata.append([P(str(r['rank']),'cell'),P(escape(r.model),'cell'),P(r.features,'cell'),P(f'{r.mae:.3f}','cell'),P(f'{r.spearman_ic:.4f}','cell')])
table=Table(tabledata,colWidths=[29,269,79,68,62],repeatRows=1,hAlign='LEFT')
commands=[('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[pale,colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3.8),('BOTTOMPADDING',(0,0),(-1,-1),3.8),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]
for i,(_,r) in enumerate(ranked.iterrows(),1):
    if r['model_key']=='vol_ratio':commands.append(('BACKGROUND',(0,i),(-1,i),green))
    if r['model_key']=='fixed_four_feature_mlp':commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#E5EDF9')))
table.setStyle(TableStyle(commands));story+=[table,Spacer(1,7),P('Input letters are defined on pages 3-4. RF feature selection varied the added input among C and E-I. MLP feature discovery varied the fourth input among D, J and K. LSTM used varying subsets of a 17-input pool listed on page 5.','small'),
 P('Controls and calibration variants are included for completeness. The RF-alone row is the component chosen for blend performance, not a separately selected standalone-RF winner. The longer-validation RF control used the same three-quarter selection history as RF tuning.','small'),PageBreak()]

def code(name):return P(escape(name).replace('__','__<br/>'),'code')
def feature(letter,title,name,meaning,result):
    return KeepTogether([P(letter+' | '+title,'feature'),code(name),P(meaning),P('<b>Measured result:</b> '+result),Spacer(1,9)])
story+=[P('Full feature names: the main inputs','title'),P('A and B are the original inputs. C and D are the main additions investigated.','subtitle')]
story.append(feature('A','5-minute liquidation historical active rate','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value','Historical frequency of cross-asset liquidation activity at the 5-minute lookback. This is an activity rate, not a liquidation dollar amount.','A and B together form the original model input pair. Their RF blend baseline scored MAE 181.526 and IC 0.0774; these are joint results, not separate feature scores.'))
story.append(feature('B','15-minute liquidation historical active rate','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value','The corresponding historical activity rate at the 15-minute lookback.','Retained alongside A in the original baseline and all main later MLP/RF comparisons. No individual A-versus-B removal test is claimed here.'))
story.append(feature('C','4-hour / 24-hour volatility ratio','PRICE__vol_ratio_4h_24h','Compares recent 4-hour price fluctuations with the past 24 hours, adjusted for window length. Calculated from past returns only.','Added to A+B, it gave the best overall RF blend: MAE 181.248 and IC 0.1068, versus 181.526 and 0.0774 for the original baseline.'))
story.append(feature('D','1-hour liquidation historical active rate','F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','The same historical activity-rate measurement at a longer, 1-hour lookback.','Added to A+B+C, MLP MAE improved 181.731 to 181.615, while IC fell 0.0616 to 0.0540. RF blend MAE worsened 181.248 to 181.971; tuning the four-input RF blend produced 182.252.'))
story+=[P('The fourth-feature MLP improvement was small and uncertain. Excluding the already successful final window, the four-input MLP scored MAE 187.801 versus 187.791 for the three-input MLP on the same first-three-window rows. Its overall gain was not an independent confirmation of the feature.','small'),PageBreak()]

story+=[P('Full feature names: other tested additions','title'),P('These are tested candidates, not a list of confirmed useful features.','subtitle')]
others=[
 ('E','4-hour realized volatility','PRICE__realized_vol_4h','Size of recent price fluctuations over 4 hours.','RF blend using A+B+E: MAE 181.957, IC 0.0881. MAE was worse than the original baseline.'),
 ('F','24-hour realized volatility','PRICE__realized_vol_24h','Size of price fluctuations over 24 hours.','A+B+F: MAE 183.030, IC 0.0489. Adding F alongside the ratio (A+B+C+F) also worsened the best RF blend: MAE 182.645.'),
 ('G','7-day realized volatility','PRICE__realized_vol_168h','Size of price fluctuations over 168 hours, or seven days.','A+B+G: MAE 181.463, IC 0.0810. A small historical improvement over the original baseline, less than the volatility-ratio result.'),
 ('H','Past 48-hour absolute endpoint move','PRICE__absolute_net_move_48h','Size of the net start-to-end move in the preceding 48 hours, ignoring direction.','A+B+H: MAE 183.153, IC -0.0086. Did not improve the baseline.'),
 ('I','15-minute funding persistence percentile','F4Z__funding__funding_persistence_percentile__15m__value','The supplied funding-persistence measure, expressed as a percentile, at a 15-minute lookback.','A+B+I: MAE 182.028, IC 0.0445. Did not improve the baseline.'),
 ('J','15-minute basis persistence percentile','F4Z__basis__basis_persistence_percentile__15m__value','The supplied basis-persistence measure, expressed as a percentile, at a 15-minute lookback.','Selected by the MLP screen in two windows. Test MAE worsened by 4.650 and 0.152 bps versus the three-input MLP in those windows.'),
 ('K','1-hour funding persistence percentile','F4Z__funding__funding_persistence_percentile__1h__value','The supplied funding-persistence measure, expressed as a percentile, at a 1-hour lookback.','Selected by the MLP screen in the July window. Test MAE worsened by 13.341 bps versus the three-input MLP in that window.')]
# Compact cards keep all seven exact identifiers readable on one page.
for letter,title,name,meaning,result in others:
    story.append(KeepTogether([P(letter+' | '+title,'feature'),P(escape(name),'code'),P(meaning+' <b>Result:</b> '+result,'small'),Spacer(1,6)]))
story+=[PageBreak(),P('Additional tested inputs and result notes','title'),P('Completeness notes for the LSTM pool, controls and tuning results','subtitle'),
 P('The broader LSTM pool','h'),P('The 17-input LSTM candidate pool contained A, B, C, E, F, G and I, plus the ten exact fields below. Different windows selected different subsets or ensembles. Their presence in the pool does not establish individual usefulness.','small')]
lstm=json.loads((ROOT/'experiments/lstm_walk_forward_v1/CONTRACT.json').read_text())['features']['market']
known={
 'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
 'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
 'PRICE__vol_ratio_4h_24h','PRICE__realized_vol_4h','PRICE__realized_vol_24h','PRICE__realized_vol_168h',
 'F4Z__funding__funding_persistence_percentile__15m__value'}
extras=[c for c in lstm if c not in known];assert len(lstm)==17 and len(extras)==10
for c in extras:story+=[P(escape(c),'code')]
story+=[P('Calibration and model diagnostics','h'),
 P('Calibration adjusted prediction levels using earlier validation data. Half-offset, full-offset and scale-only versions all worsened the best three-input RF blend\'s test MAE: 181.538, 182.895 and 183.575 respectively, versus 181.248 without calibration.'),
 P('The linear-only controls show what the fitted linear component achieved without a neural correction. Three-input linear MAE was 181.744, versus 181.731 with the initialized MLP. Four-input linear MAE was 181.708, versus 181.615 with the initialized MLP.'),
 P('In the first window of the earlier residual-MLP/transformer comparison, validation selected zero training epochs for all seeds, so both predicted a constant training median. Later linear initialization replaced that constant starting point with a fitted input-dependent linear model.'),
 P('What RF tuning actually selected','h'),
 P('January and April retained the original RF settings with a 25% RF blend weight. July selected depth 5 and minimum leaf size 64, with 50% RF weight. October selected depth 5, minimum leaf size 128 and 75% feature sampling, with 25% RF weight. All selected forests had 128 trees.'),
 P('The same-validation-history original-parameter control scored MAE 181.670. The tuned blend scored 182.252, so selected RF changes did not improve the test result. The pooled three-quarter validation score is not directly comparable with earlier single-quarter validation scores.'),
 P('Evidence and reporting scope','h'),
 P('Numbers were read directly from completed experiment scores.csv files. Rank order uses full-precision test MAE; the PDF displays three decimals for MAE and four for IC. Repeated references and the identical challenger/selected MLP feature-discovery outputs were deduplicated. Training-only trials, stopped drafts and unexecuted proposals are excluded. No new training was performed to create this report.','small'),
 P('The adjacent ranking CSV maps every row to its run and source file. The report manifest records source-file hashes. All results are from repeatedly examined historical data; none is presented as fresh-data confirmation.','small')]

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D5DEE6'));canvas.line(44,32,A4[0]-44,32)
    canvas.setFont('Helvetica',8);canvas.setFillColor(gray)
    canvas.drawString(44,20,'ScaleWave | Full model ranking | 4 November 2026')
    canvas.drawRightString(A4[0]-44,20,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=32,bottomMargin=44,title='ScaleWave - full model ranking and feature findings - 4 November 2026',author='ScaleWave research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF);text='\n'.join(page.extract_text() for page in reader.pages)
for _,r in ranked.iterrows():assert f'{r.mae:.3f}' in text and f'{r.spearman_ic:.4f}' in text
assert len(reader.pages)==5,f'Unexpected page count: {len(reader.pages)}'
# Full identifiers may wrap across lines; verify them after removing whitespace.
compact=''.join(text.split())
for c in known|set(extras)|{r[2] for r in others}|{'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value'}:
    assert c in compact,c
(TMP/'extracted_text.txt').write_text(text,encoding='utf-8')
sys.path.insert(0,str(ROOT/'tmp/pdf_render_dependencies'))
import pypdfium2 as pdfium
pdf=pdfium.PdfDocument(str(PDF))
for i in range(len(pdf)):
    page=pdf[i];page.render(scale=1.5).to_pil().save(TMP/f'page_{i+1}.png');page.close()
pdf.close()
manifest=dict(pdf=str(PDF),pages=len(reader.pages),ranked_comparisons=len(ranked),sources=sources,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),text_checks='PASS',visual_review='pending')
(OUT/'ScaleWave_Full_Model_Ranking_2026-11-04_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest|{'sources':len(sources)},indent=2))
