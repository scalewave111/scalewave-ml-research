from pathlib import Path
import json,re,math,html
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape,A4
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parents[1];SUP=ROOT/'Report/combined_magnitude_support_2026-11-14';TMP=ROOT/'tmp/pdfs/combined';TMP.mkdir(parents=True,exist_ok=True)
WS=json.loads((SUP/'winner_models_and_features.json').read_text());HS=['15m','1h','4h','8h','16h','48h','72h','120h','200h','300h']
OUT=ROOT/'Report/BTC_All_Horizons_MAE_Models_and_Features_2026-11-14.pdf'
navy=colors.HexColor('#17354a');teal=colors.HexColor('#087f8c')
def side(w):return 'UP' if w['target'].startswith('up') else 'DOWN'
def membership(w,f):return ','.join(q['issue'][5:] for q in w['quarters'] if f in q['features'])
# Wide matrix retains exact names, not feature IDs. Vector text remains sharp at zoom.
cw=610;left=190;rowh=3180;mw=left+10*cw+40;mh=2*rowh+175
c=canvas.Canvas(str(TMP/'matrix.pdf'),pagesize=(mw,mh));c.setTitle('Full feature matrix by horizon and target')
c.setFillColor(navy);c.setFont('Helvetica-Bold',28);c.drawString(25,mh-43,'BEST OBSERVED MAE | Full feature matrix | 15 minutes to 300 hours')
c.setFont('Helvetica',14);c.drawString(25,mh-70,'Exact input names; union across quarterly fits. [01,04,07,10] denotes January, April, July, October 2024 models. October continues through January 2025.')
c.drawString(25,mh-92,'This is an oversized vector matrix: zoom to read. The following standard-size pages give metrics, model architectures, hyperparameters and formulas. Missing indicators are described in model cards.')
top=mh-125
for i,h in enumerate(HS):
 x=left+i*cw;c.setFillColor(navy);c.rect(x,top-30,cw,30,fill=1,stroke=0);c.setFillColor(colors.white);c.setFont('Helvetica-Bold',17);c.drawString(x+10,top-22,h)
for j,target in enumerate(['UP','DOWN']):
 ytop=top-30-j*rowh;c.setFillColor(teal);c.rect(20,ytop-rowh,left-20,rowh,fill=1,stroke=0);c.setFillColor(colors.white);c.setFont('Helvetica-Bold',16);c.drawString(30,ytop-30,target)
 c.setFont('Helvetica',11);c.drawString(30,ytop-50,target.lower()+'_magnitude_log_bps')
 for i,h in enumerate(HS):
  w=next(w for w in WS if w['horizon']==h and side(w)==target);x=left+i*cw
  c.setFillColor(colors.HexColor('#eef4f7') if i%2==0 else colors.white);c.rect(x,ytop-rowh,cw,rowh,fill=1,stroke=0)
  c.setStrokeColor(colors.HexColor('#becbd4'));c.rect(x,ytop-rowh,cw,rowh,fill=0,stroke=1)
  c.setFillColor(navy);c.setFont('Helvetica-Bold',12);c.drawString(x+9,ytop-20,f"MAE {w['MAE']:.6f} | {len(w['features'])} distinct named inputs")
  c.setFont('Helvetica',10);c.drawString(x+9,ytop-37,w['forecast']);c.drawString(x+9,ytop-52,'Quarter membership in brackets; no predictors in a constant fallback.')
  yy=ytop-74
  for f in w['features']:
   text=f+' ['+membership(w,f)+']';size=min(9.2,(cw-20)/(max(len(text),1)*.60));c.setFont('Courier',size);c.drawString(x+9,yy,text);yy-=12.3
  assert yy>ytop-rowh+10
c.showPage();c.save()

W,H=landscape(A4);WIDTH=W-64;styles={k:ParagraphStyle(k,fontName=font,fontSize=size,leading=leading,textColor=navy,spaceAfter=6,splitLongWords=True) for k,font,size,leading in [('title','Helvetica-Bold',22,26),('head','Helvetica-Bold',12,16),('body','Helvetica',9,12),('small','Helvetica',7.6,10),('cell','Helvetica',7.7,10),('th','Helvetica-Bold',7.8,10)]};styles['th'].textColor=colors.white
story=[]
def p(t,st='body'):return Paragraph(html.escape(str(t)),styles[st])
def text(t,st='body'):story.append(p(t,st))
def page(t):
 if story:story.append(PageBreak())
 text(t,'title')
def table(head,rows,widths):
 a=Table([[p(x,'th') for x in head]]+[[p(x,'cell') for x in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');a.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f7'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(a);story.append(Spacer(1,8))
def compact(x):
 if isinstance(x,dict):return '; '.join(str(k)+'='+compact(v) for k,v in x.items())
 if isinstance(x,list):return ', '.join(map(str,x))
 return str(x)
page('Combined magnitude model report')
text('15m | 1h | 4h | 8h | 16h | 48h | 72h | 120h | 200h | 300h','head')
text('Prepared 14 November 2026. Best means lowest pooled mean absolute error among completed saved forecasts with full matching coverage within each horizon and target. The first page is the requested full-name feature matrix. It includes every active quarterly predictor and the extra 120h output-scale input. Both targets are log basis points; DOWN is not ordinary price basis points.')
text('Targets: UP=max(0,10000*ln(max future path price / origin price)); DOWN=max(0,10000*ln(origin price / min future path price)), under each saved label contract. Each horizon is ranked separately; magnitudes and retained origins differ, so scores are not a cross-horizon contest.')
text('This report independently hashes completed prediction files, aligns origin keys and actuals, and recomputes MAE, median absolute error (MedianAE), RMSE, Spearman rank IC and bias. No model was trained or run for this report. Mean/median baselines are the saved training-only forecasts evaluated on the same rows. Percent improvement = 100*(baseline MAE-model MAE)/baseline MAE.')
text('Selection status matters: fixed variants and rejected candidates can win a hindsight MAE comparison. They are labelled explicitly. This audit finds 72h UP price_only MAE 234.005244, below the formerly highlighted selected MLP at 234.344132. Repeated research on these historical outcomes creates selection bias; best observed does not establish unseen-period superiority.')
for s in ['UP','DOWN']:
 page(s+' | MAE winners and baseline scores')
 rows=[]
 for w in WS:
  if side(w)!=s:continue
  b=w['baselines'];rows.append([w['horizon'],f"{w['MAE']:.6f}",f"{w['MedianAE']:.3f}",f"{w['RMSE']:.3f}",f"{w['IC']:.4f}",f"{b['mean']['MAE']:.3f}",f"{b['median']['MAE']:.3f}",f"{100*(1-w['MAE']/b['mean']['MAE']):.2f}%",f"{100*(1-w['MAE']/b['median']['MAE']):.2f}%"])
 table(['Horizon','Model MAE','MedianAE','RMSE','Rank IC','Mean MAE','Median MAE','Gain mean','Gain median'],rows,[50,88,75,75,65,85,85,77,WIDTH-600])
 text('MAE and MedianAE are different metrics. Model selection uses mean absolute error only. Positive gains indicate improvement. Baseline RMSE and MedianAE are supplied with each detailed model card.')
for w in WS:
 page(w['horizon']+' | '+side(w)+' | Best observed MAE model')
 text(w['forecast']+' | '+w['status'],'head');text('Source: Code/experiments/'+w['run'].replace('\\','/'),'small')
 table(['Forecast','MAE','MedianAE','RMSE','Rank IC','Bias'],[[name]+[f"{r[k]:.6f}" for k in ['MAE','MedianAE','RMSE','IC','Bias']] for name,r in [('Model',w),('Training mean',w['baselines']['mean']),('Training median',w['baselines']['median'])]] ,[140]+[(WIDTH-140)/5]*5)
 text(f"{w['rows']:,} evaluation origins. {len(w['features'])} distinct named inputs across quarterly models; exact names and quarter membership are in the opening matrix and companion HTML/JSON.")
 for q in w['quarters']:
  text(q['issue']+' | '+q['architecture']+f" | {len(q['features'])} inputs",'head')
  text(compact(q['specification']),'small');text(q['formula'])
  if 'components' in q:
   for n in q['components']:
    text(n['id']+' | '+n['kind']+' | '+n['formula'],'small');text(compact(n['settings']),'small')
  else:text('Hyperparameters: '+compact(q['hyperparameters']),'small')
  if 'reference_parameters' in q:text('Reference hyperparameters: '+compact(q['reference_parameters']),'small')
 if w['horizon']=='200h':text('Component identifiers are local references retained from the verified 200h artifact graph. Blends preserve the source formulas. Transformed correction uses training median/IQR, raw or signed-log/quadratic basis, missing flags then training mean/std scaling. Quadratic basis concatenates scaled x, sign(x)*log(1+abs(x)), x^2 and flags. Target unit is max(training median(y),1). Neural anchor GRUs use four exact snapshots and a bounded correction around the median. Fitted thresholds, leaf values and network weights remain in the hashed artifacts.')

page('Feature engineering | definitions and timing')
text('The matrix lists the exact engineered model inputs, not just their underlying data sources. SEQ prefixes are expanded in full. Training-created missing flags are architectural channels, not separately named raw features. A constant median fallback contributes no predictors. Formula tables below map all used base features; the SEQ rule reconstructs every expanded sequence input.')
table(['Family','Calculation and availability'],[
['NEWPRICE','From strictly prior five-minute log returns r: rv=sqrt(sum r^2), up_rv=sqrt(sum max(r,0)^2), down_rv=sqrt(sum min(r,0)^2), net=sum r, past_range=H-L, distance_from_high=H-C, distance_from_low=C-L, max_abs_return=max abs(r). H/L/C use the n+1 prior sampled close-price cumulative path. rv_ratio short/long=(RV_short/sqrt(short hours))/(RV_long/sqrt(long hours)), denominator floor 1e-6.'],
['LONG__ hourly','Sum exactly 12 finite 5-minute returns in [t-1h,t), label the completed hour at t. Complete h-hour windows required; half-hour forecast origins use the latest completed hour. N=sum R; V=sqrt(sum R^2); A=sum abs(R); C/H/L=current/local high/local low cumulative hourly level. Specific formulas follow.'],
['STATE / INTERACT','Existing masked market observations are sampled on the exact half-hour grid. Rolling windows [t-h,t) exclude the current origin, requiring ceil(.8*2*h) finite observations. Deviation=current previous snapshot x(t-30m) minus rolling mean. Interactions multiply specific historical state and price summaries. Exact sources appear in the feature formula dictionary.'],
['SEQk__feature','Exact historical lookup of the base feature at t-k hours; missing timestamps remain missing. 72h lags=18,12,6,0; 200h lags=60,40,20,0 where encoded by saved names; 300h lags=90,60,30,0. The full exact lag in each matrix name is authoritative.'],
['48h rolling transforms','Underlying x is reindexed on elapsed 30-minute grid; z(t)=x(t-30m). lag1=z. For w=7,14,30 observations: stdw=population std; rangew=max-min; iqrw=Q75-Q25; zscorew=(z-mean)/std, missing if std<=1e-12. ma_ratio7_30/14_30=short mean / 30-period mean, missing if denominator abs<=1e-8. Complete windows required. These are periods, not days.'],
['48h calendar','weekday_k=1 if UTC weekday k (Monday0); week_of_month_k=1 if floor((day-1)/7)+1=k; quarter_k=1 if UTC quarter k.'],
['48h LONG_','Prior five-minute endpoints [t-h,t): rv, up_rv, down_rv as above; abs_net=abs(sum r); abs_sum=sum abs(r); range_logbps=max cumulative endpoint level-min level. This range does not separately add the left boundary price.'],
['Inherited F4Z','Named masked canonical source inputs. Their upstream RZ/percentile/history transformations are inherited, not newly calculated by these models. Exact upstream normalization membership is not independently reconstructed here; no guessed formula is substituted.']],[140,WIDTH-140])
text('Tree prediction formula: fitted initial value plus shrinkage-weighted leaf updates. Saved sklearn tree contributions already contain learning-rate scaling; do not multiply them twice. GRU: r=sigmoid(Wr x+Ur h+b), z=sigmoid(Wz x+Uz h+b), n=tanh(Wn x+b+ r*(Un h+b)); h_new=(1-z)*n+z*h. The final hidden state feeds the saved output layer. Learned coefficients and preprocessing statistics require the saved model artifact; architecture alone is not numerical reproduction.')

definitions={}
for w in WS:
 d=ROOT/'Code/experiments'/w['run']
 for f in list(d.glob('*dictionary*.csv'))+list(d.glob('*definitions*.csv')):
  df=pd.read_csv(f).fillna('')
  if 'feature' in df and 'formula' in df:
   for r in df.to_dict('records'):definitions[r['feature']]=r['formula']+(' | sources: '+r['sources'] if r.get('sources') else '')
for h in ['200','300']:
 for f in (ROOT/f'Report/{h}h_report_support_2026-11-14').glob('*dictionary*.csv'):
  df=pd.read_csv(f).fillna('')
  for r in df.to_dict('records'):
   if 'feature' in r and 'formula' in r:definitions.setdefault(r['feature'],r['formula']+(' | sources: '+r['sources'] if r.get('sources') else ''))
used=sorted(set(re.sub(r'^SEQ\d+__','',f) for w in WS for f in w['features']))
def formula(f):
 if f in definitions:return definitions[f]
 if f.startswith('F4Z__'):return 'Inherited masked canonical source measurement. Exact family, metric and timeframe are encoded in the full name; upstream normalization not independently reconstructed.'
 if f.startswith('SHORTPRICE__'):return 'Same eight five-minute-return statistics as NEWPRICE; complete strictly prior 15m or 30m window (3 or 6 returns), with n+1 cumulative path points. Measure: '+f.split('__')[1]
 if f.startswith('NEWPRICE__'):return 'Five-minute price formula above: '+f.split('__')[1]+'; lookback '+f.split('__')[-1]
 if f.startswith('LONG_') and not f.startswith('LONG__'):return '48h five-minute LONG_ formula above; measure and horizon encoded in full name.'
 if re.match(r'^(lag1|std\d+|range\d+|iqr\d+|zscore\d+|ma_ratio\d+_\d+)__',f):return '48h rolling transform '+f.split('__')[0]+' applied to '+f.split('__',1)[1]+'. Formula and complete-window rule above.'
 if f.startswith(('weekday_','week_of_month_','quarter_')):return 'UTC calendar indicator, as defined above.'
 return 'See saved source definition; no inferred formula supplied.'
formula_rows=[{'feature':f,'formula':formula(f)} for f in used]
pd.DataFrame(formula_rows).to_csv(SUP/'used_base_feature_formulas.csv',index=False)
page('Exact base features | formula dictionary')
text('Sequence inputs reduce to the base name only in this formula dictionary; the opening matrix retains all exact SEQ prefixes. The same formula applies independently at every encoded snapshot. All names below are written in full.')
table(['Exact feature name','Formula / source'],[[r['feature'],r['formula']] for r in formula_rows],[390,WIDTH-390])
page('Audit, provenance and reproducibility')
text('Supporting files are in Report/combined_magnitude_support_2026-11-14. all_saved_full_coverage_forecasts.csv contains the complete MAE ranking, including aliases; winners_metrics.json contains full precision metrics; winner_models_and_features.json contains quarterly architectures, hyperparameters and exact feature membership; prediction_audit.json and source_hashes.json record SHA256 evidence; used_base_feature_formulas.csv contains the formula dictionary.')
text('Forecast rows must be finite and cover every retained origin for their horizon/target to enter this ranking. Partial per-quarter runs and mismatched conditional signed-return targets are excluded. Actual targets and ordered timestamps are matched across compared runs. Baselines are the saved training-only mean and median. Overlapping outcomes are dependent; no significance or profitability claim is made. The best historical MAE model may have worse RMSE than a baseline.')
text('The report reads saved fitted model settings, and verifies the selected artifact hashes where supplied. It does not repeat saved replay tests or reconstruct predictions from model weights. Each model card gives its original run path. Source data and historical model results are preserved.')
def footer(c,d):
 c.setFillColor(navy);c.setFont('Helvetica',8);c.drawString(32,18,'BTC magnitude research | 14 November 2026 | Saved-result MAE audit');c.drawRightString(W-32,18,str(d.page+1))
SimpleDocTemplate(str(TMP/'body.pdf'),pagesize=(W,H),leftMargin=32,rightMargin=32,topMargin=28,bottomMargin=34).build(story,onFirstPage=footer,onLaterPages=footer)
writer=PdfWriter();writer.append(TMP/'matrix.pdf');writer.append(TMP/'body.pdf');writer.add_metadata({'/Title':'BTC Combined Magnitude Models: MAE Winners, Features and Formulas','/Author':'Research report'});writer.write(OUT)
# Companion HTML makes the full matrix easy to scroll and search.
cells=[]
for s in ['UP','DOWN']:
 cells.append('<tr><th>'+s.lower()+'_magnitude_log_bps</th>'+''.join('<td><b>'+w['forecast']+'</b><p>MAE '+f"{w['MAE']:.6f}"+'</p><pre>'+'\n'.join(html.escape(f)+' ['+membership(w,f)+']' for f in w['features'])+'</pre></td>' for h in HS for w in WS if w['horizon']==h and side(w)==s)+'</tr>')
(ROOT/'Report/BTC_All_Horizons_Feature_Matrix_2026-11-14.html').write_text('<!doctype html><meta charset="utf-8"><title>Best MAE feature matrix</title><style>body{font-family:Arial;color:#17354a}table{border-collapse:collapse}td,th{border:1px solid #abc;padding:12px;vertical-align:top}th{background:#17354a;color:white;position:sticky;top:0}pre{font-size:12px;white-space:pre}td:nth-child(even){background:#eef4f7}</style><h1>Best observed MAE: full feature matrix</h1><p>Quarter membership [01,04,07,10]. See the combined PDF for status, metrics, architectures and formulas.</p><table><tr><th>Target</th>'+''.join('<th>'+h+'</th>' for h in HS)+'</tr>'+''.join(cells)+'</table>',encoding='utf-8')
reader=PdfReader(OUT);assert len(reader.pages)>20
mt=reader.pages[0].extract_text()
for w in WS:
 for f in w['features']:assert f in mt,f
print(json.dumps({'pdf':str(OUT),'pages':len(reader.pages),'base_features':len(used),'undefined_formulas':[r['feature'] for r in formula_rows if r['formula'].startswith('See saved')]}))

