"""Build a report from saved 1h results; no model fitting."""
from pathlib import Path
import json, hashlib
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Code/experiments/excursion_pipeline_1h_v1/20261113T101316741479Z'
SEARCH=RUN/'search/20261113T101328048744Z'
FOLLOW=RUN/'price_followup/20261113T104924262943Z'
RANK=FOLLOW/'feature_reliance/20261113T110708413275Z'
OUT=ROOT/'Report/output/pdf/ScaleWave_1h_Excursion_Models_Report.pdf'
SUP=ROOT/'Report/one_hour_excursion_report_sources'
SUP.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True)
metrics=pd.read_csv(FOLLOW/'metrics_all_variants.csv')
sels=json.loads((FOLLOW/'selections.json').read_text())
ranking=pd.read_csv(RANK/'feature_ranking.csv')
for directory in [SEARCH,FOLLOW]:
    done=json.loads((directory/'COMPLETE.json').read_text())
    assert hashlib.sha256((directory/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#25364a')))
styles.add(ParagraphStyle(name='Cell2',fontName='Helvetica',fontSize=8,leading=10))
styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
story=[]
def p(t,style='Body2'): return Paragraph(t,styles[style])
def title(n,t):
    if story: story.append(PageBreak())
    story.extend([p('SCALEWAVE / BTCUSDT / 1-HOUR EXCURSIONS','Small2'),p(f'{n:02d}  {t}','Title'),Spacer(1,12)])
def para(t): story.append(p(t))
def table(headers,rows,widths=None):
    data=[[p(escape(str(c)),'Cell2') for c in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#607b95')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')])]))
    story.extend([t,Spacer(1,10)])
def num(v): return f'{v:.4f}'
def side(t): return 'UP' if t.startswith('up_') else 'DOWN'
def label(s):
    return {'selected':'Validation-selected strategy','original':'Original F4Z model','training_mean':'Training mean baseline','training_median':'Training median baseline','all_price_plus_original':'Original + all 51 price inputs','price_only':'51 price inputs only','remove_original_features':'Remove original features'}.get(s,s.replace('add_','Add ').replace('price_','price ').replace('volatility_ratios','volatility ratios').replace('_',' '))

title(1,'Models and results')
para('Prepared 13 November 2026. This report covers the completed <b>1-hour UP/DOWN path-excursion experiment</b>. Both sides use separately fitted histogram gradient boosting models with absolute-error loss. The final strategy selects features using earlier validation periods.')
table(['Target','Initial MAE','Final MAE','Final RMSE','Final rank IC'],[[side(t),num(metrics[(metrics.target==t)&(metrics.period=='ALL')&(metrics.model=='original')].iloc[0].mae),num(g.mae),num(g.rmse),num(g.ic)] for t in ['up_magnitude_log_bps','down_magnitude_log_bps'] for _,g in metrics[(metrics.target==t)&(metrics.period=='ALL')&(metrics.model=='selected')].iterrows()],[60,110,110,110,110])
para('<b>What improved:</b> price features lowered pooled MAE from 23.5440 to 23.2177 for UP and from 24.4890 to 24.2814 for DOWN. UP RMSE slightly increased from 39.4531 to 39.5435; DOWN RMSE fell from 44.5981 to 44.3502.')
para('All errors are <b>log basis points</b>. MAE is average absolute error; RMSE puts more weight on large errors; rank IC is Spearman correlation (higher is better). The pooled scores combine four quarterly fitted models, not just the latest model.')
table(['Scope','Definition'],[['Asset and horizon','BTCUSDT; next 60 minutes'],['Initial / evaluation origins','29,615 / 18,481 original timestamps'],['Evaluation','January 2024 through 22 January 2025'],['Final model family','HistGradientBoostingRegressor; separate UP and DOWN'],['Latest model inputs','UP: 55 raw features; DOWN: 3 raw features'],['Status','Both stages complete; saved prediction hashes verified; saved-model replay passed in run records']],[145,355])
para('Historical evaluation has been examined repeatedly. Rankings describe this saved experiment; they are not independent confirmation. Earlier absolute endpoint-magnitude 1h experiments have a different target and are excluded from these rankings.')

title(2,'Targets and training design')
para('<b>UP:</b> 10,000 x ln(highest future reference-path price / origin price).<br/><b>DOWN:</b> 10,000 x ln(origin price / lowest future reference-path price).')
para('The canonical one-minute reference-price path includes the origin and ends one hour later. Both targets are nonnegative. Every valid observation, including zeros, enters both models; there is no endpoint-direction filter or direction-probability combination.')
table(['Step','Method'],[['Quarterly fitting','1 Jan, 1 Apr, 1 Jul and 1 Oct 2024; expanding history. October model retained for January 2025.'],['Maturity','Training labels must end strictly before the fit or validation boundary; one-hour purge. Original 48h split gap remains absent.'],['Inner validation','Three earlier three-month windows starting 9, 6 and 3 months before each fit. Validation labels also mature before their window end.'],['Stage 1 candidates','522 masked F4Z value columns; no mandatory legacy features.'],['Quality and shortlist','At least 80% finite coverage, 1,000 finite readings and variation in every inner training set. Depth-2 MAE tree screening, up to 4,096 evenly spaced training rows. Earliest-training absolute Spearman redundancy threshold 0.90; shortlist up to 16.'],['Stage 1 selection','Independent Ridge and MAE boosting greedy additions up to 6 features; removal checks. Addition needs at least 0.1% pooled validation MAE gain and wins in 2 of 3 windows; median fallback.'],['Stage 2 selection','Original features plus up to two greedy price-block additions; also all-price + original, price-only and group removals. Selected alternative must beat both original and median by the validation gate; otherwise original is retained.']],[130,370])
para('Missing inputs do not remove evaluation observations. Availability masking and provisional source limitations are inherited from the supplied X package. Adaptive reuse of validation windows is disclosed in the run contracts.')

title(3,'Architecture and hyperparameters')
para('<b>Prediction path:</b> raw feature vector -&gt; training-median imputation + missing indicators -&gt; training-only centering/scaling -&gt; additive boosted trees -&gt; clip prediction at zero.')
para('For every raw input, one missing-value indicator is appended. The model therefore receives twice the raw feature count: the latest UP model has 110 transformed inputs and DOWN has 6. Each boosting iteration adds a small tree to the running prediction. This is a tree ensemble, not a neural network or a Ridge/tree blend.')
table(['Setting','Value'],[['Estimator','sklearn.ensemble.HistGradientBoostingRegressor'],['Loss / objective','absolute_error; median-oriented regression'],['Boosting iterations','80'],['Learning rate','0.05'],['Maximum depth / leaves','3 / 7'],['Minimum samples per leaf','128'],['L2 regularization','1.0'],['Early stopping','False; no random validation split'],['Random seed','48'],['Execution thread limit','2'],['Preprocessing','Training medians; missing indicators; training mean and standard deviation. Near-zero scales set to 1.'],['Output rule','Nonnegative predictions via max(0, prediction)']],[185,315])
para('<b>Other search candidates:</b> Ridge uses alpha = 0.01 x training row count (squared-error fit, selected by MAE). Screening uses a DecisionTreeRegressor with absolute_error, max_depth=2, min_samples_leaf=128, random_state=48. Mean and median constants are computed from eligible training labels. All eight final selected quarterly side models are boosting trees.')

title(4,'How the 51 new features were made')
para('Source: checksum-verified canonical five-minute BTC log returns. Let r be a five-minute log return and L its cumulative log-price level. Windows use return endpoints <b>strictly before</b> forecast time. These inputs summarize the past; the target still looks one hour ahead.')
table(['Eight measures per lookback','Calculation'],[['Realized volatility (rv)','sqrt(sum(r squared))'],['Upward volatility (up_rv)','sqrt(sum(max(r, 0) squared))'],['Downward volatility (down_rv)','sqrt(sum(min(r, 0) squared))'],['Net return (net)','sum(r)'],['Past range (past_range)','max(L) - min(L) over the reconstructed close path'],['Distance from high','max(L) - latest past L'],['Distance from low','latest past L - min(L)'],['Largest absolute 5m return','max(abs(r))']],[205,295])
para('Each measure is calculated at <b>1h, 4h, 12h, 24h, 48h and 168h</b>: 8 x 6 = 48 inputs. The range and distance measures use five-minute close-path levels, not intrabar exchange highs/lows.')
para('Three additional duration-normalized volatility ratios use pairs <b>1h/24h, 4h/24h and 24h/168h</b>: (RV_short / sqrt(short hours)) / max(RV_long / sqrt(long hours), 0.000001). Total = <b>51</b>.')
para('A complete h-hour window requires h x 12 returns. Incomplete windows stay missing and are handled by training-fitted preprocessing. Prefix and future-perturbation checks verify that changing future returns cannot change an origin\'s features.')
para('<b>Exact naming:</b> NEWPRICE__{measure}__{h}h, using rv, up_rv, down_rv, net, past_range, distance_from_high, distance_from_low and max_abs_return. Ratios: NEWPRICE__rv_ratio__1h_24h, NEWPRICE__rv_ratio__4h_24h and NEWPRICE__rv_ratio__24h_168h.')

for n,target in [(5,'up_magnitude_log_bps'),(6,'down_magnitude_log_bps')]:
    title(n,f'{side(target)} model ranking')
    m=metrics[(metrics.target==target)&(metrics.period=='ALL')&metrics.full_period_coverage.astype(bool)&metrics.rows.eq(18481)].copy()
    m['MAE rank']=m.mae.rank(method='min').astype(int);m['RMSE rank']=m.rmse.rank(method='min').astype(int);m['IC rank']=m.ic.rank(ascending=False,method='min').astype(int)
    m=m.sort_values('mae');m.to_csv(SUP/f'{side(target)}_model_ranking.csv',index=False)
    para('Ranked by pooled MAE on the same 18,481 origins. RMSE and IC have their own ranks. Rows are comparable model/feature strategies; quarterly selected feature sets can differ within a strategy.')
    table(['MAE rank','Model / strategy','MAE','RMSE (rank)','IC (rank)'],[[r['MAE rank'],label(r.model),num(r.mae),f'{r.rmse:.3f} ({r["RMSE rank"]})',f'{r.ic:.3f} ({r["IC rank"]})'] for _,r in m.iterrows()],[45,220,65,85,85])
    para('<b>Interpretation:</b> Validation-selected strategy is the earlier-data decision rule. The lowest outer MAE among all diagnostic variants is a hindsight ranking, not a new validated winner. Partial-coverage combinations are excluded from this table; their saved rows are exported separately.')
metrics[(metrics.period=='ALL')&~metrics.full_period_coverage.astype(bool)].to_csv(SUP/'partial_coverage_variants.csv',index=False)

title(7,'Metrics by evaluation period')
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    para(f'<b>{side(target)} - final validation-selected strategy</b>')
    m=metrics[(metrics.target==target)&(metrics.model=='selected')]
    table(['Period','Rows','MAE','RMSE','IC','MAE gain vs initial'],[[r.period,r.rows,num(r.mae),num(r.rmse),num(r.ic),num(r.mae_gain_vs_original)] for _,r in m.iterrows()],[70,50,85,85,85,125])
para('Positive gain means less error than the original F4Z model. UP MAE improves in all four quarters but worsens in January 2025. DOWN improves in Q1-Q3 and retains original predictions in Q4 and January. RMSE improvement is not uniform across periods.')

# Compact dictionary for exact original inputs; membership table uses IDs.
original=sorted({c for s in sels for c in s['features'] if c.startswith('F4Z__')})
ids={c:f'F{i+1:02d}' for i,c in enumerate(original)}
title(8,'Exact original feature dictionary')
para('The IDs below abbreviate exact saved column names in the cycle-membership table. RZ denotes the source robust score; percentile and historical active rate remain distinct source measurements.')
table(['ID','Exact original column'],[[ids[c],c] for c in original],[40,460])

title(9,'Features used in each fitted model')
rows=[]
for s in sels:
    orig=[ids[c] for c in s['features'] if c in ids]
    new=[c for c in s['features'] if c.startswith('NEWPRICE')]
    block='All 51 price inputs' if len(new)==51 else ('Eight measures at 4h and 48h (16)' if len(new)==16 else 'None')
    rows.append([s['issue'][:7],side(s['target']),len(s['features']),', '.join(orig),block])
table(['Fit','Side','Count','Original feature IDs','Added price features'],rows,[65,40,40,180,175])
para('All rows use MAE histogram gradient boosting. January 2025 reuses the October model. The October DOWN model keeps its three original features because the earlier validation gate did not select a price-feature alternative.')
para('The 55 October UP inputs are F IDs listed above plus all 51 price features defined on page 4. The October DOWN inputs are only its three listed F IDs. Raw feature counts exclude the automatically appended missing indicators.')
para('The complete machine-readable selected_features.csv and selections.json remain in the source run. Copies of the exact selections and all feature rankings accompany the report source directory.')
pd.read_csv(FOLLOW/'selected_features.csv').to_csv(SUP/'selected_features.csv',index=False)
ranking.to_csv(SUP/'feature_ranking.csv',index=False)

title(10,'Feature ranking: latest models')
para('October models on 5,425 origins, 1 October 2024 to 22 January 2025. Each feature is shuffled in 48-row blocks, five fixed-seed repetitions, including missingness. Score = average increase in MAE (log bps); larger values show greater model reliance. No retraining.')
def short(c): return c.replace('F4Z__','').replace('NEWPRICE__','').replace('__value','').replace('__',' / ')
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    para(f'<b>{side(target)} - '+('top 10 of 55' if target.startswith('up') else 'all 3')+'</b>')
    f=ranking[(ranking.target==target)&(ranking.kind=='individual')].sort_values('rank').head(10)
    table(['Rank','Feature','MAE increase'],[[r['rank'],short(r.feature),f'{r.mean_mae_increase:.5f}'] for _,r in f.iterrows()],[40,370,90])
para('Correlated inputs can substitute for each other; individual contributions are not additive. Negative scores do not automatically justify removal. Shuffle variation is not a confidence interval. This ranking covers the latest models, not all four cycles.')

title(11,'Evidence and reproducibility')
para('Run root: Code/experiments/excursion_pipeline_1h_v1/20261113T101316741479Z')
table(['Evidence','Relative location within run root'],[['Initial search','search/20261113T101328048744Z'],['Price-feature follow-up','price_followup/20261113T104924262943Z'],['Feature reliance','price_followup/20261113T104924262943Z/feature_reliance/20261113T110708413275Z']],[140,360])
para('Metrics: follow-up metrics_all_variants.csv and initial metrics.csv. Feature membership: selections.json and selected_features.csv. Architecture: saved discover_excursion_features_1h.py and improve_excursion_features_1h.py snapshots. Settings, dependencies and input identities: CONTRACT.json. Model paths/hashes: per-target selection.json. Ranking method and replay: feature_reliance/AUDIT.json.')
para('The report builder checked both prediction-file SHA256 values against their COMPLETE.json records. Scores in this report are read from saved metric tables rather than independently recomputed. Feature reliance was separately executed from saved models and checked against original predictions and actual targets. No training was performed to prepare this report.')
para('The results concern canonical reference-path excursions on sampled half-hour forecast origins. They do not establish endpoint-direction accuracy, trading profitability, transaction-cost performance or statistical significance. Overlapping horizons, correlated predictors and repeatedly exposed history affect interpretation.')
para('Model ranking includes only full-coverage variants of this 1h excursion experiment. Earlier one-hour absolute endpoint-return models and other horizons are not comparable targets. Partial-coverage variants are provided separately without a pooled rank.')
manifest={}
for f in [SEARCH/'CONTRACT.json',SEARCH/'metrics.csv',FOLLOW/'CONTRACT.json',FOLLOW/'metrics_all_variants.csv',FOLLOW/'selections.json',FOLLOW/'discover_excursion_features_1h.py',FOLLOW/'improve_excursion_features_1h.py',RANK/'feature_ranking.csv',RANK/'AUDIT.json']:
    manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(SUP/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607b95'))
    canvas.drawString(42,24,'ScaleWave | 1-hour UP/DOWN excursions | 13 Nov 2026')
    canvas.drawRightString(553,24,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
