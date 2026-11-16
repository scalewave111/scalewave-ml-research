"""Completed three-stage 15m report. Reads saved artifacts; no fitting."""
from pathlib import Path
import json, hashlib, shutil, zipfile
from xml.sax.saxutils import escape
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'Code/experiments/excursion_pipeline_15m_v1/20261113T104418654180Z'
A=P/'search/20261113T104426239026Z'
B=P/'price_followup/20261113T110533531198Z'
C=ROOT/'Code/experiments/excursion_tuning_15m_v1/20261113T113145289092Z'
OUT=ROOT/'Report/output/pdf'
SUP=ROOT/'Report/excursion_15m_complete_report_2026-11-13'
OUT.mkdir(parents=True,exist_ok=True); SUP.mkdir(parents=True,exist_ok=True)
sources={}
def track(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(track(p))
W,H=landscape(A4); WIDTH=W-76
navy=colors.HexColor('#17354A'); teal=colors.HexColor('#087E86')
styles={name:ParagraphStyle(name,fontName=font,fontSize=size,leading=lead,textColor=col,spaceAfter=after)
        for name,font,size,lead,col,after in [('title','Helvetica-Bold',23,28,navy,12),
        ('head','Helvetica-Bold',13,17,teal,9),('body','Helvetica',10,14,navy,9),
        ('small','Helvetica',8.5,11,navy,6),('cell','Helvetica',8.2,10.7,navy,0),
        ('th','Helvetica-Bold',8.2,10.7,colors.white,0),('code','Courier',7.3,10,navy,0)]}
story=[]
def para(text,style='body'):return Paragraph(escape(str(text)),styles[style])
def txt(text,style='body'):story.append(para(text,style))
def page(title,subtitle=None):
    if story:story.append(PageBreak())
    txt(title,'title')
    if subtitle:txt(subtitle,'small')
def table(headers,rows,widths=None,codecol=None):
    data=[[para(v,'th') for v in headers]]
    for row in rows:
        data.append([para(v,'code' if codecol==i else 'cell') for i,v in enumerate(row)])
    t=Table(data,colWidths=widths or [WIDTH/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F7'),colors.white]),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),.7,teal)]))
    story.append(t);story.append(Spacer(1,8))
def fmt(v):return '--' if pd.isna(v) else f'{v:.3f}'
def side(t):return 'UP' if t.startswith('up_') else 'DOWN'

data={}; contracts={}; cards={}; metrics={}
for label,run,metricfile in [('A',A,'metrics.csv'),('B',B,'metrics_all_variants.csv'),('C',C,'metrics.csv')]:
    done=js(run/'COMPLETE.json')
    assert done['status']=='COMPLETE'
    assert hashlib.sha256((run/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']
    f=csv(run/'predictions.csv').sort_values(['target','decision_time']).reset_index(drop=True)
    assert len(f)==36962 and not f.duplicated(['target','decision_time']).any()
    data[label]=f
    contracts[label]=js(run/'CONTRACT.json');cards[label]=js(run/'selections.json')
    metrics[label]=csv(run/metricfile)
    for filename in ['CONTRACT.json','selections.json','COMPLETE.json']:
        shutil.copyfile(run/filename,SUP/f'{label}_{filename}')
    metrics[label].to_csv(SUP/f'{label}_all_metrics.csv',index=False)
for label in ['B','C']:
    assert data[label][['target','decision_time','issue']].equals(data['A'][['target','decision_time','issue']])
    np.testing.assert_allclose(data[label].actual,data['A'].actual,rtol=0,atol=1e-10)
np.testing.assert_allclose(data['A'].prediction,data['B'].original,rtol=0,atol=1e-7)
np.testing.assert_allclose(data['B'].selected,data['C'].current,rtol=0,atol=1e-7)
np.testing.assert_allclose(data['B'].all_price_plus_original,data['C'].previous_all_price,rtol=0,atol=1e-7)
for old,new in [('selected','current_features__unchanged'),('all_price_plus_original','all_price__unchanged')]:
    np.testing.assert_allclose(data['B'][old],data['C'][new],rtol=0,atol=1e-7)
for filename in ['tune_excursions_15m.py','discover_excursion_features_15m.py','improve_excursion_features_15m.py','canonical_return_features.py']:
    track(C/filename)
trials=[];feature_rows=[]
for label,run in [('A',A),('B',B),('C',C)]:
    for p in sorted(run.glob('*/validation_trials.csv')):
        f=csv(p);f.insert(0,'experiment',label);f.insert(1,'fit_target',p.parent.name);trials.append(f)
    for p in sorted(run.glob('*/selection.json')):
        card=js(p)
        specs=card.get('variants',card.get('specs',{'original':card}))
        for name,spec in specs.items():
            for i,col in enumerate(spec['features']):
                feature_rows.append(dict(experiment=label,fit_target=p.parent.name,variant=name,position=i+1,feature=col))
pd.concat(trials,ignore_index=True).to_csv(SUP/'all_validation_trials.csv',index=False)
pd.DataFrame(feature_rows).to_csv(SUP/'exact_features_all_variants.csv',index=False)

# Consolidate repeated controls once, preserving their aliases and verification.
b=metrics['B'].copy();b['name']='B/'+b.model
c=metrics['C'].copy()
aliases={'current':'B/selected','previous_all_price':'B/all_price_plus_original',
    'training_mean':'B/training_mean','training_median':'B/training_median',
    'current_features__unchanged':'B/selected','all_price__unchanged':'B/all_price_plus_original'}
c=c[~c.model.isin(aliases)].copy();c['name']='C/'+c.model
c['full_period_coverage']=True
all_scores=pd.concat([b,c],ignore_index=True)
def ranks(f):
    f=f.copy()
    for m,asc in [('mae',True),('rmse',True),('ic',False)]:
        f[m+'_rank']=f[m].round(10).rank(method='min',ascending=asc)
    return f.sort_values(['mae_rank','name'])
pooled=pd.concat([ranks(g) for _,g in all_scores[all_scores.period.eq('ALL')&all_scores.full_period_coverage.eq(True)].groupby('target')])
periods=pd.concat([ranks(g) for _,g in all_scores[~all_scores.period.eq('ALL')].groupby(['target','period'])])
assert pooled.rows.eq(18481).all()
pooled.to_csv(SUP/'ranked_full_period_models.csv',index=False)
periods.to_csv(SUP/'ranked_all_models_by_period.csv',index=False)
pd.DataFrame([dict(alias='C/'+a,canonical=b) for a,b in aliases.items()]+[dict(alias='A/prediction',canonical='B/original')]).to_csv(SUP/'duplicate_control_aliases.csv',index=False)
configs=contracts['C']['configs']
config_ids={name:f'H{i}' for i,name in enumerate(configs)}
pack_ids={'current_features':'S','all_price':'P','all_price_short':'Q'}
def display(name):
    stage,model=name.split('/',1)
    if stage=='C' and model!='selected':
        pack,config=model.split('__');return f'C: {pack_ids[pack]} / {config_ids[config]}'
    if name=='C/selected':return 'C: tuned selection policy'
    if name=='B/selected':return 'B: price-feature selection policy'
    return 'B: '+model
def rank_table(f):
    table(['MAE rank','Model / strategy','MAE','RMSE','RMSE rank','IC','IC rank'],
      [[int(r.mae_rank),display(r.name),f'{r.mae:.6f}',f'{r.rmse:.6f}',int(r.rmse_rank),f'{r.ic:.6f}',int(r.ic_rank)] for r in f.itertuples()],
      [48,WIDTH-398,80,80,60,75,55])

page('BTC | 15-minute excursion models','Complete results through the tuning run | 13 November 2026')
txt('Separate UP and DOWN models forecast the largest excursion above and below the starting price during the next 15 minutes. This report covers all three completed stages: original feature search (A), price-feature follow-up (B), and model/short-price tuning (C).')
rows=[]
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    for stage,name,label in [('B','training_median','Median baseline'),('B','original','A: original selected model'),('B','selected','B: price-feature selection'),('C','selected','C: tuned selection')]:
        r=metrics[stage].query('target==@target and period=="ALL" and model==@name').iloc[0]
        rows.append([side(target),label,fmt(r.mae),fmt(r.rmse),fmt(r.ic)])
table(['Target','Strategy','MAE','RMSE','Spearman IC'],rows,[60,WIDTH-345,95,95,95])
txt('Tuned selection reduces MAE by 0.22% for UP and 0.44% for DOWN relative to B. Relative to the training median, the reductions are 7.40% and 6.79%. Errors are log bps; these percentages are error reductions, not price returns.')
txt('Navigation: model families and architecture first; metric rules and full-period rankings next; then hyperparameters, feature construction and exact lists, quarterly model maps, validation summaries, partial-coverage comparisons and provenance. Full-precision scores and all per-period rankings are in the supporting ZIP.','small')

page('Model families and architecture')
table(['Family / strategy','Architecture','Role'],[
['Histogram Gradient Boosting (HGB)','Median initial prediction plus sequential small regression-tree corrections. Absolute-error loss; separate UP/DOWN fits.','All selected A models, nonconstant B variants, and all 24 direct C candidates.'],
['Tuned selection / blend','prediction = (1-w)*previous selected prediction + w*candidate prediction; w in {0,0.5,1}.','C selects candidate and weight on earlier validation; w=0 preserves B. A blend is not guaranteed to be the conditional median.'],
['Ridge','Intercept plus weighted standardized features, L2-regularized squared-error regression.','A earlier-validation challenger only; no saved comparable later-period Ridge predictions.'],
['Screening decision tree','One depth-2 absolute-error tree; single-feature fits on at most 4,096 evenly spaced training rows.','A screening only; earlier validation scores, not later-period ranking.'],
['Mean / median controls','One training-only constant per target and quarterly fit.','Evaluated across the same later observations.']],[160,365,WIDTH-525])
txt('Preprocessing: raw features -> training medians fill missing values -> append one missing indicator per raw feature -> training-only centering and scaling -> model -> clip negative predictions to zero. A model with d raw features receives 2d transformed inputs. No neural network, direction probability or endpoint-direction filter is used.')

page('Metrics, evaluation and ranking rules')
table(['Metric','Formula','Interpretation'],[
['MAE','mean(abs(actual - prediction))','Primary training/selection/ranking criterion; lower is better; log bps.'],
['RMSE','sqrt(mean((actual - prediction)^2))','Lower is better; more sensitive to large misses; log bps.'],
['Spearman IC','Correlation of actual and prediction ranks','Higher is better; ordering rather than magnitude calibration.'],
['Matched MAE gain','Comparison-model MAE minus candidate MAE on identical rows','Positive indicates improvement. Required when a candidate covers only some quarters.']],[125,300,WIDTH-425])
txt('Full-period ranks use the same 18,481 timestamps per target and compare all distinct named strategies with complete coverage. A/prediction is B/original. C/current and C/current_features__unchanged reproduce B/selected; C/previous_all_price and C/all_price__unchanged reproduce B/all_price_plus_original. Repeated controls are listed once, with all aliases in the ZIP.')
txt('MAE is the primary rank; RMSE and IC have separate ranks. Numerical ties at 10 decimal places share the minimum rank. Different named strategies with equal scores are retained. Partial-coverage variants have no full-period rank; all available variants are ranked within each common quarter/January in the accompanying CSV.')
txt('Ridge and screening-tree scores come from earlier validation and are kept separate. Observed ranking is descriptive: choosing the best diagnostic model after viewing these outcomes would be an additional selection step on exposed history.')

page('How to read the tuning model names')
table(['Code','Feature pack'],[['S','Current selected B features for that target and quarterly refit.'],['P','Original A features plus all admitted 51 long-lookback price features.'],['Q','P plus admitted 16 new short-lookback price features at 15 and 30 minutes.']],[65,WIDTH-65])
table(['Code','Configuration','Depth','Leaves','Min leaf','Trees','Rate','L2'],
 [[config_ids[k],k,v['max_depth'],v['max_leaf_nodes'],v['min_samples_leaf'],v['max_iter'],v['learning_rate'],v['l2_regularization']] for k,v in configs.items()],
 [45,WIDTH-405,50,50,60,60,65,75])
txt('Example: C: Q / H6 means HGB using the all-price-plus-short pack and deeper_regularized settings. All configurations use absolute_error, seed48, early_stopping=False, max_bins=255 and full feature sampling. Fixed iteration budgets are compared using earlier validation; there is no random early-stopping split. H0 is the original A/B boosting configuration.')
txt('B variants use H0: add_X adds X to original A features; price_only uses the long-price inputs alone; remove_X removes X from the greedy-search endpoint. An empty remaining set predicts the training median. C direct rows use weight1; the tuned selection policy may blend at weight0.5.','small')

for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    f=pooled[pooled.target.eq(target)]
    for start in range(0,len(f),19):
        page(f'{side(target)} | all full-period models ranked',f'{start+1}-{min(start+19,len(f))} of {len(f)} strategies / baselines | same 18,481 origins | primary rank: MAE')
        rank_table(f.iloc[start:start+19])

page('Targets and chronological design')
table(['Item','Definition'],[
['UP','10000 * ln(Pmax / P0), canonical up_magnitude_log_bps'],
['DOWN','10000 * ln(P0 / Pmin), canonical down_magnitude_log_bps'],
['Price path','16 one-minute reference-price observations including origin through minute15. Both excursions can be positive. Zeros are retained.'],
['Training population','29,615 supplied initial origins, 22 Apr2022 00:30 through 29 Dec2023 23:30 UTC.'],
['Evaluation population','18,481 supplied origins from Jan2024 through Jan2025; origins sampled every30minutes.'],
['Walk-forward fits','Jan/Apr/Jul/Oct2024 expanding fits; Jan2025 uses October and is scored separately.'],
['Purging','Only training labels with origin+15minutes strictly before fit boundary. Inner validation labels must also mature strictly before their window end.'],
['Validation','Three consecutive earlier quarterly windows for each fit. Training-only quality checks and preprocessing.'],
['Inherited gap','Original48h package boundary gap preserved; no missing origins synthesized.']],[150,WIDTH-150])
txt('Previously examined historical evaluation, reused validation windows and inherited feature availability/provisional-source limits remain. This is not an independently untouched evaluation.')

page('Selection methods and full hyperparameters')
table(['Stage','Procedure'],[
['A: 522 F4Z inputs','Each inner fit: >=80% finite coverage, >=1000 readings, variation. Rank single features by depth2 median-tree MAE; remove |Spearman|>=.90 redundancy using earliest training rows. Keep up to16; greedy forward up to6 features separately for Ridge and HGB; refit removal checks.'],
['A: models / gate','Ridge alpha=.01*n, squared-error fit selected by MAE. Screening tree minleaf128, seed48. HGB H0. Additions require >=.1% pooled MAE gain and2/3 earlier-window wins. Otherwise median fallback. All8 final A fits selected HGB.'],
['B: price-feature search','Same H0. Greedy additions up to2 price groups; all-price+original, price-only and group-removal controls. Candidate must beat original AND median by >=.1% pooled MAE with2/3 window wins. Otherwise original.'],
['C: model/feature tuning','Three independent packs x8 configurations =24 direct candidates. Evaluate blends w=.5 and1 against B selected; require >=.1% pooled MAE gain and2/3 window wins versus BOTH B selected and median. Choose lowest eligible pooled earlier MAE; otherwise w=0.'],
['Common details','HGB loss absolute_error; max_bins255; max_features1; no monotonic/interaction constraints; seed48; no early stopping. Imputation median defaults0 for all-missing columns; scales below1e-8 become1. Predictors are numerical; no categorical inputs.']],[155,WIDTH-155])
txt('More flexible candidates do not automatically replace the baseline. Hyperparameter and feature decisions are frozen before each later-period evaluation. This adaptive search reuses earlier validation and historically selected feature lists; it is not fully nested confirmation.')

page('How all 67 new price features are created')
txt('Source: canonical five-minute BTC log returns indexed at their END timestamps. At forecast time T, every input return endpoint is strictly before T. Long windows are 1h,4h,12h,24h,48h,168h (12*h returns); new short windows are15m and30m (3 and6 returns). A missing return makes the affected complete-window statistics missing.')
table(['Measurement','Formula','Meaning'],[
['rv','sqrt(sum(r_i^2))','Realized volatility in log bps.'],['up_rv','sqrt(sum(max(r_i,0)^2))','Upward semivolatility.'],['down_rv','sqrt(sum(min(r_i,0)^2))','Downward semivolatility, nonnegative.'],['net','sum(r_i)','Historical signed log return.'],['past_range','max(L)-min(L)','Range of reconstructed log-price path, including window start.'],['distance_from_high','max(L)-L_latest','Distance below the past high.'],['distance_from_low','L_latest-min(L)','Distance above the past low.'],['max_abs_return','max(abs(r_i))','Largest individual five-minute move.'],['Volatility ratios','(RV_a/sqrt(a))/max(RV_b/sqrt(b),1e-6)','a/b =1h/24h,4h/24h,24h/168h; dimensionless.']],[140,290,WIDTH-430])
txt('Eight measurements x6 long windows +3 ratios =51 B features. Eight measurements x2 short windows =16 C features. L is the cumulative sum of the window returns, including starting level0. Range features use five-minute closes, not intrabar extremes. New short columns must pass training-only quality checks.')

page('Feature construction | worked short-window example')
txt('At forecast T, suppose the last three eligible five-minute returns are +10,-5,+20 log bps. Their endpoints are T-15,T-10,T-5 minutes; the return ending exactly at T is excluded.')
table(['Quantity','Calculation','Result'],[
['Relative path','0 ->10 ->5 ->25','Four levels, including start'],['rv','sqrt(10^2+(-5)^2+20^2)','22.913 log bps'],['up_rv','sqrt(10^2+20^2)','22.361 log bps'],['down_rv','sqrt((-5)^2)','5.000 log bps'],['net','10-5+20','25 log bps'],['past_range','25-0','25 log bps'],['distance_from_high','25-25','0 log bps'],['distance_from_low','25-0','25 log bps'],['max_abs_return','max(10,5,20)','20 log bps']],[170,330,WIDTH-500])
txt('The same calculations apply to30minutes using six returns and to every long lookback. For ratios, RV_1h=30 and RV_24h=100 gives (30/sqrt(1))/(100/sqrt(24))=1.470. This measures relative past volatility; it is not a future target.')

f4z=sorted({r['feature'] for r in feature_rows if r['feature'].startswith('F4Z__')})
ids={c:f'F{i+1:02d}' for i,c in enumerate(f4z)}
page('Exact original F4Z feature names','Feature IDs are used only to make the quarterly maps readable; exact names are preserved here and in the CSV bundle.')
table(['ID','Exact name'],[[ids[c],c] for c in f4z],[45,WIDTH-45],codecol=1)
long_names=[c for cols in contracts['B']['groups'].values() for c in cols]
short_names=sorted({r['feature'] for r in feature_rows if r['feature'].startswith('SHORTPRICE__')})
assert len(long_names)==51 and len(short_names)==16
newnames=long_names+short_names
for start in range(0,len(newnames),20):
    page('Exact derived feature names',f'{start+1}-{min(start+20,len(newnames))} of67 | NEWPRICE = B long windows; SHORTPRICE = C short windows')
    table(['No.','Exact name'],[[i+1,c] for i,c in enumerate(newnames) if start<=i<start+20],[45,WIDTH-45],codecol=1)

page('A and B | features at each quarterly refit')
table(['Fit','Target','A original features','B selected addition','B raw count'],
 [[r['issue'][:7],side(r['target']),', '.join(ids[c] for c in r['features'] if c in ids),r['variant'],len(r['features'])] for r in cards['B']],
 [65,50,205,WIDTH-390,70])
txt('add_price_h uses all eight statistics for that lookback. all_price_plus_original adds all51. volatility_ratios adds three. original means no added group. In these runs, B retains each A selected original feature set. Exact feature membership for every diagnostic variant, including removals, is in exact_features_all_variants.csv.')

page('C | exact feature packs and selected models')
table(['Fit','Target','S count','P count','Q count','Selected candidate','Weight'],
 [[r['issue'][:7],side(r['target']),len(r['packs']['current_features']),len(r['packs']['all_price']),len(r['packs']['all_price_short']),display('C/'+r['candidate']),r['weight']] for r in cards['C']],
 [65,50,65,65,65,WIDTH-435,125])
txt('S is precisely the B selected feature list on the previous page. P is that quarter\'s A F4Z list plus all51 NEWPRICE columns. Q is P plus all16 SHORTPRICE columns. All16 passed admission in every fit. Every H configuration within a pack uses the same ordered feature list.')
assert all(len(r['packs']['all_price_short'])-len(r['packs']['all_price'])==16 for r in cards['C'])
txt('All eight selections chose H6 (deeper_regularized): depth4, leaves12, minleaf128,160 iterations, rate.05,L2=10. Weight.5 means two independent models are averaged: B selected and the listed candidate. Weight1 uses only the candidate. Raw counts exclude missing indicators, which double each model\'s input dimension.')
txt('October UP uses Q:6 F4Z +51 long +16 short =73 raw /146 transformed inputs. October DOWN blends B\'s3 raw /6 transformed inputs with Q:3+51+16=70 raw /140 transformed inputs. October models remain in use for January2025.')

page('Tuned selection | quarterly error changes')
f=metrics['C'][metrics['C'].model.eq('selected') & ~metrics['C'].period.eq('ALL')]
table(['Target','Period','MAE','RMSE','IC','MAE gain vs B','RMSE gain vs B'],
 [[side(r.target),r.period,fmt(r.mae),fmt(r.rmse),fmt(r.ic),f'{r.mae_gain_vs_current:+.4f}',f'{r.rmse_gain_vs_current:+.4f}'] for r in f.itertuples()],
 [55,80,100,100,90,140,WIDTH-565])
txt('Positive gains mean lower error. DOWN MAE improves in all four quarters and January. UP MAE improves except Q4 (+0.0080 log-bps deterioration). DOWN Q4 RMSE worsens slightly even though its MAE improves. Pooled gains therefore do not imply improvement on every metric in every period.')

for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    partial=b[b.target.eq(target)&b.period.eq('ALL')&~b.full_period_coverage.eq(True)]
    page(f'{side(target)} | partial-coverage B variants','These cannot be assigned a comparable full-period rank. Their per-period MAE ranks below include C candidates on the same period.')
    rr=[]
    for row in partial.itertuples():
        pr=periods[periods.target.eq(target)&periods.name.eq(row.name)]
        label=', '.join(f'{r.period}: {int(r.mae_rank)}' for r in pr.sort_values('period').itertuples())
        rr.append([row.model,int(row.rows),fmt(row.mae),fmt(row.mae_gain_vs_original),label])
    table(['Variant','Rows','Pooled MAE','Matched gain vs A','MAE rank by period'],rr,[260,50,75,85,WIDTH-470])
    txt('A positive matched gain compares this variant with B/original (the A model) on exactly its available rows. Lower pooled MAE across different row populations is not a model ranking. Complete period MAE/RMSE/IC rankings are supplied in ranked_all_models_by_period.csv.','small')

for start in [0,12]:
    v=pd.concat(trials,ignore_index=True)
    rows=[]
    for (fit,model),g in v[v.experiment.eq('A')].groupby(['fit_target','model']):
        g=g[g.features.ne('[]')]
        if len(g):rows.append([fit[:7],'UP' if '_up_' in fit else 'DOWN',model,len(g),fmt(g.pooled_mae.min())])
    page('Earlier-validation model-family comparison','A trials only; minimum tried MAE is not an outer score or an independently validated selected model.')
    table(['Fit','Target','Family','Trials','Minimum trial MAE'],rows[start:start+12],[90,65,150,100,WIDTH-405])
    txt('screen = depth2 median tree; ridge = regularized linear model; tree = HGB. These families cannot share later-period ranks because only selected HGB predictions were saved. Every trial, feature set and window error is retained in the supporting bundle.','small')

page('Sources, checks and interpretation')
for label,run in [('A',A),('B',B),('C',C)]:
    txt(label+': '+run.relative_to(ROOT).as_posix(),'small')
txt('Checked all three prediction SHA256 values against completion records; verified identical target/timestamp/issuance keys and actual outcomes, and verified reproduced baseline predictions across stages. Metrics are read from the saved score tables; this report does not retrain or independently recompute all metrics. Completion records report successful model replay.')
txt('The strongest observed UP pooled MAE is C:S/H6 (11.206481). DOWN is the tuned selection policy (11.617624). These historical rankings do not guarantee future superiority. C all-price experiments were designed after prior results had been examined. Feature lists, hyperparameters and model choice reused earlier validation; uncertainty and multiple-search effects have not been quantified.')
txt('No feature-importance ranking is claimed: feature membership identifies inputs, not their individual contributions. New short inputs were selected frequently, but the best UP diagnostic variant uses the previous features, so short inputs cannot be credited with all gains.')
txt('Supporting ZIP: full-period ranks; every per-period rank; all A/B/C score tables; duplicate-control aliases; exact ordered features for every model variant/refit; all saved validation trials; selection cards, contracts, completion records, source hashes and report checks.')

def footer(canvas,doc):
    canvas.setStrokeColor(teal);canvas.line(38,30,W-38,30)
    canvas.setFont('Helvetica',8);canvas.setFillColor(navy)
    canvas.drawString(38,18,'ScaleWave | BTC 15-minute excursions | Complete three-stage report | 13 November 2026')
    canvas.drawRightString(W-38,18,str(doc.page))
pdf=OUT/'BTC_15m_Complete_Models_Features_and_Rankings_2026-11-13.pdf'
SimpleDocTemplate(str(pdf),pagesize=(W,H),leftMargin=38,rightMargin=38,topMargin=30,bottomMargin=42,
    title='BTC 15-minute complete models, features and rankings',author='ScaleWave research').build(story,onFirstPage=footer,onLaterPages=footer)
texts=[p.extract_text() for p in PdfReader(pdf).pages]
assert all(t.strip() for t in texts)
assert all(c in '\n'.join(texts) for c in newnames+f4z)
(SUP/'SOURCE_MANIFEST.json').write_text(json.dumps(sources,indent=2))
(SUP/'REPORT_CHECKS.json').write_text(json.dumps(dict(pages=len(texts),all67derived_names_present=True,
    full_period_models_by_target=pooled.groupby('target').size().to_dict(),prediction_hashes_verified=True,
    timestamp_truth_and_control_alignment_verified=True,validation_trials=sum(len(f) for f in trials)),indent=2))
bundle=OUT/'BTC_15m_Complete_Supporting_Tables_2026-11-13.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in SUP.iterdir():
        if p.is_file():z.write(p,p.name)
print(json.dumps(dict(pdf=str(pdf),bundle=str(bundle),pages=len(texts))))
