"""Build a source-linked report from saved 4h excursion results; no model fitting."""
from pathlib import Path
import json, hashlib, shutil, zipfile
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'Code/experiments/excursion_feature_search_4h_v1/20261113T082016050782Z'
B=ROOT/'Code/experiments/excursion_price_followup_4h_v1/20261113T093511890828Z'
R=B/'feature_reliance/20261113T100345312551Z'
OUT=ROOT/'Report/output/pdf'
SUP=ROOT/'Report/excursion_4h_report_2026-11-13'
OUT.mkdir(parents=True,exist_ok=True); SUP.mkdir(parents=True,exist_ok=True)
sources={}
def track(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(track(p))
for run in [A,B]:
    done=js(run/'COMPLETE.json')
    assert done['status']=='COMPLETE'
    assert hashlib.sha256((run/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']
    track(run/'predictions.csv')
orig=csv(A/'metrics.csv'); scores=csv(B/'metrics_all_variants.csv')
sa=js(A/'selections.json'); sb=js(B/'selections.json')
ca=js(A/'CONTRACT.json'); cb=js(B/'CONTRACT.json')
ranking=csv(R/'feature_ranking.csv'); audit=js(R/'AUDIT.json')
for filename in ['discover_excursion_features_4h.py','improve_excursion_features_4h.py','rank_excursion_features_4h.py']:
    track(ROOT/'Code'/filename)
trials=[]
for label,run in [('F4Z search',A),('Price follow-up',B)]:
    for p in sorted(run.glob('*/validation_trials.csv')):
        f=csv(p); f.insert(0,'experiment',label); f.insert(1,'fit_target',p.parent.name)
        trials.append(f)
all_trials=pd.concat(trials,ignore_index=True)
all_trials.to_csv(SUP/'all_validation_trials.csv',index=False)
orig.to_csv(SUP/'original_outer_metrics.csv',index=False)
scores.to_csv(SUP/'all_followup_outer_metrics.csv',index=False)
ranking.to_csv(SUP/'feature_reliance.csv',index=False)
for run,label in [(A,'original'),(B,'followup')]:
    shutil.copyfile(run/'selections.json',SUP/f'{label}_selections.json')

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
def metric_table(f,period=False):
    headers=(['Period'] if period else [])+['Variant','Rows','MAE','RMSE','IC','MAE gain*']
    rows=[]
    for _,r in f.iterrows():
        rows.append(([r.period] if period else [])+[r.model,int(r.rows),fmt(r.mae),fmt(r.rmse),fmt(r.ic),fmt(r.mae_gain_vs_original)])
    widths=([65] if period else [])+[WIDTH-(65 if period else 0)-290,48,61,61,55,65]
    table(headers,rows,widths)
    txt('* Positive gain means lower MAE than the original on the SAME rows. Units: log bps. IC: Spearman rank correlation.','small')

page('BTC | Four-hour excursion models','Research report - 13 November 2026 | Saved experimental results, model design and feature construction')
txt('Objective: find features that predict the largest upward and downward excursions over the next four hours. Two separate models predict nonnegative log-bps magnitudes on the same observations, including zero excursions.')
f=scores[scores.period.eq('ALL') & scores.model.isin(['original','selected','training_mean','training_median'])]
table(['Target','Approach','MAE','RMSE','Rank IC'],[[side(r.target),r.model,fmt(r.mae),fmt(r.rmse),fmt(r.ic)] for r in f.itertuples()],[70,290,135,135,WIDTH-630])
txt('Selected follow-up versus original: UP MAE falls 1.28%; DOWN MAE falls 1.21%. These are combined walk-forward scores across four refits, not the scores of one fixed October feature set.')
txt('Scope: the two completed 4h canonical excursion experiments only. Earlier 1h/4h absolute endpoint-magnitude studies and 48h models have different targets and are not mixed into these tables.','small')
txt('Sources: A = first F4Z search (20261113T082016050782Z); B = price follow-up (20261113T093511890828Z). All saved evaluation variants appear later; every inner screening/validation trial is supplied in the companion CSV bundle.','small')

page('Which models did we use?','Model families and their roles - distinct from feature-set variants')
table(['Model family','What it does','Where tested','Evaluation ranking'],[
    ['Histogram Gradient Boosting regression','Sequential ensemble of small regression trees, trained with absolute-error loss. Separate UP and DOWN models. No Ridge blending.','Earlier feature selection and every later-period model variant in both experiments.','Ranked on the next two pages, separately for each feature-set strategy.'],
    ['Ridge regression','Regularized linear regression; squared-error training, with earlier MAE used to select features.','Earlier validation in the first experiment.','Not ranked against later-period models: no saved outer Ridge predictions.'],
    ['Depth-2 decision tree','One small absolute-error tree for individual-feature screening.','Earlier validation in the first experiment.','Not ranked against later-period models: screening scores use earlier data and a smaller fit sample.'],
    ['Training median baseline','Constant median excursion from the eligible training period, updated at refits.','All evaluation periods; MAE-oriented benchmark.','Ranked alongside models as a baseline.'],
    ['Training mean baseline','Constant average excursion from the eligible training period, updated at refits.','All evaluation periods; squared-error benchmark.','Ranked alongside models as a baseline.']],[155,270,180,WIDTH-605])
txt('original, add_price_4h, price_only and similar names are feature-set variants of histogram boosting, not different model architectures. selected is the policy that chooses a variant using earlier validation at each refit. The selected policy and fixed variants can use different features across quarters.','small')
txt('There is no hybrid, neural network or separate direction model in these two 4-hour excursion experiments. Model families that have only earlier-validation scores are not assigned a later-period rank.','small')

ranked=[]
labels={'original':'Original F4Z boosting','selected':'Earlier-selected follow-up policy',
    'all_price_plus_original':'Boosting: original + all 51 price inputs','price_only':'Boosting: price inputs only',
    'remove_original_features':'Boosting: remove original group*','training_mean':'Training mean baseline',
    'training_median':'Training median baseline','add_volatility_ratios':'Boosting: original + volatility ratios'}
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    f=scores[scores.target.eq(target)&scores.period.eq('ALL')&scores.full_period_coverage.eq(True)].copy()
    assert f.rows.eq(18481).all()
    f['MAE_rank']=f.mae.rank(method='min').astype(int)
    f['RMSE_rank']=f.rmse.rank(method='min').astype(int)
    f['IC_rank']=f.ic.rank(method='min',ascending=False).astype(int)
    f=f.sort_values(['MAE_rank','model'])
    f['display_name']=[labels.get(m,'Boosting: original + '+m.removeprefix('add_').replace('price_','price ').replace('_',' ')) for m in f.model]
    ranked.append(f)
    page(f'{side(target)} | ranked model variants','Same 18,481 evaluation origins. Primary rank: pooled MAE, lower is better. RMSE and IC ranks are shown separately.')
    table(['MAE rank','Model / feature strategy','MAE','RMSE','RMSE rank','IC','IC rank'],
        [[r.MAE_rank,r.display_name,fmt(r.mae),fmt(r.rmse),r.RMSE_rank,fmt(r.ic),r.IC_rank] for r in f.itertuples()],
        [58,WIDTH-353,62,62,62,55,54])
    txt('Ranking describes the observed historical evaluation; it does not reselect or promote a model after viewing those outcomes. The first-ranked diagnostic variant may differ from the earlier-selected policy. Partial-period variants are excluded from this common-population ranking.','small')
    txt('* remove_original_features removes the original F4Z group from the greedy-search endpoint for each cycle. If no new block was added, that removal produces the median fallback. It is not the same as price_only.','small')
pd.concat(ranked,ignore_index=True).to_csv(SUP/'ranked_full_coverage_models.csv',index=False)

page('Targets, dates and walk-forward evaluation')
table(['Item','Definition'],[
['UP target','up_magnitude_log_bps = 10000 * ln(Pmax / P0)'],
['DOWN target','down_magnitude_log_bps = 10000 * ln(P0 / Pmin)'],
['Path','Canonical BTC reference-price minute path, including origin and four-hour endpoint. Both targets can be positive in the same window.'],
['Initial training origins','22 Apr 2022 00:30 to 29 Dec 2023 23:30 UTC; 29,615 original observations.'],
['Development origins','1 Jan 2024 to 22 Jan 2025 00:00 UTC; 18,481 original observations per target.'],
['Refits','January, April, July and October 2024; expanding history. January 2025 uses the October model and is reported separately.'],
['Purging','A training outcome must end strictly before its fit boundary. Validation outcomes must finish before their validation-window boundary. Horizon = 4h.'],
['Original gap','The original 48h package boundary gap remains absent; no extra observations were synthesized.'],
['Earlier validation','Three consecutive earlier quarters. January fit: Apr-Jun, Jul-Sep, Oct-Dec 2023. October fit: Jan-Mar, Apr-Jun, Jul-Sep 2024.']],[150,WIDTH-150])
txt('These runs preserve the supplied sampled population and inherited X validity masking. They do not add the later 48h experiment\'s source-binding-crossing exclusion. Original provisional/source-timing limitations remain. Historical evaluation has been exposed through repeated research.','small')

page('Model architecture and hyperparameters')
txt('Architecture: historical feature vector -> training-median imputation -> append one missing flag per feature -> training-only centering/scaling -> predictor -> clip output below at zero. UP and DOWN have independent fitted models. There is no neural network and no blended hybrid.')
table(['Model','Architecture / settings','Role'],[
['Histogram Gradient Boosting','Absolute-error loss; median initialization plus 80 sequential boosted regression trees; learning_rate=0.05; max_depth=3; max_leaf_nodes=7; min_samples_leaf=128; l2_regularization=1; early_stopping=False; random_state=48.','Selected in every first-run quarterly fit; same settings retained for all follow-up variants.'],
['Ridge regression','Linear intercept + weighted standardized inputs; squared-error objective with L2 penalty; alpha=0.01 * training-row count. Selected by earlier MAE despite squared-error fit.','First experiment feature-combination challenger only. No separate later-period Ridge predictions were saved.'],
['Screening decision tree','Single tree; criterion=absolute_error; max_depth=2; min_samples_leaf=128; random_state=48. At most 4,096 evenly spaced training observations.','First experiment individual-feature screening; validation uses all eligible validation rows.'],
['Mean / median baselines','One constant per target and refit, calculated using all eligible training outcomes.','Later-period evaluation benchmarks; median is also the first-run fallback.']],[145,430,WIDTH-575])
txt('Preprocessing uses training rows only. Imputation median defaults to zero for all-missing columns; near-zero standard deviations are set to one. Missing indicators are appended even when constant. Latest UP: 19 raw features -> 38 transformed inputs. Latest DOWN: 55 -> 110.','small')
txt('A fitted boosting model adds small tree corrections sequentially to its initial prediction. It is an ensemble of trees, but not a blend of Ridge and trees.','small')
txt('Verified fitted defaults: max_bins=255; max_features=1.0; warm_start=False; no monotonic or interaction constraints. No categorical inputs. Early stopping is disabled, so its validation_fraction/tolerance/patience settings do not control these fits.','small')

page('How features were selected')
table(['Stage','Experiment A: F4Z search','Experiment B: price follow-up'],[
['Universe','522 supplied masked F4Z value columns; no mandatory features.','Original per-refit feature set plus 51 new price inputs, grouped into six lookbacks and ratios.'],
['Quality','In every inner training window: >=80% finite coverage, >=1,000 readings and variation.','Same quality checks for newly derived inputs; original sets retained as comparison anchors.'],
['Screen / search','Median-tree single-feature screen. Remove redundancy at |Spearman| >=0.90 on earliest training rows. Retain up to16 candidates; greedy forward selection up to6 features for Ridge and boosting.','Test each block addition; greedy search adds up to2 blocks. Independently test all price + original and price-only. Test removal of each chosen block and original feature group.'],
['Admission','Each addition: >=0.1% pooled MAE improvement and wins in >=2 of3 earlier windows.','Same gate for additions. Final candidate must pass against BOTH original and median.'],
['Pruning / fallback','Refit removals; drop features if no worse pooled and in>=2 windows. Choose eligible lower-MAE model; otherwise training median.','Select lowest eligible earlier MAE among tried variants; otherwise unchanged original.'],
['Later data','Used only after each selection freezes.','Every tested variant evaluated after selection freezes; extra scores are diagnostic, not selector inputs.']],[90,335,WIDTH-425])
txt('Selection is adaptive on reused inner windows. Greedy screening can miss interaction-only signals. Original feature sets in B were already selected historically; this is not a new nested confirmation study.','small')

page('New price features | construction')
txt('Source: Week 1/Dataset/canonical_returns_5m.parquet, checksum verified by canonical_return_features.py. Each forward five-minute return is indexed at its END time. For forecast T, use only endpoints strictly before T. A lookback h includes 12*h returns. Incomplete windows remain missing.')
table(['Suffix','Formula / calculation','Meaning'],[
['rv','sqrt(sum(r_i^2))','Realized volatility in log bps.'],
['up_rv','sqrt(sum(max(r_i,0)^2))','Positive-return semivolatility.'],
['down_rv','sqrt(sum(min(r_i,0)^2))','Negative-return semivolatility; nonnegative output.'],
['net','sum(r_i)','Signed net historical log return.'],
['past_range','max(L) - min(L)','Range of reconstructed relative log-price levels.'],
['distance_from_high','max(L) - L_latest','Distance below historical five-minute-close high.'],
['distance_from_low','L_latest - min(L)','Distance above historical five-minute-close low.'],
['max_abs_return','max(abs(r_i))','Largest individual five-minute move.'],
['rv_ratio__a h_b h','(RV_a/sqrt(a)) / max(RV_b/sqrt(b), 0.000001)','Dimensionless duration-normalized volatility ratio.']],[160,270,WIDTH-430])
txt('L is the cumulative sum of past log returns, including the window starting level. These are relative log-price levels, not dollar prices. Historical range features use five-minute closes and do not capture between-close extrema.','small')
txt('Eight measurements at 1h,4h,12h,24h,48h,168h =48 inputs. Three ratios (1h/24h,4h/24h,24h/168h) bring the total to51. Prefix truncation and future-return perturbation checks passed.','small')

page('Worked calculation | four five-minute returns')
txt('Illustrative 20-minute window: +10, -5, +20, -10 log bps. Actual 1h/4h windows use12/48 returns respectively.')
table(['Point','Return','Relative log-price level'],[['Start','--','0'],['5 minutes','+10','10'],['10 minutes','-5','5'],['15 minutes','+20','25'],['20 minutes','-10','15']],[160,200,WIDTH-360])
table(['Feature','Arithmetic','Result'],[
['RV','sqrt(100+25+400+100)','25.00 log bps'],['Up RV','sqrt(100+400)','22.36 log bps'],
['Down RV','sqrt(25+100)','11.18 log bps'],['Net','10-5+20-10','+15.00 log bps'],
['Range / distance from high / distance from low','25-0 / 25-15 / 15-0','25 / 10 / 15 log bps'],
['Maximum absolute return','max(10,5,20,10)','20.00 log bps'],
['Separate ratio example','If RV_1h=30 and RV_24h=100: (30/sqrt(1))/(100/sqrt(24))','1.47; recent normalized volatility is47% higher']],[190,350,WIDTH-540])

# Exact unique feature dictionary and per-cycle mappings keep lists complete without repetition.
f4z=sorted({c for s in sa+sb for c in s['features'] if c.startswith('F4Z__')})
ids={c:f'F{i+1:02d}' for i,c in enumerate(f4z)}
page('Exact F4Z feature names','IDs below are used only to keep the quarterly feature-set tables readable.')
table(['ID','Exact feature name'],[[ids[c],c] for c in f4z],[42,WIDTH-42],codecol=1)
for block,names in cb['groups'].items():
    pass
newnames=[c for names in cb['groups'].values() for c in names]
for start in range(0,len(newnames),20):
    page('Exact derived feature names',f'{start+1}-{min(start+20,len(newnames))} of51; all generated by the price follow-up runner')
    table(['No.','Exact feature name','October UP','October DOWN'],[[i+1,c,'Yes' if c in sb[-2]['features'] else '--','Yes' if c in sb[-1]['features'] else '--'] for i,c in enumerate(newnames) if start<=i<start+20],[40,WIDTH-220,90,90],codecol=1)

page('Features used at each quarterly refit')
table(['Fit','Target','Original F4Z features','Selected follow-up additions','Total'],[
    [s['issue'][:7],side(s['target']),', '.join(ids[c] for c in s['features'] if c in ids),
     s['variant'].replace('add_','').replace('all_price_plus_original','All51 new price inputs'),len(s['features'])] for s in sb],[75,55,235,WIDTH-425,60])
txt('price_1h (or another lookback) means the complete eight-measurement block admitted at that lookback. volatility_ratios means the three ratios. The exact ordered lists are in followup_selections.json.','small')
txt('Latest October UP:3 F4Z+16 price=19 raw inputs. Latest DOWN:4 F4Z+51 price=55 raw inputs. The same October models predict January2025. Earlier-cycle original sets above are also the final selected sets in experiment A.','small')

page('Original experiment | all later-period models')
table(['Target','Model','Rows','MAE','RMSE','IC'],[[side(r.target),r.model,int(r.rows),fmt(r.mae),fmt(r.rmse),fmt(r.ic)] for r in orig[orig.period.eq('ALL')].itertuples()],[65,240,70,120,120,WIDTH-615])
txt('prediction = selected histogram boosting in every refit. Ridge and depth-2 screening trees were evaluated on earlier validation only; no comparable later-period Ridge/screening predictions were saved. Their complete trial scores are in all_validation_trials.csv.','small')
v=all_trials[all_trials.experiment.eq('F4Z search')]
rows=[]
for (fit,model),g in v.groupby(['fit_target','model']):
    # Empty-feature fallback labeled tree is not a trained tree trial.
    g=g[g.features.ne('[]')]
    if g.empty:continue
    rows.append([fit[:7], 'UP' if '_up_' in fit else 'DOWN',model,len(g),fmt(g.pooled_mae.min())])
for start in [0,12]:
    page('Earlier-validation search summary',('January and April fits' if start==0 else 'July and October fits')+' | Minimum recorded trial, not an outer score; exact trials are in the companion CSV.')
    table(['Fit','Target','Family','Trials','Minimum MAE'],rows[start:start+12],[95,70,160,100,WIDTH-425])

for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    for full in [True,False]:
        f=scores[scores.target.eq(target)&scores.period.eq('ALL')&scores.full_period_coverage.eq(full)]
        page(f'{side(target)} | '+('all full-coverage variants' if full else 'partial-coverage variants'),
             'Saved pooled evaluation scores. These are comparisons, not a post-hoc promotion rule.')
        metric_table(f)
        txt('Variant naming: add_X appends X to that cycle\'s original features; price_only uses all admitted new price inputs; all_price_plus_original combines both. remove_X removes that group from the greedy-search endpoint, which can differ by cycle. selected is the earlier-validation selection policy.','small')
        if not full:txt('These variants were tried in only some cycles. Their absolute errors must NOT be compared to full-period errors. Use rows and matched-original gains.','small')

for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    page(f'{side(target)} | selected versus original by period')
    f=scores[scores.target.eq(target)&~scores.period.eq('ALL')&scores.model.isin(['original','selected'])]
    metric_table(f,True)
    txt('UP: MAE improves in all2024quarters but worsens slightly inJanuary. DOWN: MAE improves in all five periods. RMSE improvements are not uniform; larger-error behavior differs from MAE.','small')

page('Feature reliance | latest fitted models')
txt('Saved October models evaluated on5,425 Oct2024-Jan2025 origins. Each raw feature was permuted in48-row blocks, including its missingness, over5fixed-seed repeats. No retraining. Larger mean MAE increase indicates greater model reliance on that input.')
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    txt(side(target)+' - top five individual inputs','head')
    f=ranking[ranking.target.eq(target)&ranking.kind.eq('individual')].head(5)
    table(['Rank','Feature','Mean MAE increase','Shuffle SD'],[[r.rank,r.feature,fmt(r.mean_mae_increase),fmt(r.shuffle_sd)] for r in f.itertuples()],[40,WIDTH-230,100,90],codecol=1)
txt('This is model reliance, not causal importance or additive contribution. Correlated inputs can substitute for one another; shuffling can create unusual combinations. Shuffle SD is not a confidence interval. The full ranking is in feature_reliance.csv.','small')

page('Reading the results and source record')
txt('The selected follow-up improves pooled MAE, RMSE and rankIC for both targets. Improvements are modest. A diagnostic full-coverage DOWN variant, all_price_plus_original, has lower pooled MAE (51.866) than selected (51.938); it was not the selected policy in every cycle. Calling it a proven winner after observing outer scores would add post-hoc selection.')
txt('Baseline IC is undefined within constant-prediction periods. Pooled baseline IC can be nonzero because the constant changes at refits. Feature counts exclude their appended missing indicators. All scores are saved experiment outputs; this report does not retrain models or independently recompute metric values.')
table(['Source','Location relative to workspace'],[
['A: original features and scores',str(A.relative_to(ROOT))],['B: price additions and all-variant scores',str(B.relative_to(ROOT))],
['Feature reliance',str(R.relative_to(ROOT))],['Code','Code/discover_excursion_features_4h.py; Code/improve_excursion_features_4h.py'],
['Exact validation evidence','Companion bundle: all_validation_trials.csv (all saved screen/forward/removal trials and window MAEs)'],
['Exact settings / provenance','Original CONTRACT.json and follow-up CONTRACT.json; report SOURCE_MANIFEST.json contains hashes.']],[155,WIDTH-155])
txt('Source prediction hashes were checked against both completion records. Both runs report successful model replay. This remains historical exposed research with adaptive selection and inherited source-availability limitations. No significance claim or guarantee of future improvement is made.','small')

# Every saved follow-up period/variant score is included, not just selected winners.
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    f=scores[scores.target.eq(target)&~scores.period.eq('ALL')]
    for start in range(0,len(f),19):
        page(f'Appendix | {side(target)} all period / variant scores',f'Rows{start+1}-{min(start+19,len(f))} of{len(f)}. Complete saved follow-up evaluation record.')
        metric_table(f.iloc[start:start+19],True)

def footer(canvas,doc):
    canvas.setStrokeColor(teal);canvas.line(38,30,W-38,30)
    canvas.setFont('Helvetica',8);canvas.setFillColor(navy)
    canvas.drawString(38,18,'ScaleWave | BTC four-hour excursions | 13 November 2026')
    canvas.drawRightString(W-38,18,str(doc.page))
path=OUT/'BTC_4h_Excursion_Models_Features_and_Results_2026-11-13.pdf'
SimpleDocTemplate(str(path),pagesize=(W,H),leftMargin=38,rightMargin=38,topMargin=30,bottomMargin=42,
                  title='BTC 4h excursion models, features and results',author='ScaleWave research').build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(path)
texts=[p.extract_text() for p in reader.pages]
assert all(t.strip() for t in texts)
assert all(c in '\n'.join(texts) for c in newnames)
(SUP/'SOURCE_MANIFEST.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
(SUP/'REPORT_CHECKS.json').write_text(json.dumps(dict(pages=len(texts),all51derived_names_present=True,
    outer_score_rows=len(scores),inner_trial_rows=len(all_trials),prediction_hashes_verified=True),indent=2))
for run,label in [(A,'original'),(B,'followup')]:
    shutil.copyfile(run/'CONTRACT.json',SUP/f'{label}_CONTRACT.json')
zip_path=OUT/'BTC_4h_Excursion_Supporting_Tables_2026-11-13.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in SUP.iterdir():
        if p.is_file():z.write(p,p.name)
print(json.dumps(dict(pdf=str(path),bundle=str(zip_path),pages=len(texts),outer_rows=len(scores),validation_trials=len(all_trials))))
