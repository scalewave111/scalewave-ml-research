"""Report-only generation from saved48h excursion artifacts. No fitting/inference."""
from pathlib import Path
import json,hashlib,zipfile,shutil,re
from xml.sax.saxutils import escape
import pandas as pd
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib import colors
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'Code/experiments/excursion_models_48h_v1/20261113T093614902840Z'
B=ROOT/'Code/experiments/hist_excursion_48h_v1/20261113T101109247925Z'
CAN=Path('C:/Users/Siddh/Desktop/ML/October/Week 3/Dataset/HANDOFF_2_CANONICAL_XY_V1_1/HANDOFF_2_CANONICAL_XY_V1_1')
OUT=ROOT/'Report/output/pdf';SUP=ROOT/'Report/excursion_48h_report_2026-11-13'
OUT.mkdir(parents=True,exist_ok=True);SUP.mkdir(parents=True,exist_ok=True)
sources={}
def track(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(track(p))
for run in [A,B]:
    done=js(run/'COMPLETE.json');assert done['status']=='COMPLETE' and done['replay_passed']
    assert hashlib.sha256(track(run/'predictions.csv').read_bytes()).hexdigest()==done['prediction_sha256']
ca,cb=js(A/'CONTRACT.json'),js(B/'CONTRACT.json')
sa,sb=js(A/'SELECTIONS.json'),js(B/'SELECTIONS.json')
metrics=csv(B/'metrics.csv');oldmetrics=csv(A/'metrics.csv')
for name in ['mean','median','fixed_ridge','ridge','tree','e2','selected']:
    left=oldmetrics[oldmetrics.model.eq(name)].sort_values(['target','period'])
    right=metrics[metrics.model.eq('prior_'+name)].sort_values(['target','period'])
    assert list(zip(left.target,left.period))==list(zip(right.target,right.period))
    np.testing.assert_allclose(left[['mae_log_bps','rmse_log_bps','rank_ic','bias_log_bps']],right[['mae_log_bps','rmse_log_bps','rank_ic','bias_log_bps']],equal_nan=True,atol=1e-9,rtol=0)
raw=cb['identity']['specs'][0]['columns'];cols=ca['identity']['packs']['E2_235'];assert len(cols)==235 and len(raw)==13
dictionary={r['column']:r for r in js(CAN/'FEATURE_DICTIONARY.json')}
for run in [A,B]:
    for f in run.glob('*.py'):track(f)
track(CAN/'team_tools/TEAM_DATASET_SPEC.md')
for r in cb['identity']['hashes']:
    p=Path(r)
    if p.suffix=='.py' and p.exists():track(p)
selected=[s for s in sb if s['policy']=='engineered235_mae'];assert len(selected)==8
modelrecords=[]
for s in selected:
    path=track(B/s['artifact']);assert sources[str(path)]==s['sha256']
    a=joblib.load(path);assert a['columns']==cols
    model=a['model'];params=model.get_params()
    assert params['loss']=='absolute_error' and model.n_iter_==80
    leaves=[int(t[0].nodes['is_leaf'].sum()) for t in model._predictors]
    modelrecords.append(dict(target=s['target'],issue=s['issue'],artifact=s['artifact'],parameters=params,
        iterations=int(model.n_iter_),trees=len(model._predictors),min_leaves=min(leaves),max_leaves=max(leaves),
        training_rows=s['training_rows'],features=len(a['columns'])))
(SUP/'chosen_model_parameters.json').write_text(json.dumps(modelrecords,indent=2,default=str))
metrics.to_csv(SUP/'all_period_scores.csv',index=False)
for run,label in [(A,'ridge_lightgbm'),(B,'histogram')]:
    shutil.copyfile(track(run/'SELECTIONS.json'),SUP/f'{label}_selections.json')
    shutil.copyfile(track(run/'validation_metrics.csv'),SUP/f'{label}_validation_trials.csv')
    shutil.copyfile(run/'CONTRACT.json',SUP/f'{label}_contract.json')

ids={c:f'B{i+1:02d}' for i,c in enumerate(raw)}
baseinfo=[
('Peer liquidation historical active rate','15m / 5m / 1h','Peer quote-liquidation total excludes BTC. Input is historical mean of activity indicators, not current size.'),
]
names={
'prior_mean':'Training mean baseline','prior_median':'Training median baseline',
'prior_fixed_ridge':'Fixed Ridge (UP3 / DOWN7)','prior_ridge':'Tuned Ridge: pack + alpha',
'prior_tree':'Direct LightGBM (235)','prior_e2':'Ridge + LightGBM residual (E2)',
'prior_selected':'Earlier Ridge/LightGBM selector',
'hgb_rmse':'HGB L2: cross-pack selector','selected_rmse':'HGB L2 selector + mean fallback',
'raw13_rmse':'HGB L2: raw13','raw13_price51_rmse':'HGB L2: raw13 + price51',
'engineered235_rmse':'HGB L2: engineered235','engineered235_price51_rmse':'HGB L2: engineered235 + price51',
'hgb_mae':'HGB L1: cross-pack selector','selected_mae':'HGB L1 selector + median fallback',
'raw13_mae':'HGB L1: raw13','raw13_price51_mae':'HGB L1: raw13 + price51',
'engineered235_mae':'HGB L1: engineered235 [chosen]','engineered235_price51_mae':'HGB L1: engineered235 + price51'}
pooled=metrics[metrics.period.eq('ALL')].copy()
pooled['mae_rank']=pooled.groupby('target').mae_log_bps.rank(method='min').astype(int)
pooled['rmse_rank']=pooled.groupby('target').rmse_log_bps.rank(method='min').astype(int)
pooled['readable_model']=pooled.model.map(names);assert pooled.readable_model.notna().all()
pooled.sort_values(['target','mae_log_bps']).to_csv(SUP/'pooled_rankings_full_precision.csv',index=False)

formula_rows=[]
for i,col in enumerate(cols,1):
    if '__' in col and col.split('__',1)[1] in ids:
        op,base=col.split('__',1);formula=f'{op} applied to {ids[base]}: {base}'
    elif col.startswith('LONG_'):formula='Long-price formula table; window encoded in final suffix.'
    elif col.startswith('weekday_'):formula='1 if UTC weekday equals encoded integer (Monday=0), else0.'
    elif col.startswith('week_of_month_'):formula='1 if floor((UTC day-1)/7)+1 equals encoded integer, else0.'
    elif col.startswith('quarter_'):formula='1 if UTC quarter equals encoded integer, else0.'
    else:raise ValueError(col)
    formula_rows.append(dict(number=i,column=col,calculation=formula))
pd.DataFrame(formula_rows).to_csv(SUP/'exact_235_feature_manifest.csv',index=False)
(SUP/'raw_feature_dictionary.json').write_text(json.dumps({c:dictionary[c] for c in raw if c in dictionary},indent=2))

W,H=landscape(A4);WIDTH=W-76
NAVY=colors.HexColor('#17354A');TEAL=colors.HexColor('#087E86')
styles={n:ParagraphStyle(n,fontName=f,fontSize=s,leading=l,textColor=c,spaceAfter=a)
 for n,f,s,l,c,a in [('title','Helvetica-Bold',22,27,NAVY,10),('head','Helvetica-Bold',12,16,TEAL,8),
 ('body','Helvetica',10,14,NAVY,8),('small','Helvetica',8.3,11,NAVY,6),
 ('cell','Helvetica',8.1,10.4,NAVY,0),('th','Helvetica-Bold',8.1,10.4,colors.white,0),
 ('code','Courier',7,9,NAVY,0)]}
story=[]
def readable(t):
    # Keep exact identifiers/paths intact while spacing prose numbers and lists.
    pieces=re.split(r'(\s+)',str(t))
    for i,v in enumerate(pieces):
        if '_' in v or '\\' in v or '.csv' in v or '.json' in v or '.py' in v or re.search(r'\d[eE][-+]\d',v):
            continue
        v=re.sub(r'(\d+)(h|m|d)(?=[a-zA-Z])',r'\1\2 ',v)
        v=re.sub(r'(?<=[a-z])(?=\d)', ' ',v)
        v=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ',v)
        v=re.sub(r'([,;])(?=[^\s\d])',r'\1 ',v)
        v=v.replace('F4 Z','F4Z')
        pieces[i]=v
    return ''.join(pieces)
def para(t,style='body'):return Paragraph(escape(str(t) if style=='code' else readable(t)),styles[style])
def txt(t,style='body'):story.append(para(t,style))
def page(title,sub=''):
    if story:story.append(PageBreak())
    txt(title,'title')
    if sub:txt(sub,'small')
def table(headers,rows,widths=None,codecol=None):
    data=[[para(v,'th') for v in headers]]+[[para(v,'code' if j==codecol else 'cell') for j,v in enumerate(row)] for row in rows]
    t=Table(data,colWidths=widths or [WIDTH/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF5F7'),colors.white]),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,7)])
def f(v):return '--' if pd.isna(v) else f'{v:.3f}'
def side(t):return 'UP' if t.startswith('up_') else 'DOWN'
def row(t,m,period='ALL'):return metrics[(metrics.target==t)&(metrics.model==m)&(metrics.period==period)].iloc[0]
U='up_magnitude_log_bps';D='down_magnitude_log_bps'

page('BTC | 48-hour excursion models','13 November 2026 | Model architecture, feature calculations and complete pooled rankings')
txt('Working choice: two separate absolute-error histogram gradient boosting models, each using the existing 235 engineered features. MAE is the primary metric requested by the user.')
table(['Target','Chosen model MAE','RMSE','Rank IC','Previous best MAE'],[
 [side(t),f(row(t,'engineered235_mae').mae_log_bps),f(row(t,'engineered235_mae').rmse_log_bps),f(row(t,'engineered235_mae').rank_ic),
  f(row(t,'prior_median' if t==U else 'prior_e2').mae_log_bps)] for t in [U,D]],[90,145,145,145,WIDTH-525])
txt('All error and bias values are log basis points. Lower MAE is better. Rank IC is Spearman correlation; higher positive values indicate better ordering of small versus large excursions.')
txt('What is ranked','head')
txt('All 19 saved model variants, baselines and selection policies from the two completed 48h excursion experiments are ranked separately for UP and DOWN. Both use the same 18,385 forecast origins. Endpoint-return models, conditional endpoint sizes, 1h models and 4h models have different targets and are not mixed into these rankings.')
txt('The chosen 235-feature pack was the strongest observed MAE pack after inspecting results. It is the working reference, not an independently confirmed winner. Its RMSE is worse than the prior best UP selector and DOWN fixed Ridge.')
txt('Source A: excursion_models_48h_v1/20261113T093614902840Z. Source B: hist_excursion_48h_v1/20261113T101109247925Z. Prediction checksums verified against COMPLETE records; both runs report replay_passed=true. This report performs no model training or metric recomputation.','small')

page('What the two targets mean','Both models train on the same valid price paths; neither is conditional on the endpoint direction.')
table(['Target','Definition','Interpretation'],[
 ['UP excursion','10000 * ln(Pmax / P0)','Largest rise above the origin anywhere in the next48h.'],
 ['DOWN excursion','-10000 * ln(Pmin / P0)','Largest fall below the origin anywhere in the next48h; stored as a nonnegative size.'],
 ['Path','2,881 minute prices including origin and48h endpoint','P0=start;Pmax/Pmin=maximum/minimum canonical minute VWAP. Both excursions can be positive.']],[130,265,WIDTH-395])
txt('Example: start100, reach110, fall95, finish102. UP excursion is953.102logbps; DOWN excursion is512.933logbps. Endpoint return is a different outcome. No direction probabilities multiply these forecasts.')
table(['Population / split','Recorded design'],[
 ['Source coverage','22Apr2022 through23Jan2025 source days; original playground forecast grid starts22Apr2022 00:30UTC.'],
 ['Input alignment','48,096 original playground rows;48,000 retained after canonical source-binding exclusions.'],
 ['Evaluation','1Jan2024 through22Jan2025;18,385 origins per target,36,770 target-row predictions.'],
 ['Quarterly refits','1Jan,1Apr,1Jul,1Oct2024. October model continues throughJan2025;January metrics are shown separately.'],
 ['Validity','Canonical VALID labels,finite/nonnegative sizes; no missing-Y imputation.96origins crossing the29Oct2024 00:01UTC source-binding transition excluded.'],
 ['Maturity','Every training label ends strictly before its fit cutoff;48h future paths cannot overlap the validation start.'],
 ['Model selection','Three earlier2month windows starting9,6,3months before issuance. L1 settings minimize pooled MAE;L2 settings minimize pooled MSE.']],[150,WIDTH-150])

page('Chosen model | architecture','scikit-learn HistGradientBoostingRegressor(loss="absolute_error")')
table(['Stage','Operation'],[
 ['1. Inputs','235causal feature columns. Missing values remainNaN; no added missing-indicator columns and no standard scaling for HGB.'],
 ['2. Histogram binning','Each nonmissing feature is quantized into up to127 training-derived bins; missing values have a separate bin/routing behavior. Bin thresholds are learned only on the training rows.'],
 ['3. Initial prediction','Constant median-oriented starting prediction for absolute-error loss.'],
 ['4. Additive ensemble','80sequential shallow regression trees. Each stage corrects the current absolute-loss fit. Maximum depth2 and at most4leaves per tree.'],
 ['5. Output','prediction(x)=max(0, initial_prediction +0.05*sum(tree_stage(x))). UP/DOWN are fitted separately; no probability weighting.'],
 ['Meaning','Median-oriented estimate of each future excursion. It is not an expected signed endpoint return or a trading decision.']],[145,WIDTH-145])
txt('The saved estimator parameters and tree structures were inspected without running inference. All eight selected quarterly models have80iterations/80trees and use the same small setting. Actual fitted leaves per tree can be fewer than the allowed maximum.','small')
table(['Parameter','Chosen value','Role'],[
 ['loss','absolute_error','Optimizes absolute prediction error (MAE orientation).'],['learning_rate','0.05','Scales each tree update.'],
 ['max_iter','80','Fixed number of boosting iterations.'],['max_depth / max_leaf_nodes','2 / 4','Limits tree complexity.'],
 ['min_samples_leaf','128','Minimum samples per leaf.'],['l2_regularization','10.0','Regularizes leaf fitting.'],
 ['max_bins / random_state','127 / 48','Histogram resolution and reproducibility.'],['early_stopping','False','No random holdout split;iteration count fixed before outer evaluation.']],[165,150,WIDTH-315])

page('Saved quarterly model records','The choice is a model recipe refitted over time, not one model trained across the entire evaluation period.')
table(['Target','Issue','Training rows','Features','Trees','Leaf range','Setting'],[
 [side(r['target']),r['issue'][:10],r['training_rows'],r['features'],r['trees'],f"{r['min_leaves']}-{r['max_leaves']}",'small'] for r in modelrecords],[65,110,105,85,75,95,WIDTH-535])
txt('Common estimator defaults recorded from saved artifacts:','head')
params=modelrecords[0]['parameters']
keys=['max_features','categorical_features','monotonic_cst','interaction_cst','warm_start','validation_fraction','n_iter_no_change','tol','scoring']
table(['Parameter','Recorded value / status'],[[k,str(params[k])+(' (inactive because early_stopping=False)' if k in ['validation_fraction','n_iter_no_change','tol','scoring'] else '')] for k in keys if k in params],[180,WIDTH-180])
txt('Exact version, all parameters, artifact names and hashes are included in chosen_model_parameters.json and SOURCE_MANIFEST.json in the companion bundle. All listed raw inputs are numeric; the calendar flags are numeric indicators.','small')

page('Feature design | 235 inputs','The additional51price columns tested in the follow-up are NOT included in the chosen235-feature model.')
table(['Block','Count','Construction'],[
 ['Market transformations','195','13underlying inputs x15transformations each:1lag +12rolling measures +2mean ratios.'],
 ['Calendar','16','7weekday flags +5week-of-month flags +4quarter flags.'],
 ['Long price history','24','6price measures x4lookbacks:24h,48h,168h,720h.'],
 ['Total','235','195+16+24. No future labels,endpoint direction probabilities or target values enter X.']],[160,75,WIDTH-235])
table(['Input IDs','Underlying measure','Source calculation'],[
 ['B01,B02,B03','Historical peer-liquidation active rate:15m,5m,1h','Start from the sum of manifest-declared peer-asset cross-venue liquidation quote totals (BTC excluded). The supplied F4Z input is mean(prior activity indicators), not the current total.'],
 ['B04','4h/24h price-volatility ratio','(RV4h/sqrt(4)) / max(RV24h/sqrt(24),1e-6).'],
 ['B05','7day realized price volatility','sqrt(sum of squared prior5m log-bps returns over168h).'],
 ['B06','1h liquidation venue-breadth percentile','Raw breadth=observed active venue count/Neligible, then historical midrank percentile.'],
 ['B07','5m funding-persistence percentile','Raw=end-sign occupancy over the valid elapsed funding path, then historical midrank percentile.'],
 ['B08','1h basis-level robust z-score','Raw=cross-venue last basis level; z=(raw-historical median)/(1.4826*MAD).']],[100,215,WIDTH-315])
txt('B01-B13 exact source column names are listed in the appendix. Source formulas are from the canonical feature dictionary; current experiments load those historical F4Z values rather than rebuilding their upstream baselines.','small')

page('Underlying measures | normalization','Complete the13-input foundation before applying the new rolling transforms.')
table(['Input IDs','Underlying measure','Source calculation'],[
 ['B09,B10,B11','Flow-activity percentiles:1h,15m,5m','Raw=sum of rolling taker-buy quote and taker-sell quote. This is total aggressive flow, not buy-minus-sell flow. Dictionary transform is ln(1+raw), then historical midrank percentile.'],
 ['B12','1h liquidation venue-fragmentation percentile','Raw=(1-HHI)/(1-1/Neligible), where HHI=sum of squared venue shares. Then historical midrank percentile. Invalid when total liquidation is zero.'],
 ['B13','1h liquidation temporal-concentration percentile','Raw=peak source-bucket liquidation quote/rolling total liquidation quote. Then historical midrank percentile. Invalid when total is zero.']],[100,225,WIDTH-325])
txt('Canonical F4Z formulas','head')
table(['Transform','Formula / interpretation'],[
 ['Historical percentile','(count(history<current)+0.5*count(history=current))/n. A rank in[0,1], not a percent price return.'],
 ['Robust z-score','(transformed_current-median(transformed_history))/(1.4826*MAD). MAD=median absolute deviation from the median. Null if MAD=0.'],
 ['Historical active rate','Mean of prior activity indicators. The active-value transform label is LN_1P_ACTIVE_VALUE, but this field is the historical activity frequency, not a logged liquidation amount.'],
 ['Validity / history','Uses supplied source quality/availability masks. Exact historical normalization membership is inherited from the source release; it is not the7/14/30period rolling window created by this experiment.']],[165,WIDTH-165])
txt('No missing inputs are converted into future-derived estimates. Invalid original values arrive asNaN; HGB learns missing-value routing during fitting. The report does not claim to have reconstructed upstream F4Z history or independently verified all raw venue inputs.','small')

page('How the195 market transformations are calculated','Let x_j(t) be underlying input B_j on the exact30minute grid. First set z_j(t)=x_j(t-30minutes).')
table(['Columns per input','Count','Exact calculation'],[
 ['lag1','1','z_j(t). Reindex to the full elapsed30minute grid before shifting; gaps are not compressed.'],
 ['std7,std14,std30','3','Population standard deviation (ddof=0) of the last w lagged values, w in{7,14,30}.'],
 ['range7,range14,range30','3','max(window)-min(window).'],
 ['iqr7,iqr14,iqr30','3','75th percentile minus25th percentile of the window (pandas quantile behavior).'],
 ['zscore7,zscore14,zscore30','3','(z_j(t)-window mean)/window standard deviation;missing if standard deviation<=1e-12.'],
 ['ma_ratio7_30,ma_ratio14_30','2','mean(short window)/mean(30period window);missing when abs(denominator)<=1e-8.'],
 ['Per input / total','15 /195','All w observations must exist and be finite after cleaning;otherwise that window-derived feature remainsNaN.']],[205,65,WIDTH-270])
txt('Window lengths:7periods=3.5h,14periods=7h,30periods=15h. These are periods, NOT7/14/30days. Values are shifted before rolling.','head')
txt('Exact name rule: operation + "__" + original column name. Example: std7__PRICE__vol_ratio_4h_24h. Combining the15operation names above with all13appendix source names defines all195exact feature names. The companion CSV enumerates every column.','small')
table(['Calendar flags','Calculation'],[
 ['weekday_0 ... weekday_6','1 if UTC weekday equals0(Monday) through6(Sunday), else0.'],
 ['week_of_month_1 ... week_of_month_5','1 if floor((day_of_month-1)/7)+1 equals1...5, else0.'],
 ['quarter_1 ... quarter_4','1 if UTC calendar quarter equals1...4, else0.']],[230,WIDTH-230])

page('How the24 long-price features are calculated','r_s=10000*ln(P_s/P_(s-5m)), indexed at its endpoint s. Use only s<t;complete h-hour windows need12*h returns.')
table(['Exact prefix (append _24h/_48h/_168h/_720h)','Formula'],[
 ['LONG_rv','sqrt(sum(r_s^2))'],['LONG_up_rv','sqrt(sum(max(r_s,0)^2))'],
 ['LONG_down_rv','sqrt(sum(min(r_s,0)^2))'],['LONG_abs_net','abs(sum(r_s))'],
 ['LONG_abs_sum','sum(abs(r_s))'],
 ['LONG_range_logbps','max(L_s)-min(L_s), L_s=cumulative sum of5m log returns;rolling endpoints in[t-h,t).']],[350,WIDTH-350])
txt('All six are evaluated at h=24,48,168,720hours:6x4=24features. The source loader verifies the canonical5m return hash. Missing source returns invalidate the required complete window.','small')
txt('Implementation detail for the range','head')
txt('The existing E2 range uses the cumulative-level values at return endpoints within the rolling interval. It is a5mclose-path range and does not include a separately added left-boundary price. It is not a full minute-high/minute-low range. The additional51feature block uses an explicit n+1point prior5mclose path, so these two range representations need not be numerically identical.')
txt('Causal safeguards','head')
txt('All rolling windows exclude the forecast origin. Prefix-only rebuilding and future-return perturbations were checked when the runner was prepared. Price windows are fixed by elapsed time; missing observations do not shorten lookbacks. Only older realized prices become predictors; the future48h path is used solely for Y.')
txt('Native missing handling means HGB has235inputs, not470. Ridge/LightGBM comparison helpers may append missing flags during preprocessing; their named feature-pack counts refer to source inputs.','small')

page('Other models and hyperparameter searches','All scores use the same two targets and evaluation rows. Model settings were chosen on earlier validation, not by the pooled ranking.')
table(['Family','Settings / architecture','Inputs'],[
 ['Mean / median','Constant training-period mean or median, recomputed at each issuance.','None'],
 ['Fixed Ridge','L2linear regression;alpha=.01*n;training-only imputation/scaling;nonnegative forecast projection.','UP:B01,B02,B04. DOWN:B01,B02,B04,B05,B06,B07,B08.'],
 ['Tuned Ridge','alpha/n in{.001,.01,.1,1};select pack and alpha by pooled earlierMSE.','original8;package7;engineered235.'],
 ['Direct LightGBM','L2boosted trees;3depth/leaves/rate settings:(3,4,.01),(3,4,.03),(4,8,.02).','engineered235'],
 ['E2-style','Fixed Ridge plus LightGBM correction of monthly out-of-fold Ridge errors. Monthly Ridge uses all valid paths,not endpoint-side subsets.','Fixed Ridge inputs +235correction inputs'],
 ['HGB L2 / L1','Squared-error / absolute-error histogram boosting;small and medium settings;no random earlystop.','raw13;raw13+51;235;235+51.']],[125,360,WIDTH-485])
txt('LightGBM common settings:1000tree cap,50round patience,latest20% chronological early-stop slice with48hpurge,then full-history refit;min_child_samples30,row/column sampling.75,row sampling everyiteration,L1=2,L2=5,seed48. Training-only affine target/residual normalization and imputation/scaling.','small')
txt('HGB alternative medium setting:depth3,max7leaves,minleaf64,120iterations,rate.05,L2=5,max_bins127,seed48. Chosen small setting:depth2,max4leaves,minleaf128,80iterations,L2=10. Both loss families used the same two setting candidates.','small')
txt('original8=B01-B08. package7=B01,B02,B09,B12,B13,B10,B11. raw13=B01-B13. The prior selector includes the mean baseline;HGB L1 fallback compares against median,while HGB L2 fallback compares against mean.','small')

page('Additional51features tested | not in the chosen model','The price block was tested in64-input and286-input packs. It was not retained in the235-feature working reference.')
table(['At each1h,4h,12h,24h,48h,168h','Calculation on strictly earlier5m returns'],[
 ['rv / up_rv / down_rv','Square root of sum of squared total/positive/negative returns.'],
 ['net','Signed sum of the returns in the window.'],
 ['past_range','Highest minus lowest cumulative level of the explicit n+1point prior5mclose path.'],
 ['distance_from_high','Highest prior-path level minus latest eligible prior level.'],
 ['distance_from_low','Latest eligible prior level minus lowest prior-path level.'],
 ['max_abs_return','Maximum absolute5mreturn in the window.'],
 ['Three extra ratios','Time-normalized RV ratios1h/24h,4h/24h,24h/168h;denominator floored at1e-6.']],[290,WIDTH-290])
txt('8measures x6horizons +3ratios =51columns. These include new path-position information as well as representations correlated with existingE2 inputs. They are not51independent market signals.','small')
table(['L1HGB pack','UP MAE','DOWN MAE'],[[label,f(row(U,m).mae_log_bps),f(row(D,m).mae_log_bps)] for m,label in [
 ('raw13_mae','13underlying inputs'),('raw13_price51_mae','13underlying +51price'),
 ('engineered235_mae','235engineered [chosen]'),('engineered235_price51_mae','235engineered +51price')]],[WIDTH-240,120,120])
txt('Adding51helps the13-input L1models, but slightly worsens both235-input L1models. These comparisons combine the feature block with earlier-selected settings;they do not isolate individual feature contributions or establish causality.','small')

for target in [U,D]:
    page(side(target)+' | all models ranked by MAE','Pooled18,385origins. MAE primary;RMSE rank also shown. L1=absolute-error training;L2=squared-error training.')
    g=pooled[pooled.target.eq(target)].sort_values(['mae_log_bps','model'])
    table(['MAE rank','Model / policy','MAE','RMSE','RMSE rank','Rank IC','Bias'],[
        [r.mae_rank,names[r.model],f(r.mae_log_bps),f(r.rmse_log_bps),r.rmse_rank,f(r.rank_ic),f(r.bias_log_bps)] for r in g.itertuples()],
        [55,WIDTH-385,67,67,62,67,67])
    txt('Ties use competition rank(minimum tied rank);display rounded to3decimals,rank computed from saved full precision. Policies are included as separate named procedures even when forecasts coincide. DOWN hgb_mae and selected_mae are identical in this run.','small')
    txt('Pack-specific HGB rows fix the feature pack but select small/medium using earlier validation each quarter. Cross-pack selectors also choose the pack. Comparing all outer scores after completion is exploratory;rank1is not a fresh-data confirmation.','small')

page('Chosen MAE model | period-by-period results','Chosen policy=engineered235_mae. Both target forecasts are evaluated on every retained origin.')
table(['Target','Period','Rows','MAE','Median MAE','Prior E2 MAE','RMSE','Rank IC'],[
 [side(t),period,int(row(t,'engineered235_mae',period).rows),f(row(t,'engineered235_mae',period).mae_log_bps),
  f(row(t,'prior_median',period).mae_log_bps),f(row(t,'prior_e2',period).mae_log_bps),
  f(row(t,'engineered235_mae',period).rmse_log_bps),f(row(t,'engineered235_mae',period).rank_ic)]
 for t in [U,D] for period in ['Cycle 1','Cycle 2','Cycle 3','Cycle 4','Jan2025']],[60,90,60,90,100,110,90,WIDTH-600])
txt('Cycle1=Jan-Mar2024;Cycle2=Apr-Jun;Cycle3=Jul-Sep;Cycle4=Oct-Dec. January2025is the tail under the October model. Period row counts reflect canonical exclusions;all models share each period\'s rows.','small')
table(['Target','Pooled MAE gain vs median','Pooled MAE gain vs prior E2','Tradeoff'],[
 [side(t),f(row(t,'prior_median').mae_log_bps-row(t,'engineered235_mae').mae_log_bps),
  f(row(t,'prior_e2').mae_log_bps-row(t,'engineered235_mae').mae_log_bps),
  'RMSE remains above prior best '+f(row(t,'prior_selected' if t==U else 'prior_fixed_ridge').rmse_log_bps)] for t in [U,D]],
 [70,180,180,WIDTH-430])
txt('A lower MAE does not imply lower tail risk or better trading performance. Bias below zero means average underprediction. Median-oriented outputs can have negative mean bias even when optimizing MAE correctly.','small')

page('Policy selections and interpretation','The chosen fixed235pack differs from the earlier cross-pack selection policy.')
table(['Target','Issue','HGB MAE cross-pack choice','HGB RMSE cross-pack choice','Chosen235 MAE setting'],[
 [side(t),issue[:10],next(s['candidate']['pack'] for s in sb if s['target']==t and s['issue']==issue and s['policy']=='hgb_mae'),
 next(s['candidate']['pack'] for s in sb if s['target']==t and s['issue']==issue and s['policy']=='hgb_rmse'),'small']
 for t in [U,D] for issue in sorted({s['issue'] for s in sb})],[65,100,215,215,WIDTH-595])
txt('User decision','head')
txt('After reviewing all results, the user chose MAE as primary and the235-feature absolute-error HGB as the working reference. That is a post-comparison research decision. Future changes should be selected with earlier-validation MAE,while retaining RMSE,bias andIC as diagnostics.')
txt('What this report establishes','head')
txt('Saved runs completed and match their prediction hashes;chosen model parameters and feature ordering were inspected;all sourceA comparison metrics carried into sourceB match to numerical tolerance. It does not independently recompute errors from predictions,retrain any model,prove significance,or test unseen data.')
txt('Every saved outer model/policy is ranked, but individual inner hyperparameter trials are not independent out-of-sample models. All recorded inner trials are included in the companion CSVs rather than misranked against outer scores.','small')

for start,end in [(0,7),(7,13)]:
    page('Appendix | exact underlying feature names',f'Input IDs B{start+1:02d}-B{end:02d};these names expand the195column formula map.')
    table(['ID','Exact source column'],[[ids[c],c] for c in raw[start:end]],[55,WIDTH-55],codecol=1)
    txt('For each B_j,prepend each operation plus double underscore:','head')
    txt('lag1;std7,range7,iqr7,zscore7;std14,range14,iqr14,zscore14;std30,range30,iqr30,zscore30;ma_ratio7_30;ma_ratio14_30.','small')
    txt('This yields195names. Add weekday_0..6,week_of_month_1..5,quarter_1..4 and the24LONG_*names shown earlier for exactly235. The exact_235_feature_manifest.csv enumerates all columns in model order.','small')
    if start==7:
        txt('The raw thirteen inputs are not fed directly to the chosen HGB. They are used to create the195lagged/rolling columns. The calendar and long-price columns are then appended. This distinction matters when reproducing saved model inference.','head')

page('Sources and reproducibility package','Workspace-relative paths; saved source hashes accompany the report.')
table(['Source','Path / contents'],[
 ['A | original48h excursion run',str(A.relative_to(ROOT))+' / metrics.csv,CONTRACT.json,SELECTIONS.json,COMPLETE.json,predictions.csv'],
 ['B | histogram48h excursion run',str(B.relative_to(ROOT))+' / metrics.csv,CONTRACT.json,SELECTIONS.json,COMPLETE.json,predictions.csv;8chosen models'],
 ['Feature engineering','Source snapshots: tune_lightgbm_residual_48h.py;lightgbm_repair_experiments_48h.py;canonical_return_features.py;improve_excursion_features_4h.py.'],
 ['Canonical definitions','Handoff2 FEATURE_DICTIONARY.json and team_tools/TEAM_DATASET_SPEC.md. Upstream F4Z formulas are source-defined;historical masked values were loaded.'],
 ['Companion rankings','pooled_rankings_full_precision.csv;all_period_scores.csv. Includes all19models/policies,all periods and both targets.'],
 ['Companion features','exact_235_feature_manifest.csv;raw_feature_dictionary.json.'],
 ['Companion model/selection evidence','chosen_model_parameters.json;both experiment contracts;both selections;both earlier-validation trial tables.'],
 ['Provenance','SOURCE_MANIFEST.json:input hashes;REPORT_QA.json:page count,render review and report-generation scope.']],[160,WIDTH-160])
txt('Preservation:raw datasets and existing results are unchanged. Model serialization was read for parameters/tree counts only;no new predictions or fitting. The report copies saved numerical scores with full-precision companions and3decimal display. Forecast timestamps and units are not changed.','small')

def footer(canvas,doc):
    canvas.setStrokeColor(TEAL);canvas.line(38,30,W-38,30)
    canvas.setFont('Helvetica',8);canvas.setFillColor(NAVY)
    canvas.drawString(38,18,'ScaleWave | BTC 48h excursions | Historical research | 13 Nov 2026')
    canvas.drawRightString(W-38,18,str(doc.page))
pdf=OUT/'BTC_48h_Excursion_Models_Features_Rankings_2026-11-13.pdf'
SimpleDocTemplate(str(pdf),pagesize=(W,H),rightMargin=38,leftMargin=38,topMargin=30,bottomMargin=42,
    title='BTC48h excursion models,feature formulas and MAE rankings',author='ScaleWave research').build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(str(pdf));assert len(reader.pages)>=15
text='\n'.join(p.extract_text() for p in reader.pages)
for needle in ['182.097','189.994','235','80','absolute_error']:assert needle in text
(SUP/'SOURCE_MANIFEST.json').write_text(json.dumps(sources,indent=2))
(SUP/'REPORT_QA.json').write_text(json.dumps(dict(pages=len(reader.pages),prediction_hashes_verified=True,
    prior_score_carryover_checked=True,chosen_model_hashes_verified=8,models_fitted=0,metrics_recomputed=False,
    rendered_review='pending'),indent=2))
bundle=OUT/'BTC_48h_Excursion_Report_Data_2026-11-13.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in SUP.iterdir():
        if p.is_file():z.write(p,p.name)
print(json.dumps(dict(pdf=str(pdf),pages=len(reader.pages),bundle=str(bundle))))
