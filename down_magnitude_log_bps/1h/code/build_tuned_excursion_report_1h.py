"""Saved-artifact report only; no training or model inference."""
from pathlib import Path
import json, hashlib, zipfile
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
old_source = (ROOT/'Code/build_excursion_report_1h.py').read_text()
old_source = old_source.replace('Original 48h split gap remains absent.', 'The original 48h gap between initial and development data is retained.')
old_source = old_source.replace('Let r be a five-minute log return and L its cumulative log-price level.', 'Let r be a five-minute log return in log bps and L its cumulative level in the same units; L is not a dollar price.')
old_source = old_source.replace('Incomplete windows stay missing and are handled by training-fitted preprocessing.', 'Incomplete windows stay missing: the previous models use training-fitted imputation; tuned candidates use native missing-value handling. Ratios are unitless; the other measures are in log bps.')
# Reuse the established report typography and evidence definitions, without
# executing the historical report's authoring sections or replacing its output.
prefix = old_source.split("title(1,'Models and results')")[0]
prefix = prefix.replace('ScaleWave_1h_Excursion_Models_Report.pdf', 'ScaleWave_1h_Tuned_Models_Report.pdf').replace("Report/one_hour_excursion_report_sources", "Report/one_hour_tuned_report_sources")
exec(prefix)
TUNE=ROOT/'Code/experiments/excursion_tuning_1h_v1/20261113T113033200467Z'
tm=pd.read_csv(TUNE/'metrics.csv')
done=json.loads((TUNE/'COMPLETE.json').read_text())
assert done['status']=='COMPLETE' and done['development_origins']==18481
assert hashlib.sha256((TUNE/'predictions.csv').read_bytes()).hexdigest()==done['predictions_sha256']
tp=pd.read_csv(TUNE/'predictions.csv'); fp=pd.read_csv(FOLLOW/'predictions.csv')
keys=['decision_time','target']
joined=tp.merge(fp,on=keys,suffixes=('_t','_f'),validate='one_to_one')
assert len(joined)==36962
np.testing.assert_allclose(joined.actual_t,joined.actual_f,rtol=0,atol=1e-10)
np.testing.assert_allclose(joined.current,joined.selected,rtol=0,atol=1e-10)
manifest={}
def record(path): manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
for directory in [SEARCH,FOLLOW,TUNE]:
    for name in ['COMPLETE.json','CONTRACT.json','predictions.csv']: record(directory/name)
for path in [FOLLOW/'metrics_all_variants.csv',FOLLOW/'selections.json',TUNE/'metrics.csv',TUNE/'tune_excursions_1h.py',FOLLOW/'improve_excursion_features_1h.py',FOLLOW/'discover_excursion_features_1h.py']: record(path)
cards=[]; feature_rows=[]; params=[]
for folder in sorted(TUNE.iterdir()):
    if not folder.is_dir() or not (folder/'selection.json').exists(): continue
    selection=json.loads((folder/'selection.json').read_text()); record(folder/'selection.json')
    issue=folder.name[:7]; target=folder.name[8:]
    ref=next(s for s in sels if s['issue'][:7]==issue and s['target']==target)
    for policy,choice in selection['choices'].items():
        spec=selection['specs'][choice['candidate']]
        model_path=folder/(choice['candidate']+'.joblib')
        expected=json.loads((folder/(choice['candidate']+'_HASH.json')).read_text())['sha256']
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==expected
        bundle=joblib.load(model_path); assert bundle['spec']==spec
        par=bundle['model'].get_params(); params.append(dict(issue=issue,target=target,policy=policy,parameters=par));record(model_path)
        cards.append(dict(issue=issue,target=target,policy=policy,choice=choice,spec=spec,reference_features=ref['features']))
        for component,cols in [('candidate',spec['features']),('reference',ref['features'] if choice['weight']<1 else [])]:
            for order,col in enumerate(cols): feature_rows.append(dict(issue=issue,target=target,policy=policy,component=component,order=order,feature=col))
pd.DataFrame(feature_rows).to_csv(SUP/'exact_tuned_features.csv',index=False)
(SUP/'fitted_hyperparameters.json').write_text(json.dumps(params,indent=2,default=str))

title(1,'One-hour models: updated results')
para('13 November 2026 | Completed feature search, price-feature follow-up and subsequent tuning. Separate models predict the maximum UP and DOWN path excursions over the next hour. This report includes the latest completed tuning run, not only the earlier price-feature experiment.')
table(['Target','Previous MAE','Tuned MAE','Tuned RMSE','Rank IC'],[[side(t),num(tm[(tm.target==t)&(tm.period=='ALL')&(tm.model=='current')].iloc[0].mae),num(r.mae),num(r.rmse),num(r.ic)] for t in ['up_magnitude_log_bps','down_magnitude_log_bps'] for _,r in tm[(tm.target==t)&(tm.period=='ALL')&(tm.model=='mae_selected')].iterrows()],[60,110,110,110,110])
para('<b>MAE-focused tuning:</b> UP MAE falls 0.31% and DOWN 0.34% versus the prior selected models. Both improve MAE in all four 2024 quarters and January 2025. Pooled RMSE also improves, although DOWN RMSE worsens in Q4 and January.')
para('<b>RMSE-focused tuning:</b> UP RMSE is 38.0863 and DOWN 42.6444, reductions of 3.68% and 3.85%. Their MAE rises to 24.2032 and 25.5833. The two objectives produce different model rankings.')
para('All errors are in <b>log basis points</b>. Every pooled row covers 18,481 origins per target. Models refit quarterly; January 2025 uses the October fit. Historical evaluation has already been exposed. The small improvements have not been significance-tested.')
table(['What is included','Scope'],[['Latest model family','Histogram gradient boosted regression trees; some predictions blend equally with the previous model.'],['Full ranking','All full-coverage price-follow-up variants plus two new validation-selected tuning policies; duplicate current controls shown once.'],['Feature details','Exact original column dictionary, 51 past-price formulas, per-cycle component membership, and companion CSV.'],['Verification','Three prediction hashes; exact follow-up/tuning keys, actuals and current forecasts; tuned model hashes and stored estimator settings. No training or inference for this report.']],[135,365])

# Preserve the established, source-inspected target and initial-search account.
exec(old_source[old_source.index("title(2,'Targets and training design')"):old_source.index("title(3,'Architecture and hyperparameters')")])

title(3,'Model architecture and settings')
para('<b>New candidate:</b> ordered raw features -&gt; histogram binning learned on training rows -&gt; additive boosted decision trees -&gt; max(0, prediction). Missing values use native tree handling. No standardization, imputation or extra missing indicators are added to these new candidates.')
para('<b>Previous reference:</b> training-median imputation + one missing indicator per input -&gt; training-only centering/scaling -&gt; 80 MAE boosting trees -&gt; nonnegative clipping. A selected blend is w x new candidate + (1-w) x previous reference, where w is 0.5 or 1.0.')
table(['Hyperparameter','New candidates','Previous reference'],[['Estimator','HistGradientBoostingRegressor','Same family'],['Loss','absolute_error or squared_error','absolute_error'],['Learning rate','0.04','0.05'],['Iterations','120 / 160 / 160 by S0/S1/S2','80'],['Depth / maximum leaves','S0: 2/4; S1: 3/7; S2: 4/12','3 / 7'],['Minimum leaf samples','S0:128; S1:64; S2:128','128'],['L2 penalty','S0:10; S1:5; S2:10','1'],['Histogram max bins','127','255 (estimator default)'],['Early stopping / seed','False / 48','False / 48'],['Feature fraction / constraints','1.0; no monotonic or interaction constraints','1.0; no constraints'],['Training threads','2','2']],[155,190,155])
para('An ensemble adds a small tree at each iteration, scaled by the learning rate. Absolute-error fitting targets a conditional median; squared-error fitting targets a conditional mean. This helps explain why lowering RMSE can increase MAE. All tuned MAE selections use S2; RMSE selections use S0, S1 or S2 as listed later.')
para('Ridge and a training median were first-stage alternatives; the saved selected reference models are all MAE boosting trees. Ridge alpha was 0.01 x training row count. Mean/median baselines are side-specific constants from mature training labels, changing with quarterly refits.')

exec(old_source[old_source.index("title(4,'How the 51 new features were made')"):old_source.index("for n,target in [(5,")])

title(5,'Tuning and evaluation metrics')
para('Tuning tests three feature packs: previous selected inputs; previous inputs plus all 51 price inputs; and price-only. Each pack is fitted with two losses and three settings (18 candidates), and assessed alone and in a 50/50 reference blend. No additional feature formulas were introduced by this tuning run.')
table(['Selection policy','Earlier-data rule'],[['MAE-selected','At least 0.1% pooled validation MAE reduction, wins in at least 2 of 3 earlier windows, and pooled validation MSE no worse than current.'],['RMSE-selected','At least 0.1% pooled validation MSE reduction and wins in at least 2 of 3 windows. No MAE guard.'],['Fallback','Retain previous selected predictions if no candidate qualifies.'],['Controls and freeze','Refit reference on each inner training window. Write chosen settings before outer fitting/scoring. Native binning is fitted on training rows only.']],[130,370])
table(['Metric','Formula / meaning','Rank direction'],[['MAE','mean(abs(actual - prediction)); average error size','Lower'],['MSE','mean((actual - prediction)^2); inner RMSE-policy objective','Lower'],['RMSE','sqrt(MSE); emphasizes large misses','Lower'],['Rank IC','Spearman correlation of actual and predicted ranks; measures ordering','Higher']],[70,350,80])
para('Pooled metrics use all observations, not the simple average of quarterly metrics. Validation-window scores are weighted by row count. The report ranks pooled MAE, RMSE and IC separately; no single metric dominates every objective. A pooled baseline IC can be slightly nonzero because its training constant changes by quarter; a truly constant within-period prediction has undefined IC.')
para('The same earlier windows were used for prior feature selection, so this is adaptive reuse, not fully nested validation. Historical outer rankings are descriptive. Validation guards do not guarantee improvements on future data; no trading or profitability metrics were evaluated.')

names={'selected':'Previous validation-selected','mae_selected':'Tuned MAE-selected','rmse_selected':'Tuned RMSE-selected'}
for n,target in [(6,'up_magnitude_log_bps'),(7,'down_magnitude_log_bps')]:
    title(n,f'{side(target)} models ranked')
    m=metrics[(metrics.target==target)&(metrics.period=='ALL')&metrics.full_period_coverage.astype(bool)&metrics.rows.eq(18481)][['model','rows','mae','rmse','ic']].copy()
    new=tm[(tm.target==target)&(tm.period=='ALL')&tm.model.isin(['mae_selected','rmse_selected'])][['model','rows','mae','rmse','ic']]
    m=pd.concat([m,new],ignore_index=True)
    for field,ascending in [('mae',True),('rmse',True),('ic',False)]:m[field+'_rank']=m[field].rank(method='min',ascending=ascending).astype(int)
    m=m.sort_values('mae');m.to_csv(SUP/f'{side(target)}_model_ranking.csv',index=False)
    para('Same 18,481 origins and canonical one-hour target. Ordered by MAE; parentheses give separate RMSE/IC ranks. Strategies combine quarterly fits, not one fixed estimator.')
    table(['Rank','Model / feature strategy','MAE','RMSE (rank)','IC (rank)'],[[r.mae_rank,names.get(r.model,label(r.model)),num(r.mae),f'{r.rmse:.3f} ({r.rmse_rank})',f'{r.ic:.3f} ({r.ic_rank})'] for r in m.itertuples()],[35,230,65,85,85])
    para('Previous selected equals tuning current and is listed only once. Original equals the first-stage F4Z selection. Fixed feature variants are hindsight comparisons; tuned policies were chosen using earlier validation. Partial-coverage combinations and older endpoint-magnitude targets are excluded. Unselected tuning candidates have validation scores only and cannot receive an outer-test rank.')

title(8,'Latest policies by evaluation period')
for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
    para(f'<b>{side(target)}</b> - current, tuned MAE policy and tuned RMSE policy')
    rows=[]
    for period in ['2024Q1','2024Q2','2024Q3','2024Q4','2025-01','ALL']:
        g=tm[(tm.target==target)&(tm.period==period)].set_index('model')
        rows.append([period,*[f'{g.loc[m,"mae"]:.3f} / {g.loc[m,"rmse"]:.3f}' for m in ['current','mae_selected','rmse_selected']]])
    table(['Period','Current MAE / RMSE','MAE policy MAE / RMSE','RMSE policy MAE / RMSE'],rows,[65,145,145,145])
para('MAE tuning improves both sides in every period. UP RMSE improves in every period; DOWN RMSE worsens in Q4 and January. RMSE tuning improves RMSE in every period for both sides, but its MAE worsens in every period. Full-precision metrics, including period IC, are in the companion data.')

original=sorted({c for c in pd.DataFrame(feature_rows).feature if c.startswith('F4Z__')})
ids={c:f'F{i+1:02d}' for i,c in enumerate(original)}
title(9,'Exact original input dictionary')
para('These source-column IDs are used in the membership tables. RZ is the supplied robust score; percentile and historical-active-rate columns are distinct measurements. The new tuning run does not recalculate these F4Z fields.')
table(['ID','Exact source column'],[[ids[c],c] for c in original],[40,460])

def describe(cols):
    f=', '.join(ids[c] for c in cols if c in ids) or 'No F4Z'
    new=[c for c in cols if c.startswith('NEWPRICE__')]
    if len(new)==51:return f+' + all 51 price'
    if not new:return f+'; no price inputs'
    groups=json.loads((FOLLOW/'CONTRACT.json').read_text())['groups']
    blocks=[k.replace('price_','') for k,v in groups.items() if set(v).issubset(new)]
    covered={c for k,v in groups.items() if set(v).issubset(new) for c in v}
    assert covered==set(new)
    return f+' + price '+', '.join(blocks)

for number,policy in [(10,'mae_selected'),(11,'rmse_selected')]:
    title(number,'MAE-policy feature membership' if policy=='mae_selected' else 'RMSE-policy feature membership')
    para('Candidate features are exact sets abbreviated with dictionary IDs and complete price blocks from page 4. S0/S1/S2 settings are on page 3. Weight is the new-candidate share; the remaining share uses the previous reference.')
    subset=[c for c in cards if c['policy']==policy]
    table(['Fit / side','Raw count','Setting / weight','Candidate input set'],[[c['issue']+' '+side(c['target']),len(c['spec']['features']),f'S{c["spec"]["setting"]} / {c["choice"]["weight"]:g}',describe(c['spec']['features'])] for c in subset],[90,45,85,280])
    para('<b>Reference components used in blends</b>')
    for c in subset:
        if c['choice']['weight']<1:para(c['issue']+' '+side(c['target'])+': '+describe(c['reference_features'])+f' ({len(c["reference_features"])} raw inputs).')
    para('October fits also predict January 2025. Native candidate input counts equal raw counts; the previous-reference component doubles its raw count through missing indicators. Exact ordered columns for each component are in exact_tuned_features.csv. All MAE-policy candidates use absolute-error loss; all RMSE-policy candidates use squared-error loss.')

title(12,'Evidence and reproducibility')
para('Source paths are relative to November/Week 2. The companion archive contains full-precision rankings, period metrics, ordered feature membership, stored model hyperparameters and source SHA256 hashes.')
table(['Stage','Saved run'],[['Initial and price follow-up','Code/experiments/excursion_pipeline_1h_v1/20261113T101316741479Z; search/20261113T101328048744Z and price_followup/20261113T104924262943Z'],['New tuning','Code/experiments/excursion_tuning_1h_v1/20261113T113033200467Z'],['Builder','Code/build_tuned_excursion_report_1h.py']],[110,390])
para('Architecture and formulas were read from saved source snapshots. Tuned estimator parameters were inspected from hash-verified saved bundles. Three prediction files match completion hashes. Follow-up/tuning key sets, targets and carried current forecasts agree. Completion records report successful model replay; this report does not rerun model inference or independently recompute metric tables.')
para('No new training was performed for the report. Earlier one-hour endpoint-magnitude experiments use a different target and are not included. Feature-reliance rankings from the earlier October model are not reused as if they described the newly tuned models.')
para('Historical data are repeatedly exposed, adjacent forecast horizons overlap, and feature choices reuse earlier validation. Results do not establish statistical significance, fresh-test performance or trading profitability. Original feature availability and provisional-source limitations remain.')
tm.to_csv(SUP/'tuning_metrics.csv',index=False)
metrics[(metrics.period=='ALL')&~metrics.full_period_coverage.astype(bool)].to_csv(SUP/'partial_coverage_variants.csv',index=False)
(SUP/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607b95'))
    canvas.drawString(42,24,'ScaleWave | Updated 1-hour excursion models | 13 Nov 2026')
    canvas.drawRightString(553,24,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(SUP.iterdir()):
        if path.is_file():archive.write(path,path.name)
print(OUT)
