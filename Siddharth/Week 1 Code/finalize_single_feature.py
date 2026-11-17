from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/baseline_single_feature_v6'
r=json.loads((OUT/'COMPLETE.json').read_text())
assert r['status']=='EXECUTED_AND_AUDITED' and r['audit']['status']=='PASS'
contract=json.loads((OUT/'CONTRACT.json').read_text());decisions=json.loads((OUT/'SELECTIONS.json').read_text())
scores=pd.read_csv(OUT/'scores.csv');test=scores[scores.split=='test'].copy()
labels={'baseline':'Original baseline','selected':'Validation-selected feature strategy','rv4h':'+ 4h realized volatility','rv24h':'+ 24h realized volatility','rv7d':'+ 7d realized volatility','past48h_move':'+ Past 48h absolute move','vol_ratio':'+ 4h/24h volatility ratio','funding_persistence':'+ Funding persistence'}
test['description']=test.model.map(labels);base=test.loc[test.model=='baseline','mae'].iloc[0]
test['mae_difference_vs_baseline']=test.mae-base
test.to_csv(OUT/'comparison.csv',index=False)
selection=[];trials=[]
for d in decisions:
 selected=d['selected'];chosen=next((t for t in d['candidates'] if t['name']==selected),None)
 selection.append(dict(issue=d['issue'],selected=selected,description=labels[selected],added_feature=chosen['feature'] if chosen else '',weight=chosen['weight'] if chosen else d['baseline_weight'],validation_gain_bps=chosen['gain_bps'] if chosen else 0,monthly_wins=chosen['monthly_wins'] if chosen else 0))
 for row in d['candidates']:trials.append(dict(issue=d['issue'],**row))
selection=pd.DataFrame(selection);selection.to_csv(OUT/'selected_features.csv',index=False)
pd.DataFrame(trials).to_csv(OUT/'all_validation_trials.csv',index=False)
p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
groups=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
uncertainty=[]
for model in test.model:
 delta=abs(p[model]-p.truth)-abs(p.baseline-p.truth)
 blocks=pd.DataFrame(dict(delta=delta.to_numpy(),group=groups)).groupby('group').delta.agg(['sum','count'])
 ids=np.random.default_rng(2026).integers(0,len(blocks),(2000,len(blocks)))
 boot=blocks['sum'].to_numpy()[ids].sum(1)/blocks['count'].to_numpy()[ids].sum(1)
 ci=np.quantile(boot,[.025,.975]);uncertainty.append(dict(model=model,difference=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
text='Controlled baseline-plus-one-feature experiment: executed on real data and audited.\n\nBoth baseline components (median regression and random forest) retain their original model parameters and preprocessing recipe. Each challenger adds exactly one source feature to the original two liquidation features. Blend weights use the original prior-quarter MAE rule. Same four outer windows and 18,481 evaluation rows.\n\nThe primary strategy selects features only from earlier validation data, requiring >=0.1 bps gain and improvements in at least two validation months; otherwise it retains the baseline. All six fixed-feature strategies were specified before this run and are reported separately. Their test ranking must not be presented as a pre-test deployment selection.\n\n'
text+=test[['description','mae','rank_ic','pearson_ic','mae_difference_vs_baseline']].to_string(index=False)
text+='\n\nValidation-selected features:\n'+selection.to_string(index=False)
text+='\n\nPer-window results:\n'+pd.read_csv(OUT/'window_scores.csv').to_string(index=False)
text+='\n\nExploratory paired weekly-block bootstrap intervals for MAE differences:\n'+pd.DataFrame(uncertainty).to_string(index=False)
text+='\n\nAudit:\n'+json.dumps(r,indent=2)
text+='\n\nLimitations: this historical evaluation has already been examined; six feature alternatives and many previous experiments were explored. Bootstrap intervals are descriptive and are not adjusted for those searches. Untouched future data is required to confirm a reliable improvement. Canonical price features use return endpoints strictly before the forecast origin; the past 48h feature follows that conservative availability rule.\n'
(OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
 dict(id='feature-intro',cell_type='markdown',metadata={},source=['# Baseline plus one feature\n','Original two liquidation features plus one candidate at a time. Model recipe and outer walk-forward splits are unchanged. The primary strategy selects additions on validation data; fixed-feature test comparisons are exploratory.\n']),
 dict(id='feature-results',cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\n","import pandas as pd, json\n","run=Path('experiments/baseline_single_feature_v6')\n","display(pd.read_csv(run/'comparison.csv'))\n","display(pd.read_csv(run/'selected_features.csv'))\n","display(pd.read_csv(run/'window_scores.csv'))\n","display(pd.read_csv(run/'uncertainty.csv'))\n","print((run/'COMPLETE.json').read_text())\n"]),
 dict(id='feature-source',cell_type='markdown',metadata={},source=['Runner: baseline_single_feature.py. The run folder preserves the frozen protocol, input hashes, source snapshot, feature-causality audit, validation trials, selected features, all fitted models, predictions, and replay audit. This notebook displays saved results without starting training.'])])
for cell in nb['cells']:
 if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
(ROOT/'BTC_48h_Baseline_One_Feature.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
context=ROOT/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8');marker='## Baseline plus one feature result — 2026-11-04'
if marker not in old:
 note='\n\n'+marker+'\n\nUser authorized controlled feature addition to beat the baseline, relaxing the earlier two-input-only restriction for this experiment. Exactly one source feature was added to the original pair at a time: 4h/24h/7d realized volatility, past 48h absolute move, 4h/24h volatility ratio, or funding persistence. BOTH original baseline components were extended while retaining model parameters, preprocessing, blend-weight grid, training rows, and outer walk-forward splits. Primary selection used prior-quarter MAE and monthly consistency, including baseline fallback. All six fixed-feature strategies were preregistered and scored separately.\n\n'
 for model in ['baseline','selected']+list(contract['candidates']):
  a=r['test_scores'][model];note+=f"- {labels[model]}: MAE {a['mae']:.6f} bps; Spearman IC {a['rank_ic']:.6f}.\n"
 note+=f"\nPrimary strategy beats baseline: {r['primary_beats_baseline']}. Fixed-feature point-estimate wins: {r['fixed_feature_point_estimate_wins']}. These are historical exploratory results, not untouched holdout confirmation.\n"
 note+='\nExecuted on real data and audited: 96 fits; same 18,481 evaluation rows; label purges, training-only preprocessing, unchanged parameters, exact feature counts, baseline reconstruction, validation selection and saved-model replay checks passed. Canonical return source reproduced original labels; causal prefix, future-perturbation and manual past-window checks passed.\n\nSources: `experiments/baseline_single_feature_v6/COMPLETE.json`, `AUDIT.json`, `FEATURE_AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_features.csv`, `all_validation_trials.csv`, `uncertainty.csv`, and `window_scores.csv`. Runner: `baseline_single_feature.py`; results viewer: `BTC_48h_Baseline_One_Feature.ipynb`. Prior results and raw data remain unchanged.\n'
 context.write_text(old+note,encoding='utf-8')
print(test[['description','mae','rank_ic','pearson_ic','mae_difference_vs_baseline']].to_string(index=False))
print(selection.to_string(index=False))
print('Saved report, checked notebook code, and updated project context.')
