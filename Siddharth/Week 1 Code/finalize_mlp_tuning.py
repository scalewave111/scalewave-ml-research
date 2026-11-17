from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/mlp_hyperparameter_tuning_v4'
r=json.loads((OUT/'COMPLETE.json').read_text())
assert r['status']=='EXECUTED_AND_AUDITED' and r['audit']['status']=='PASS'
decisions=json.loads((OUT/'SELECTIONS.json').read_text())
rows=[]
for d in decisions:
 c=d['selected']['config']
 rows.append(dict(issue=d['issue'],**c,parameters=d['selected']['parameters'],outer_epochs=d['outer_epochs'],validation_mae=d['selected']['mae'],validation_ic=d['selected']['rank_ic']))
chosen=pd.DataFrame(rows);chosen.to_csv(OUT/'selected_hyperparameters.csv',index=False)
comparison=pd.read_csv(ROOT/'experiments/mlp_two_liquidation_comparison/scores.csv')
m=r['test_scores']['mlp']
comparison=pd.concat([comparison,pd.DataFrame([dict(model='Hyperparameter-tuned MLP',mae_bps=m['mae'],spearman_ic=m['rank_ic'],pearson_ic=m['pearson_ic'],mae_difference_vs_baseline=r['mae_difference_mlp_minus_baseline'])])],ignore_index=True)
comparison.to_csv(OUT/'comparison.csv',index=False)
text='Small MLP hyperparameter tuning: executed on real data and audited.\n\nExactly two original liquidation inputs (5m and 15m historical active rates). Same four quarterly outer walk-forward windows and 18,481 evaluation rows.\n24 predeclared settings; three preceding purged monthly validation folds; seed48 screening; top3 settings confirmed with seeds149 and250; three-seed ensemble outer refits. MAE selects settings, with IC reported separately.\n\n'
text+=comparison.to_string(index=False)+'\n\nSelected hyperparameters:\n'+chosen.to_string(index=False)
text+='\n\nPer-window results:\n'+pd.read_csv(OUT/'window_scores.csv').to_string(index=False)
text+='\n\nAudit and uncertainty:\n'+json.dumps(r,indent=2)
text+='\n\nLimitations: previously examined historical evaluation, not an untouched holdout. Monthly inner validation differs from earlier experiments, so their validation MAEs are not directly comparable. No outer-test-driven retries were performed within this frozen search.\n'
(OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
 dict(id='tuning-summary',cell_type='markdown',metadata={},source=['# Two-feature small MLP hyperparameter tuning\n','Exactly the original 5m and 15m liquidation activity rates. Same outer walk-forward evaluation; three earlier monthly inner folds. MAE selects settings; Spearman and Pearson IC are reported. This historical period was previously examined.\n']),
 dict(id='tuning-results',cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\n","import pandas as pd, json\n","run=Path('experiments/mlp_hyperparameter_tuning_v4')\n","display(pd.read_csv(run/'comparison.csv'))\n","display(pd.read_csv(run/'selected_hyperparameters.csv'))\n","display(pd.read_csv(run/'window_scores.csv'))\n","print((run/'COMPLETE.json').read_text())\n"]),
 dict(id='tuning-source',cell_type='markdown',metadata={},source=['Runner: mlp_hyperparameter_tuning.py. Individual checkpoints support resumption; a completed run is preserved. This notebook views saved results without training. Run folder includes frozen configuration, source snapshot, manifests, per-window screening/confirmation records, selected models, predictions, and replay audit.'])])
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
(ROOT/'BTC_48h_MLP_Hyperparameter_Tuning.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
context=ROOT/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8');marker='## MLP hyperparameter tuning result — 2026-11-04'
if marker not in old:
 note='\n\n'+marker+'\n\nExecuted on real data and audited: 372 model checkpoints, 24 settings, three purged monthly inner folds per outer window, three seeds for shortlisted settings and outer refits. Exactly two original liquidation inputs; unchanged four outer windows and 18,481 test rows.\n\n'
 b=r['test_scores']['baseline']
 note+=f"- Baseline MAE {b['mae']:.6f} bps; Spearman IC {b['rank_ic']:.6f}.\n- Tuned MLP MAE {m['mae']:.6f} bps; Spearman IC {m['rank_ic']:.6f}.\n- Beats baseline MAE: {r['beats_baseline']}.\n"
 note+='\nSources: `experiments/mlp_hyperparameter_tuning_v4/COMPLETE.json`, `AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_hyperparameters.csv`, and `window_scores.csv`; results viewer `BTC_48h_MLP_Hyperparameter_Tuning.ipynb`. All purges, preprocessing, frozen selection, baseline reconstruction, and checkpoint replay checks passed. Earlier runs remain unchanged. Previously examined historical evaluation remains exploratory, not an untouched holdout.\n'
 context.write_text(old+note,encoding='utf-8')
print(comparison.to_string(index=False))
print(chosen.to_string(index=False))
print('Saved and checked report, notebook code, and project context.')
