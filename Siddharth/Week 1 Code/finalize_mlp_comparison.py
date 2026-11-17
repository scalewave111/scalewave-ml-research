"""Summarize completed, audited runs. Does not train or choose a new model."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/mlp_two_liquidation_comparison'
RUNS=[('Standard MLP','mlp_two_liquidation_v1'),('Initialized MLP','mlp_two_liquidation_warmstart_v2'),('Recency-weighted MLP','mlp_two_liquidation_recency_v3')]
rows=[];windows=[];selections=[];results={}
for label,name in RUNS:
 folder=ROOT/'experiments'/name
 result=json.loads((folder/'COMPLETE.json').read_text())
 assert result['status']=='EXECUTED_AND_AUDITED' and result['audit']['status']=='PASS'
 assert result['audit']['input_dimension']==2 and result['audit']['rows']==18481
 results[label]=result
 if not rows:
  b=result['test_scores']['baseline'];rows.append(dict(model='Baseline',mae_bps=b['mae'],spearman_ic=b['rank_ic'],pearson_ic=b['pearson_ic'],mae_difference_vs_baseline=0.))
 m=result['test_scores']['mlp']
 rows.append(dict(model=label,mae_bps=m['mae'],spearman_ic=m['rank_ic'],pearson_ic=m['pearson_ic'],mae_difference_vs_baseline=result['mae_difference_mlp_minus_baseline']))
 w=pd.read_csv(folder/'window_scores.csv');w['experiment']=label;windows.append(w)
 for s in json.loads((folder/'SELECTIONS.json').read_text()):
  c=s['selected'];selections.append(dict(experiment=label,issue=s['issue'],architecture=c['name'],parameters_per_network=c['parameters'],epochs=c['epochs'],validation_mae=c['mae']))
OUT.mkdir(exist_ok=True)
table=pd.DataFrame(rows);table.to_csv(OUT/'scores.csv',index=False)
pd.concat(windows).to_csv(OUT/'window_scores.csv',index=False)
pd.DataFrame(selections).to_csv(OUT/'selected_architectures.csv',index=False)
intro='Two original liquidation features only: 5m and 15m cross-asset liquidation historical active rates. Exactly two network inputs; no additional indicators or lags.\nSame four quarterly expanding walk-forward windows, strict 48-hour label maturity, same 18,481 evaluation rows. MAE selected configurations; Spearman IC is reported separately.\n\n'
limitations='\n\nThese are sequential exploratory experiments on previously examined historical evaluation data. They are not fresh holdout evidence. No source datasets or historical experiment results were overwritten. Three seeds are averaged per selected MLP. All saved-model replay audits passed.\nTraining duration excludes tool/setup overhead.\n'
details='\n\n'.join(label+': '+json.dumps(dict(beats_baseline=r['beats_baseline'],mae_difference=r['mae_difference_mlp_minus_baseline'],weekly_block_bootstrap_95pct_ci=r['weekly_block_bootstrap_95pct_ci'],training_and_audit_seconds=r['elapsed_seconds'])) for label,r in results.items())
(OUT/'RESULTS.txt').write_text(intro+table.to_string(index=False)+limitations+'\n'+details+'\n\nSelected architectures:\n'+pd.DataFrame(selections).to_string(index=False),encoding='utf-8')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
 dict(cell_type='markdown',metadata={},source=['# Two-feature small MLP comparison\n',intro,limitations]),
 dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\n","import pandas as pd\n","run = Path('experiments/mlp_two_liquidation_comparison')\n","display(pd.read_csv(run/'scores.csv'))\n","display(pd.read_csv(run/'window_scores.csv'))\n","display(pd.read_csv(run/'selected_architectures.csv'))\n"]),
 dict(cell_type='markdown',metadata={},source=['Runners: mlp_two_liquidation.py, mlp_two_liquidation_warmstart.py, mlp_two_liquidation_recency.py. Each run folder contains its frozen contract, full training source, selected checkpoints, predictions, and replay audit. This notebook displays results without starting training.'])])
for i,cell in enumerate(nb['cells']):cell['id']=f'mlp-comparison-{i}'
(ROOT/'BTC_48h_MLP_Two_Feature_Comparison.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
context=ROOT/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8')
marker='## Two-input MLP experiments — executed and audited, 2026-11-04'
if marker not in old:
 text='\n\n'+marker+'\n\nUser requested a small MLP using only the original 5m and 15m liquidation historical-active-rate features, preserving the walk-forward comparison. User explicitly removed the time constraint and requested IC as well as MAE. MAE remained the selection objective. Exactly two inputs were used, with training-only median imputation and scaling; no missingness flags or lagged/extra features. Three seeds were averaged.\n\n'
 for row in rows:text+=f"- {row['model']}: test MAE {row['mae_bps']:.6f} bps; Spearman IC {row['spearman_ic']:.6f}.\n"
 text+='\nAll three runs completed on real data and passed strict label-purge, feature-count, training-only preprocessing, baseline reconstruction, and checkpoint replay checks. The recency experiment changes training-example weights only, retaining every expanding-window row; dates are not network inputs. Model selection uses only each preceding purged validation quarter. Subsequent experiments and the evaluation period are exploratory, not an untouched holdout.\n\n'
 text+='Sources: `experiments/mlp_two_liquidation_comparison/RESULTS.txt`, `scores.csv`, `window_scores.csv`, `selected_architectures.csv`; individual runs `experiments/mlp_two_liquidation_v1/`, `experiments/mlp_two_liquidation_warmstart_v2/`, `experiments/mlp_two_liquidation_recency_v3/`. Runners share those names without the version suffix. Results viewer: `BTC_48h_MLP_Two_Feature_Comparison.ipynb`. Each folder contains COMPLETE.json, AUDIT.json, contract, source snapshot, predictions and models.\n'
 text+='\nCorrection to earlier in-progress LSTM note: `experiments/lstm_feature_architecture_v3/COMPLETE.json` records a completed audited run, with MAE 184.177232 versus baseline 181.525993; it did not win. `diagnose_lstm_features.py` was generated, but a completed diagnostics_v1 output was not observed before the user switched to MLP work. The previously stopped `lstm_optimize.py` remains an unexecuted draft.\n'
 context.write_text(old+text,encoding='utf-8')
print(table.to_string(index=False))
print('Saved comparison report, notebook, and project context.')
