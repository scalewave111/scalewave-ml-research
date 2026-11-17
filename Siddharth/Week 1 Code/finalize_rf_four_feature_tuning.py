from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/rf_four_feature_tuning_v1'
done=json.loads((out/'COMPLETE.json').read_text());assert done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
w=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
selections=json.loads((out/'SELECTIONS.json').read_text());manifest=json.loads((out/'MANIFEST.json').read_text())
settings=[]
for d in selections:
    r=d['selected'];p=manifest['full_rf_parameters'][r['config']]
    settings.append(dict(issue=d['issue'],config=r['config'],rf_weight=r['weight'],validation_mae=r['mae'],fold_wins=r['fold_wins'],**{k:p[k] for k in ['n_estimators','max_depth','min_samples_leaf','max_features','max_samples','criterion','random_state']}))
pd.DataFrame(settings).to_csv(out/'selected_hyperparameters.csv',index=False)
lines=['Four-feature RF tuning executed and audited. Same fixed liquidation5m/15m/1h historical active rates and4h/24h volatility ratio.',
       'Eight RF configurations and six blend weights, selected using three earlier purged quarters per outer window; pooled MAE plus2/3-fold consistency gate. Median regression and preprocessing unchanged. No test-driven retries.','']
for n,r in test.iterrows():lines.append(f'{n}: test MAE {r.mae:.6f} bps; Spearman IC {r.rank_ic:.6f}; Pearson IC {r.pearson_ic:.6f}.')
for ref in ['baseline','vol_ratio','untuned_4features','nested_original_control']:
    lines.append(f'Tuned blend minus {ref} MAE: {test.loc["tuned_blend","mae"]-test.loc[ref,"mae"]:+.6f} bps; wins in {int((w.tuned_blend<w[ref]-1e-8).sum())}/4 windows.')
lines+=['','Selected settings:\n'+pd.DataFrame(settings).to_string(index=False),'',f'Beats original baseline MAE: {bool(test.loc["tuned_blend","mae"]<test.loc["baseline","mae"])}. Beats best three-feature regression/RF MAE: {bool(test.loc["tuned_blend","mae"]<test.loc["vol_ratio","mae"])}.',
    'Nested original-parameter control isolates RF/blend tuning from changing the validation history. Nested validation repeats historical rows across outer windows and is selection-used, so do not compare its pooled score with old one-quarter validation scores. Entire test history is exposed; this is not independent confirmation. Bootstrap intervals are descriptive and unadjusted for earlier searches.']
text='\n'.join(lines)+'\n';(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## RF tuning for fixed four-feature set result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/rf_four_feature_tuning_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `selected_hyperparameters.csv`, `all_validation_trials.csv`, `uncertainty.csv`, `SELECTIONS.json`, `FIT_LEDGER.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `rf_four_feature_tuning.py`; viewer: `BTC_48h_RF_Four_Feature_Tuning.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Four-feature RF hyperparameter tuning\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/rf_four_feature_tuning_v1')\nfor name in ['scores','window_scores','selected_hyperparameters','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_RF_Four_Feature_Tuning.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
