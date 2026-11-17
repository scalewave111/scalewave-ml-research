from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/linear_initialized_mlp_v1'
done=json.loads((out/'COMPLETE.json').read_text());assert done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
windows=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
lines=['Executed and audited: training-only fitted linear skip plus zero-initialized 8/4 residual MLP.',
       'Three inputs: liquidation 5m/15m historical active rates and 4h/24h volatility ratio. Same purged quarterly expanding walk-forward; three seeds. Frozen linear branch; nonlinear correction selected by earlier validation MAE.',
       '']
for n,r in test.iterrows():lines.append(f'{n}: test MAE {r.mae:.6f} bps; Spearman IC {r.rank_ic:.6f}; Pearson IC {r.pearson_ic:.6f}.')
for ref in ['baseline','vol_ratio','linear_only']:
    lines.append(f'Initialized MLP minus {ref}: {test.loc["initialized_mlp","mae"]-test.loc[ref,"mae"]:+.6f} bps; wins in {int((windows.initialized_mlp<windows[ref]-1e-8).sum())}/4 windows.')
lines+=['',f'Beats original baseline MAE: {bool(test.loc["initialized_mlp","mae"]<test.loc["baseline","mae"])}. Beats three-feature regression/RF MAE: {bool(test.loc["initialized_mlp","mae"]<test.loc["vol_ratio","mae"])}.',
    'No fresh holdout; historical comparisons are exploratory. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool overlapping expanding windows. The exact own-linear comparison tests added nonlinear correction; comparison with earlier softplus MLP also changes the output link and freezes the skip.']
text='\n'.join(lines)+'\n';(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## Linear-initialized three-feature MLP result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/linear_initialized_mlp_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json` and saved models. Runner: `linear_initialized_mlp.py`; viewer: `BTC_48h_Linear_Initialized_MLP.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Linear-initialized residual MLP\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/linear_initialized_mlp_v1')\nfor name in ['scores','window_scores','seed_window_scores','extreme_scores','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_Linear_Initialized_MLP.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
