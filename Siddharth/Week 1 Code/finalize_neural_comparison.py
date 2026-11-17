"""Save the completed neural comparison summary and local research context."""
from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/neural_snapshot_vs_sequence_v1'
done=json.loads((out/'COMPLETE.json').read_text());assert done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
windows=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
lines=['Executed and audited: residual MLP versus small custom PatchTST-style transformer, both with liquidation 5m/15m historical active rates and 4h/24h volatility ratio.',
       'Same four purged expanding walk-forward windows, three seeds, validation-only early stopping; 48 checkpoints. MLP uses current values; transformer uses 48 half-hour observations ending at prediction origin.',
       '']
for n,r in test.iterrows():lines.append(f'{n}: test MAE {r.mae:.6f} bps; Spearman IC {r.rank_ic:.6f}; Pearson IC {r.pearson_ic:.6f}.')
for n in ['residual_mlp','patch_transformer']:
    for ref in ['baseline','vol_ratio']:
        lines.append(f'{n} minus {ref} MAE: {test.loc[n,"mae"]-test.loc[ref,"mae"]:+.6f} bps; wins in {int((windows[n]<windows[ref]).sum())}/4 windows.')
best=test.loc[['residual_mlp','patch_transformer'],'mae'].idxmin()
lines+=['',f'Lowest neural test MAE in this experiment: {best}. Beats original baseline MAE: {bool(test.loc[best,"mae"]<test.loc["baseline","mae"])}. Beats three-feature regression/RF candidate MAE: {bool(test.loc[best,"mae"]<test.loc["vol_ratio","mae"])}.',
    'These are exploratory historical comparisons, not untouched confirmation. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores are in-sample and pool overlapping expanding windows. The custom patch transformer is not a full PatchTST reproduction; the comparison changes architecture and capacity as well as history access. No GARCH model was trained.']
text='\n'.join(lines)+'\n';(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## Neural snapshot versus sequence result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/neural_snapshot_vs_sequence_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `PREFLIGHT.json`, `AUDIT.json`, `COMPLETE.json`, and saved `.pt` checkpoints. Runner: `neural_snapshot_vs_sequence.py`. Viewer: `BTC_48h_Neural_Snapshot_vs_Sequence.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Neural snapshot versus sequence comparison\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/neural_snapshot_vs_sequence_v1')\nfor name in ['scores','window_scores','seed_window_scores','extreme_scores','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_Neural_Snapshot_vs_Sequence.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
