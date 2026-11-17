"""Publish the completed fixed four-feature comparison without changing models."""
from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/rf_fixed_liquidation_1h_v1'
done=json.loads((out/'COMPLETE.json').read_text())
assert done['status']=='EXECUTED_AND_AUDITED' and done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
w=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
lines=['Fixed 1h liquidation feature experiment completed and audited: 10 unique models support 16 window artifacts.',
       'Features: original 5m and 15m liquidation historical-active-rate inputs, 4h/24h volatility ratio, and 1h liquidation historical active rate.',
       'Models: original median regression and depth-3 random forest blend; same parameters, expanding training and prior-quarter blend selection. No calibration.',
       '']
for name,row in test.iterrows():lines.append(f'{name}: test MAE {row.mae:.6f} bps; Spearman IC {row.rank_ic:.6f}; Pearson IC {row.pearson_ic:.6f}.')
for ref in ['baseline','vol_ratio']:
    lines.append(f'Four-feature MAE minus {ref}: {test.loc["rf_plus_liq1h","mae"]-test.loc[ref,"mae"]:+.6f} bps; wins in {int((w.rf_plus_liq1h<w[ref]).sum())}/4 windows.')
lines+=['','This is an exploratory comparison on already-examined history. Weekly block-bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool overlapping expanding windows. No fresh confirmation or deployment.']
text='\n'.join(lines)+'\n'
(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## Fixed 1h liquidation regression/RF result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/rf_fixed_liquidation_1h_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `AUDIT.json`, `COMPLETE.json`, `CONTRACT.json`, `MANIFEST.json`, `SELECTIONS.json`, and saved model checkpoints. Runner: `rf_fixed_liquidation_1h.py`. Viewer: `BTC_48h_RF_Fixed_Liquidation_1h.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Fixed 1h liquidation addition to regression/RF\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/rf_fixed_liquidation_1h_v1')\nfor name in ['scores','window_scores','extreme_scores','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_RF_Fixed_Liquidation_1h.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
