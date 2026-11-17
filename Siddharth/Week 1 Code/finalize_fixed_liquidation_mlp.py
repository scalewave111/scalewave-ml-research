from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/mlp_fixed_liquidation_1h_v1'
done=json.loads((out/'COMPLETE.json').read_text());assert done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
w=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
lines=['Fixed fourth-feature MLP executed and audited. Added 1h liquidation historical active rate to the original 5m/15m rates and 4h/24h volatility ratio in every window.',
    'Same linear-initialized 8/4 residual MLP and three seeds. 18 new neural fits plus six verified reused checkpoints; three new linear fits plus two reused. No feature gate or fallback.','']
for n,r in test.iterrows():lines.append(f'{n}: test MAE {r.mae:.6f} bps; Spearman IC {r.rank_ic:.6f}; Pearson IC {r.pearson_ic:.6f}.')
for ref in ['baseline','vol_ratio','mlp_3features','linear_4features']:
    lines.append(f'Four-feature MLP minus {ref} MAE: {test.loc["fixed_four_feature_mlp","mae"]-test.loc[ref,"mae"]:+.6f} bps; wins in {int((w.fixed_four_feature_mlp<w[ref]-1e-8).sum())}/4 windows.')
lines+=['','First three windows, excluding reused final window:\n'+pd.read_csv(out/'first_three_windows_scores.csv').to_string(index=False),
    '\nFeature chosen after observing its final-window success. This is a post-selection historical retest, not independent confirmation. Reused final-window results are not new evidence. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool repeated timestamps across expanding windows.']
text='\n'.join(lines)+'\n';(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## Fixed 1h liquidation fourth-feature MLP result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/mlp_fixed_liquidation_1h_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `first_three_windows_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `mlp_fixed_liquidation_1h.py`; viewer: `BTC_48h_MLP_Fixed_Liquidation_1h.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# Fixed 1h liquidation feature MLP\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/mlp_fixed_liquidation_1h_v1')\nfor name in ['scores','window_scores','first_three_windows_scores','seed_window_scores','extreme_scores','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_MLP_Fixed_Liquidation_1h.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
