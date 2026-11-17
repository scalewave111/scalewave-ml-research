from pathlib import Path
import json,subprocess,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/mlp_feature_discovery_v1'
done=json.loads((out/'COMPLETE.json').read_text());assert done['audit']['status']=='PASS'
scores=pd.read_csv(out/'scores.csv');test=scores.query("split=='test'").set_index('model')
windows=pd.read_csv(out/'window_scores.csv').pivot(index='issue',columns='model',values='mae')
decisions=json.loads((out/'SELECTIONS.json').read_text())
screen=json.loads((out/'SCREEN_COMPLETE.json').read_text())['decisions']
rows=[]
for d,s in zip(decisions,screen):
    assert d['issue']==s['issue'];w=windows.loc[d['issue']]
    rows.append(dict(issue=d['issue'],feature=d['feature'],screen_validation_gain=s['screen_result']['gain_bps'] if s['feature'] else None,
        actual_mlp_validation_gain=d.get('validation_gain'),validation_accepted=d['use_addition'],
        challenger_test_gain_vs_three_feature_mlp=w.mlp_3features-w.challenger_mlp,
        primary_test_gain_vs_three_feature_mlp=w.mlp_3features-w.selected_mlp))
contributions=pd.DataFrame(rows);contributions.to_csv(out/'feature_contributions.csv',index=False)
lines=['Executed and audited: feature discovery followed by fixed MLP confirmation.',
    'Screened 520 non-core original fields using earlier data only; selected at most one added feature per window. Existing three core features and 8/4 residual MLP settings retained.',
    'Screening proxy uses regularized binned training residuals; actual MLP must separately pass prior-quarter MAE and monthly consistency gate. No test-driven feature selection.', '']
for n,r in test.iterrows():lines.append(f'{n}: test MAE {r.mae:.6f} bps; Spearman IC {r.rank_ic:.6f}; Pearson IC {r.pearson_ic:.6f}.')
lines+=['','Per-window feature contribution:\n'+contributions.to_string(index=False),'',
    f'Primary uses the added feature in {sum(d["use_addition"] for d in decisions)}/4 windows. Beats original baseline MAE: {bool(test.loc["selected_mlp","mae"]<test.loc["baseline","mae"])}. Beats three-feature MLP: {bool(test.loc["selected_mlp","mae"]<test.loc["mlp_3features","mae"])}. Beats three-feature regression/RF: {bool(test.loc["selected_mlp","mae"]<test.loc["vol_ratio","mae"])}.',
    'Broad multiple-feature search on already-examined history; no untouched confirmation. Selected features vary by window; pooled score is a selection strategy, not a single fixed feature set. Proxy gains alone do not establish neural benefit. Results do not establish that untested or differently combined features cannot help.']
text='\n'.join(lines)+'\n';(out/'SUMMARY.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';marker='## MLP feature discovery result — 2026-11-04'
if marker not in context.read_text(encoding='utf-8'):
    with context.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n\n'+text+'\nSources: `experiments/mlp_feature_discovery_v1/SUMMARY.txt`, `RESULTS.txt`, `feature_contributions.csv`, `all_feature_screen.csv`, `shortlist.csv`, `selected_feature_definitions.csv`, `scores.csv`, `window_scores.csv`, `uncertainty.csv`, `SCREEN_COMPLETE.json`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `mlp_feature_discovery.py`; viewer: `BTC_48h_MLP_Feature_Discovery.ipynb`.\n')
nb=dict(cells=[dict(cell_type='markdown',metadata={},source=['# MLP feature discovery\n\n'+text]),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=["from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nroot=Path('experiments/mlp_feature_discovery_v1')\nfor name in ['scores','feature_contributions','window_scores','shortlist','uncertainty']:\n    print(name)\n    display(pd.read_csv(root / (name+'.csv')))\n"])],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
(HERE/'BTC_48h_MLP_Feature_Discovery.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True)
print(text)
