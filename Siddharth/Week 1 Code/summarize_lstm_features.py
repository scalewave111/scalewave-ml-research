from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/lstm_feature_architecture_v3'
complete=json.loads((OUT/'COMPLETE.json').read_text())
assert complete['status']=='EXECUTED_AND_AUDITED'
fs=json.loads((OUT/'FEATURE_SELECTIONS.json').read_text())
a=json.loads((OUT/'ARCHITECTURE_SELECTIONS.json').read_text())
f=pd.concat([pd.read_csv(p).assign(window=p.parent.name) for p in (OUT/'features').glob('*/importance.csv')])
importance=f.groupby('feature').agg(mean_mae_increase_bps=('mae_increase_bps','mean'),positive_windows=('mae_increase_bps',lambda s:int((s>0).sum()))).sort_values('mean_mae_increase_bps',ascending=False)
importance.to_csv(OUT/'feature_importance_summary.csv')
chosen=pd.DataFrame([dict(issue=s['issue'],feature_count=len(s['features']),lstm_layers=s['outer_spec']['layers'],hidden_units=s['outer_spec']['hidden'],dense_units=s['outer_spec']['dense'],epochs=s['selected']['epoch'],validation_mae=s['selected']['validation_mae']) for s in a])
chosen.to_csv(OUT/'selected_architectures.csv',index=False)
trials=pd.DataFrame([dict(issue=s['issue'],layers=r['spec']['layers'],hidden=r['spec']['hidden'],dense=r['spec']['dense'],validation_mae=r['validation_mae']) for s in a for r in s['trials']])
trials.groupby(['layers','hidden','dense']).validation_mae.mean().sort_values().to_csv(OUT/'architecture_mean_window_validation_mae.csv')
scores=pd.read_csv(OUT/'scores.csv')
text='FEATURE SELECTION THEN LSTM ARCHITECTURE TESTS\nExecuted on actual data; audit passed.\n\n'+scores.to_string(index=False)+'\n\nSelected architectures by window:\n'+chosen.to_string(index=False)+'\n\nModel-specific feature importance: mean validation MAE increase when shuffled; not causal importance.\n'+importance.to_string()+'\n\nFeature selection completed before architecture tuning. Each window uses only earlier labels.\nThe full 17-input set was retained when it beat the narrowed subsets in validation.\nOne seed and six bounded architectures were tested; no claim of a globally optimal network.\nAdamW, bias-excluded weight decay, dropout, gradient clipping, warmup/cosine LR and early stopping were used.\nThe historical test period was previously examined.\n'
(OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
def cell(kind,source):
 c=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
 if kind=='code':c.update(execution_count=None,outputs=[])
 return c
cells=[cell('markdown','# Feature selection and LSTM architecture results\nExecuted and audited locally. Set CODE to the Code folder if needed.'),cell('code',"from pathlib import Path\nimport pandas as pd\nCODE=Path.cwd()\nOUT=CODE/'experiments/lstm_feature_architecture_v3'\npd.read_csv(OUT/'scores.csv')"),cell('code',"pd.read_csv(OUT/'feature_importance_summary.csv')"),cell('code',"pd.read_csv(OUT/'selected_architectures.csv')"),cell('code',"pd.read_csv(OUT/'architecture_mean_window_validation_mae.csv')")]
(HERE/'BTC_48h_LSTM_Features_Architecture.ipynb').write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=4),indent=2),encoding='utf-8')
print(text)
