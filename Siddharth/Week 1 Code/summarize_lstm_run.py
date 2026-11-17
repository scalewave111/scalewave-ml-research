"""Create a factual run summary and notebook after the LSTM audit finishes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
out=HERE/'experiments/lstm_walk_forward_v1'
complete=json.loads((out/'COMPLETE.json').read_text())
assert complete['status']=='EXECUTED_AND_AUDITED'
scores=pd.read_csv(out/'scores.csv')
p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True)
rows=[]
for model in ['lstm','lstm_hybrid']:
 delta=abs(p[model]-p.truth)-abs(p.baseline-p.truth)
 for days in [7,14]:
  blocks=((p.index-p.index.min()).total_seconds()//(days*86400)).astype(int)
  a=pd.DataFrame(dict(delta=delta,block=blocks)).groupby('block').delta.agg(['sum','count'])
  idx=np.random.default_rng(48+days).integers(0,len(a),size=(2000,len(a)))
  draws=a['sum'].to_numpy()[idx].sum(1)/a['count'].to_numpy()[idx].sum(1)
  rows.append(dict(model=model,block_days=days,delta_mae=float(delta.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
(out/'UNCERTAINTY.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
text='LSTM WALK-FORWARD RESULTS\nExecuted on actual data; audit passed.\n\nSame quarterly expanding history and 48h label purge as report baseline.\nAll 18,481 test rows retained. Six fixed LSTM configurations; validation-only selection.\nStandalone and median-regression hybrid scored separately. No test-driven retries.\n\n'+scores.to_string(index=False)+'\n\nPaired calendar-block bootstrap: negative delta MAE favors LSTM.\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nPreviously exposed historical evaluation, not a fresh holdout.\nConfidence intervals are exploratory and do not account for prior research selection.\n\nSee CONTRACT.json, SELECTIONS.json, MANIFEST.json, AUDIT.json and predictions.csv.\n'
(out/'RESULTS.txt').write_text(text,encoding='utf-8')
def cell(kind,source):
 c=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
 if kind=='code':c.update(execution_count=None,outputs=[])
 return c
cells=[cell('markdown','# LSTM walk-forward comparison\nSame four expanding quarterly windows as the report baseline. Results were executed and audited locally. Set CODE below to your Code folder. Existing completed runs are never overwritten.'),cell('code',"from pathlib import Path\nimport pandas as pd, json\nCODE = Path.cwd()  # Change if this notebook is opened elsewhere\nOUT = CODE / 'experiments/lstm_walk_forward_v1'\nassert (OUT/'COMPLETE.json').exists(), 'Run python lstm_walk_forward.py run first'\njson.loads((OUT/'COMPLETE.json').read_text())"),cell('code',"pd.read_csv(OUT/'scores.csv')"),cell('code',"pd.read_csv(OUT/'window_scores.csv')"),cell('code',"pd.DataFrame(json.loads((OUT/'UNCERTAINTY.json').read_text()))"),cell('markdown','To reproduce saved predictions and leakage checks, run `python lstm_walk_forward.py audit`. Training command: `python lstm_walk_forward.py run`; it refuses to overwrite a completed run. These are retrospective results, not an untouched holdout.')]
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=4)
(HERE/'BTC_48h_LSTM_WalkForward.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
print(text)
