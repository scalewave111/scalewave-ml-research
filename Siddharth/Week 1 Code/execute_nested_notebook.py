from pathlib import Path
import os,json,hashlib
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
out=root/'Experiments/nested_ic_rf_v3'
for name in ['jupyter_runtime','ipython']:(out/name).mkdir(exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(out/'jupyter_runtime')
os.environ['IPYTHONDIR']=str(out/'ipython')
path=root/'Code/BTC_48h_Nested_IC_RandomForest.ipynb'
ledger_hash=hashlib.sha256((out/'fit_ledger.json').read_bytes()).hexdigest()
n=nbformat.read(path,as_version=4)
NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
assert hashlib.sha256((out/'fit_ledger.json').read_bytes()).hexdigest()==ledger_hash
assert not any(o.output_type=='error' for c in n.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(n,path)
(out/'NOTEBOOK_VALIDATION.json').write_text(json.dumps({'status':'PASS','executed_code_cells':sum(c.cell_type=='code' for c in n.cells),'additional_fits':0},indent=2))
print('NOTEBOOK EXECUTION PASS; no additional fits')
