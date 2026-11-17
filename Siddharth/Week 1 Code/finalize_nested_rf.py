"""Finish serialization/reporting and independently audit saved predictions, with zero new fits."""
from pathlib import Path
import json,hashlib,sys,importlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/nested_ic_rf_v3'
ledger=json.loads((OUT/'fit_ledger.json').read_text())
assert all(r['status'] in ['PASS','INTERRUPTED'] for r in ledger), 'Training still active or a fit failed.'
assert (OUT/'overall_training_testing.csv').exists()
before_predictions=hashlib.sha256((OUT/'predictions.csv').read_bytes()).hexdigest()
attempts=len(ledger)
source=ROOT/'Code/nested_ic_rf.py'
old_source_digest=hashlib.sha256(source.read_bytes()).hexdigest()
original=source.read_text(encoding='utf-8')
old="scores=scoreboard.reset_index().to_dict('records')"
new="scores=[dict(model=name,**metrics(values[name],truth)) for name in scoreboard.index]"
if old in original:
 # Constant M0 has undefined Pearson IC. Keep JSON null instead of pandas NaN.
 fixed=original.replace(old,new)
 assert fixed!=original
 compile(fixed,str(source),'exec')
 source.write_text(fixed,encoding='utf-8')
 (OUT/'evaluation_serialization_fix.json').write_text(json.dumps(dict(
  reason='Serialize undefined Pearson IC for constant baseline as JSON null, not pandas NaN.',
  affects='Result serialization only; no model, score, prediction, or selection changes.',
  previous_source_sha256=old_source_digest,
  prediction_sha256=before_predictions,additional_fits=0),indent=2))
 hashes=json.loads((OUT/'source_hashes.json').read_text())
 hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
 (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
sys.path.insert(0,str(ROOT/'Code'))
import nested_ic_rf as engine
if not (OUT/'RESULT.json').exists():
 raw,y,dev,report=engine.load()
 ref=engine.prior.frozen_predictions(raw).magnitude_log_bps
 engine.evaluate(engine.read_csv(OUT/'predictions.csv'),y,ref)
engine.verify()
import audit_nested_rf
audit_nested_rf.audit()
assert hashlib.sha256((OUT/'predictions.csv').read_bytes()).hexdigest()==before_predictions
assert len(json.loads((OUT/'fit_ledger.json').read_text()))==attempts
import build_nested_rf_deliverables as deliver
deliver.build_notebook()
deliver.build_report()
result=json.loads((OUT/'RESULT.json').read_text())
context=ROOT/'PROJECT_CONTEXT.md'
with context.open('a',encoding='utf-8') as stream:
 stream.write('\n## Completed nested experiment, 2026-11-03\n\n')
 stream.write(f"Executed on {result['rows']} unique outer-test rows. {attempts} attempts, {sum(r['status']=='PASS' for r in ledger)} completed, {sum(r['status']=='INTERRUPTED' for r in ledger)} interrupted for resource-only restart. No tuning based on outer scores. Exact scores: Experiments/nested_ic_rf_v3/RESULT.json and overall_training_testing.csv. Replay and independent checks passed.\n")
 stream.write('Deliverables: Code/BTC_48h_Nested_IC_RandomForest.ipynb and Experiments/nested_ic_rf_v3/RESULTS.html. Main source Code/nested_ic_rf.py, independent audit Code/audit_nested_rf.py. Training source snapshots, resource-only resume and result-serialization repair are retained. Raw/old results unchanged.\n')
 stream.write('Same-row comparisons: '+json.dumps(result['comparisons'])+'\n')
subprocess.run([sys.executable,str(ROOT/'refresh_file_index.py')],check=True)
print('FINALIZATION COMPLETE; no additional fits.')
