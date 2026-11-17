from pathlib import Path
import json,hashlib
p=Path('Code/nested_ic_rf.py')
out=Path('Experiments/nested_ic_rf_v3')
source=p.read_text(encoding='utf-8')
old_hashes=json.loads((out/'source_hashes.json').read_text())
ledger=json.loads((out/'fit_ledger.json').read_text())
for r in ledger:
 if r['status']=='STARTED':
  if (out/'models'/(r['id']+'.joblib')).exists():
   raise RuntimeError('Unexpected model for incomplete fit: inspect before resuming')
  r.update(status='INTERRUPTED',reason='Resource-only restart from 2 to 6 RF workers; no completed result discarded.')
(out/'fit_ledger.json').write_text(json.dumps(ledger,indent=2))
source=source.replace('self.rankings={}; self.receipts=[]',"self.rankings={}; self.receipts=json.loads((OUT/'fit_ledger.json').read_text()) if (OUT/'fit_ledger.json').exists() else []")
source=source.replace('random_state=SEED,n_jobs=2','random_state=SEED,n_jobs=6')
source=source.replace("except Exception as error:\n   receipt.update(status='FAILED',error=repr(error));","except BaseException as error:\n   receipt.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=repr(error));")
source=source.replace('def register(x,months,report):\n OUT.mkdir',"""def register(x,months,report,resume=False):
 if OUT.exists() and resume:
  contract=json.loads((OUT/'CONTRACT.json').read_text())
  assert contract['configs']==CONFIGS and contract['outer_months']==list(map(str,months))
  assert (OUT/'RESUME.json').exists(), 'Resume must be explicitly recorded'
  return
 OUT.mkdir""")
source=source.replace('def run():\n raw,','def run(resume=False):\n raw,')
source=source.replace('register(x,months,report)','register(x,months,report,resume=resume)')
source=source.replace("assert r['status']=='PASS' and","assert r['status'] in ['PASS','INTERRUPTED'] and")
source=source.replace("choices=['run','verify']","choices=['run','resume','verify']")
source=source.replace("run() if args.command=='run' else verify()","verify() if args.command=='verify' else run(resume=args.command=='resume')")
compile(source,str(p),'exec')
p.write_text(source,encoding='utf-8')
(out/'training_resume_source.py').write_text(source,encoding='utf-8')
record=dict(reason='Resource-only resume, 2 to 6 RF workers; trees, seeds, candidates, splits unchanged',
 previous_source_hashes=old_hashes,previous_fit_attempts=len(ledger),interrupted=sum(r['status']=='INTERRUPTED' for r in ledger))
(out/'RESUME.json').write_text(json.dumps(record,indent=2))
new_hashes=dict(old_hashes); new_hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
new_hashes[str((out/'training_source_snapshot.py').resolve())]=hashlib.sha256((out/'training_source_snapshot.py').read_bytes()).hexdigest()
new_hashes[str((out/'training_resume_source.py').resolve())]=hashlib.sha256((out/'training_resume_source.py').read_bytes()).hexdigest()
(out/'source_hashes.json').write_text(json.dumps(new_hashes,indent=2))
print('Resume prepared:',record['previous_fit_attempts'],'attempts,',record['interrupted'],'interrupted')
