"""Derive an isolated 1h runner without touching the active 15m implementation."""
import ast
import re
from pathlib import Path

root=Path(__file__).parent
source=(root/'screen_15m_runner.py').read_text(encoding='utf-8')
source=source.replace('15m','1h')
source=re.sub(r'\b15\b','60',source)
source=source.replace('SCREEN_Y_1h.npz','SCREEN_Y_60m.npz')
marker='def run_1h(config, expected_pilot, hashes, session_hours=4):'
assert marker in source
helper='''class IsolatedStore:
    """Write only to this horizon; borrow only complete shared caches read-only."""
    def __init__(self, private, shared, months):
        self.private,self.shared,self.months=private,shared,months
    def read(self,name):
        raw=self.private.read(name)
        if raw is not None: return raw
        if not (name.startswith('SCREEN_X_F4Z_') or name=='SCREEN_Y_60m.npz'):
            return None
        raw=self.shared.read(name)
        if raw is None: return None
        import io
        with np.load(io.BytesIO(raw),allow_pickle=False) as cache:
            meta=json.loads(str(cache['metadata'].item()))
        if meta['completed']!=self.months: return None
        # REF.read_cache verifies source identity and full array checksum before use.
        return raw
    def write(self,name,data,mime='application/json'):
        return self.private.write(name,data,mime)
    def json(self,name,value):
        return self.private.json(name,value)


'''
source=source.replace(marker,helper+marker)
anchor="    identity={'target':config['target'],'horizon_minutes':60,'pilot_signature':pilot['signature'],"
assert anchor in source
source=source.replace(anchor,"    shared_store=store\n    work_root=folder(service,config['folder'],'Timing_1h_Run')\n    store=IsolatedStore(REF['DriveStore'](service,work_root),shared_store,config['months'])\n"+anchor)
source=source.replace("folder(service,config['folder'],'Timing_1h_Fit_Scores')","folder(service,work_root,'Timing_1h_Fit_Scores')")
source=source.replace("folder(service,config['folder'],'Timing_Research_Reports')","folder(service,work_root,'Timing_Research_Reports')")
old="    assert ym['completed']==config['months'] and ts is not None, 'Existing complete 1h pilot target cache required'"
assert old in source
replacement='''    if ym['completed']!=config['months']:
        class TargetBudgetStore:
            def read(self,name): return store.read(name)
            def write(self,name,data,mime='application/json'):
                store.write(name,data,mime)
                if time.monotonic()>=deadline: raise BudgetStop('Target month saved; resume next session')
        try:
            if time.monotonic()>=deadline: raise BudgetStop('Session budget reached')
            print('Loading 1h target via remote ZIP ranges; research months only.',flush=True)
            yarchive=REF['RemoteArchive'](credentials,config['audit_sources']['Y'])
            manifest_name=next(n for n in yarchive.members if n.endswith('/HANDOFF_2_MANIFEST.json'))
            yroot=manifest_name[:-len('HANDOFF_2_MANIFEST.json')]
            manifest={yroot+entry['path']:entry['sha256'] for entry in json.loads(yarchive.read(manifest_name))['files']}
            loaded_times,y,yhash=REF['load_target'](yarchive,TargetBudgetStore(),yroot,manifest,60,config['audit_sources']['Y'],config)
            ts=loaded_times.asi8
            ym,_,_=REF['read_cache'](store,'SCREEN_Y_60m.npz',yidentity)
        except BudgetStop:
            state['status']='PAUSED_TARGET_LOADING_1h_ONLY'
            store.json('SCREEN_1h_RESULTS.json',state); save_reports()
            print('Target-loading progress saved. Rerun the same 1h notebook to resume.',flush=True)
            return
    assert ym['completed']==config['months'] and ts is not None
'''
source=source.replace(old,replacement)
ast.parse(source)
target=root/'screen_1h_runner.py'
if target.exists(): target.read_text(encoding='utf-8')
target.write_text(source,encoding='utf-8')
print('Prepared isolated 1h source; 15m source unchanged.')
