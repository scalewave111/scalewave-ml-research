"""Record the cache-path correction without changing screened or selected choices."""
import json
from pathlib import Path
import focused_four_feature_4h as run

manifest_path=run.OUT/'MANIFEST.json'
manifest=json.loads(manifest_path.read_text())
source_path=str(Path(run.__file__))
old=manifest['sources'][source_path]
snapshot=run.OUT/'source_snapshot.py'
assert run.s.q.sha(snapshot)==old
new=run.s.q.sha(source_path)
assert old!=new
amendment=run.OUT/'ORCHESTRATION_AMENDMENT.json'
if amendment.exists():
    assert json.loads(amendment.read_text())['new_source_sha']==new
else:
    run.s.dump(amendment,dict(reason='RF inner and outer fits shared one linear cache path despite different training cutoffs. Isolate the RF outer cache beneath outer/fit. Screen results and prior-validation selections are unchanged.',old_source_sha=old,new_source_sha=new,original_snapshot=str(snapshot),amended_snapshot=str(run.OUT/'source_snapshot_amended.py')))
    (run.OUT/'source_snapshot_amended.py').write_bytes(Path(run.__file__).read_bytes())
manifest['sources'][source_path]=new
run.s.dump(manifest_path,manifest)
print('Recorded RF cache-path amendment',flush=True)
