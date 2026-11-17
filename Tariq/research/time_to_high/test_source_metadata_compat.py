"""Offline metadata-only compatibility and checkpoint-identity regression tests."""
import ast
import copy
import hashlib
import json
import runpy
from pathlib import Path

root=Path(__file__).parent
module=runpy.run_path(str(root/'source_metadata_compat.py'))
config=json.loads((root/'screen_config.json').read_text())
expected=config['audit_sources']['X']
actual=dict(expected,modifiedTime='2026-11-16T07:46:35.633Z')
assert module['compare'](actual,expected)==['modifiedTime']
assert module['compare'](expected,expected)==[]
for key in ('id','name','size','md5Checksum'):
    bad=dict(actual); bad[key]='changed'
    try: module['compare'](bad,expected)
    except AssertionError: pass
    else: raise AssertionError('Accepted changed '+key)
for bad in [{k:v for k,v in actual.items() if k!='md5Checksum'},dict(actual,md5Checksum=''),dict(actual,new_field='unexpected')]:
    try: module['compare'](bad,expected)
    except AssertionError: pass
    else: raise AssertionError('Accepted incomplete/different metadata')

class Service:
    def files(self): return self
    def get(self,**kwargs): return self
    def execute(self): return dict(actual)
class Store:
    written={}
    def __init__(self,service,folder): pass
    def json(self,name,value): self.written[name]=copy.deepcopy(value)

for horizon in ('15m','1h'):
    source=(root/f'screen_{horizon}_runner.py').read_text()
    namespace={'REF':{'DriveStore':Store}}
    evidence=module['compile_runner'](source,namespace,horizon,'test-patch-hash')
    assert namespace['SOURCE_METADATA_CHECK'](Service(),config,'X',expected['id'])
    assert Store.written[f'SOURCE_METADATA_COMPAT_{horizon}.json']['sources']['X']['observed']==actual
    assert evidence['original_runner_sha256']==module['APPROVED'][horizon]
    patched=source.replace(module['OLD'],module['NEW'])
    assert evidence['executed_runner_sha256']==hashlib.sha256(patched.encode()).hexdigest()
    old_tree=ast.parse(source); new_tree=ast.parse(patched)
    old_fit=next(v for v in old_tree.body if isinstance(v,ast.FunctionDef) and v.name=='fit_job')
    new_fit=next(v for v in new_tree.body if isinstance(v,ast.FunctionDef) and v.name=='fit_job')
    assert ast.dump(old_fit)==ast.dump(new_fit)
    # Reverse exactly the two permitted assertions: no scheduling/model/identity edit.
    assert patched.count(module['NEW'])==2 and patched.replace(module['NEW'],module['OLD'])==source
    try: module['compile_runner'](source+'\n',{},horizon,'test')
    except AssertionError: pass
    else: raise AssertionError('Unapproved runner accepted')
print('PASS: timestamp-only acceptance; ID/name/size/MD5/field-change rejection; audit evidence; exact two-assertion patch; model and checkpoint-identity code untouched.')
