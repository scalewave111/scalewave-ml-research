"""Build self-contained Colab comparison. Does not execute experiments."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
names={'reference':'screen_runner.py','optimized':'screen_throughput.py',
       'profile':'screen_runtime_profile.py','worker':'screen_worker_check.py'}
sources={k:(root/v).read_text(encoding='utf-8') for k,v in names.items()}
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
for source in sources.values(): ast.parse(source)
pilot=json.loads((root/'SCREEN_RESULTS.json').read_text())
config=json.loads((root/'screen_config.json').read_text())
code='\n'.join([
    'import hashlib, json', 'sources = '+repr(sources), 'hashes = '+repr(hashes),
    'assert all(hashlib.sha256(v.encode()).hexdigest()==hashes[k] for k,v in sources.items())',
    'REF = {}', 'exec(compile(sources["reference"], "screen_runner.py", "exec"), REF)',
    'OPT = {"REF":REF}', 'exec(compile(sources["optimized"], "screen_throughput.py", "exec"), OPT)',
    'PROFILE = {"REF":REF,"OPT":OPT}',
    'exec(compile(sources["profile"], "screen_runtime_profile.py", "exec"), PROFILE)',
    'WORKER = {"REF":REF,"OPT":OPT,"PROFILE":PROFILE}',
    'exec(compile(sources["worker"], "screen_worker_check.py", "exec"), WORKER)',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'WORKER["run_check"](config,pilot_identity,sources,hashes)',
])
ast.parse(code)
intro='''# Compare CPU workers without changing the experiment
Upload to Colab, select a CPU runtime, and Run all using the same Google account. No GPU, new dependencies or Drive mount required.

After an interrupted older worker check, restart the Colab session before running this updated notebook. Interrupting a threaded fit may leave work active in the old session. This version prints each fit completion, a 20-second waiting message, and applies a 240-second per-task timeout. A timeout marks the comparison incomplete; it never changes scores or skips difficult rows. Thread cancellation cannot forcibly terminate an active native fit, so restart the session after a timeout too.

This compares **two threads versus two separate processes**, using the same three cached feature/horizon pairs and latest validation fold. Each phase has 21 fits; 42 diagnostic fits total. Same six settings, rows, purges, preprocessing and random seed. This is an execution test, not new feature research.

Both phases must reproduce saved pilot scores and setting selections. Their prediction digests must match for every inner and outer fit; independent tree traversal remains enabled. No change is adopted automatically. The full screen stays disabled.

Only existing research caches are used. The prior passing RUNTIME_PROFILE.json is required. Source metadata, runtime versions and cache hashes are checked. Missing/changed inputs stop the check; no source ZIP download fallback. No model binaries or row predictions are saved. Process workers receive only arrays and model code, not Drive access.

The timings include worker startup and data transfer. Uploads happen outside both timed phases. One fixed-order comparison is preliminary; it does not guarantee a speedup across all features. If only one CPU is usable, no fits run. If interrupted, at most 42 diagnostic fits restart; a matching completed result is reused.

The receipt is **WORKER_CHECK.json** in the existing output folder. Send the final printed summary. Earlier research and verification receipts stay unchanged.
'''
notebook={'nbformat':4,'nbformat_minor':5,
          'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
          'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                   {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
destination=root/'Time_To_High_Worker_Check.ipynb'
if destination.exists(): json.loads(destination.read_text())
destination.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
print('Built Time_To_High_Worker_Check.ipynb; no real-data fits launched.')
