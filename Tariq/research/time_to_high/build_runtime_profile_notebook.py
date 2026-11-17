"""Package a cache-only Colab profile; no local market-data execution."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
names = {'reference':'screen_runner.py', 'optimized':'screen_throughput.py', 'profile':'screen_runtime_profile.py'}
sources = {k:(root/name).read_text(encoding='utf-8') for k,name in names.items()}
hashes = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
for source in sources.values(): ast.parse(source)
pilot = json.loads((root/'SCREEN_RESULTS.json').read_text())
config = json.loads((root/'screen_config.json').read_text())
code = '\n'.join([
    'import hashlib, json',
    'sources = '+repr(sources),
    'hashes = '+repr(hashes),
    'assert all(hashlib.sha256(v.encode()).hexdigest() == hashes[k] for k,v in sources.items())',
    'REF = {}', 'exec(compile(sources["reference"], "screen_runner.py", "exec"), REF)',
    'OPT = {"REF":REF}', 'exec(compile(sources["optimized"], "screen_throughput.py", "exec"), OPT)',
    'PROFILE = {"REF":REF, "OPT":OPT}',
    'exec(compile(sources["profile"], "screen_runtime_profile.py", "exec"), PROFILE)',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'PROFILE["run_profile"](config, pilot_identity, hashes)',
])
ast.parse(code)
intro = '''# Find what makes the timing screen slow
Upload to Google Colab, select a CPU runtime and Run all. Authorize the same Google account. No GPU, dependency installation or Drive mount is needed.

Updated: uses explicit per-task timers, fixing the Python 3.13 concurrent cProfile "tool 2 is already in use" failure. Re-upload this updated notebook. The failed diagnostic is not a completed result; its at-most-21 fits restart to collect consistent timings.

This is the next diagnostic, not the completed equivalence check. It runs **21 optimized fits** on the same three cached pairs, latest validation fold: activity 5m / 15m forecast; pressure 1h / 48h forecast; CVD efficiency 4h / 300h forecast. Six inner settings and one outer fit per pair. These repeated fits measure previously unmeasured timing; they are not new research results.

The notebook separately measures training, preprocessing, prediction verification, checkpoint submission and upload. Two fit workers maximum, one ordered writer, no change to models/settings/rows. Timing categories overlap; do not add them together. Profiling overhead and the small sample mean this is not a reliable full-screen runtime forecast.

It requires the existing passing THROUGHPUT_CHECK.json and matching source identities, cache hashes and software versions. Only compact caches and receipts are read; no source ZIP contents or reserved rows are read. Scores must match the saved pilot, and predictions undergo independent traversal. No model binaries or row predictions are saved.

Output: **RUNTIME_PROFILE.json** in the existing Drive output folder. Existing research results and the prior verification receipt remain unchanged. The full screen stays disabled. A matching completed profile is reused. An interruption may restart at most these 21 diagnostic fits.

Send the final printed summary afterward. If identity/version checks fail, send the error instead of bypassing them.
'''
notebook = {'nbformat':4,'nbformat_minor':5,
            'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                     {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':code.splitlines(True)}]}
destination = root/'Time_To_High_Runtime_Profile.ipynb'
if destination.exists(): json.loads(destination.read_text())
destination.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
print('Built Time_To_High_Runtime_Profile.ipynb; no fits launched.')
