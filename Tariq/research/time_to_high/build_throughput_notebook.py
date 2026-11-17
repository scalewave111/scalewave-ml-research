"""Build the self-contained Colab verification notebook; never run real fits locally."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
reference = (root/'screen_runner.py').read_text(encoding='utf-8')
optimized = (root/'screen_throughput.py').read_text(encoding='utf-8')
config = json.loads((root/'screen_config.json').read_text())
pilot = json.loads((root/'SCREEN_RESULTS.json').read_text())
old = json.loads((root/'Time_To_High_Screen_Pilot.ipynb').read_text())
assert reference.rstrip() in ''.join(old['cells'][1]['source']).replace('\r\n','\n')
rh = hashlib.sha256(reference.encode()).hexdigest()
oh = hashlib.sha256(optimized.encode()).hexdigest()
code = '\n'.join([
    'import hashlib, json',
    'reference_source = '+repr(reference),
    'optimized_source = '+repr(optimized),
    'reference_hash = '+repr(rh),
    'optimized_hash = '+repr(oh),
    'assert hashlib.sha256(reference_source.encode()).hexdigest() == reference_hash',
    'assert hashlib.sha256(optimized_source.encode()).hexdigest() == optimized_hash',
    'REF = {}', 'exec(compile(reference_source, "frozen_reference", "exec"), REF)',
    'namespace = {"REF": REF}',
    'exec(compile(optimized_source, "optimized_runner", "exec"), namespace)',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'namespace["run_benchmark"](config, reference_hash, optimized_hash, pilot_identity)',
])
ast.parse(code)
intro = '''# Timing screen: speed and equivalence check
Upload this notebook to Google Colab, choose a CPU runtime, and Run all. Authorize the same Google account as the pilot. No GPU or Drive mount is needed.

This checks three fixed pairs using the latest validation fold: activity 5m / 15m horizon; pressure 1h / 48h horizon; CVD efficiency 4h / 300h horizon. It runs 21 reference and 21 optimized fits. Feature lookbacks differ from forecast horizons. No new feature research or full screen is launched.

Only existing compact research caches are read, with identity/content checks. Source ZIP metadata is checked, but no ZIP contents are downloaded. Missing caches or changed software versions stop the check. The reserved period is excluded.

The current reference matches the pilot notebook code, but its content hash differs from the historical declared runner hash. Both are recorded. Every tested score must also match its saved pilot receipt; this check does not authorize checkpoint migration.

Exact predictions, errors, coverage, imputation, tree size and selected settings must match. Only compact scores/timings are saved to **THROUGHPUT_CHECK.json** in the existing output folder. Pilot results remain untouched. Both phases use the new scheduling/checkpoint machinery, so their timing comparison isolates preparation reuse. Treat timings as a small benchmark, not a full-runtime forecast.

A matching completed verification is reused. An interrupted check may restart its bounded 42 fits because predictions needed for comparison are not stored. Send the final status/timings or the output-folder link afterward.
'''
notebook = {'nbformat':4, 'nbformat_minor':5,
            'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                     {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':code.splitlines(True)}]}
destination = root/'Time_To_High_Throughput_Check.ipynb'
if destination.exists():
    json.loads(destination.read_text())  # Inspect existing artifact before replacing in place.
destination.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
print('Built self-contained cache-only notebook; no training launched.')
