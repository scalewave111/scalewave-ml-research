"""Package isolated 1h Colab screen; never alter the running 15m notebook."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
names={'reference':'screen_runner.py','optimized':'screen_throughput.py','report':'timing_results_report.py','runner':'screen_1h_runner.py'}
names['io_guard']='drive_rate_guard.py'
names['source_compat']='source_metadata_compat.py'
sources={k:(root/name).read_text(encoding='utf-8') for k,name in names.items()}
old=json.loads((root/'Time_To_High_Screen_Pilot.ipynb').read_text())
sources['range_reader']=''.join(old['cells'][1]['source']).split('CONFIG =',1)[0]
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
for source in sources.values(): ast.parse(source)
pilot=json.loads((root/'SCREEN_RESULTS.json').read_text()); config=json.loads((root/'screen_config.json').read_text())
code='\n'.join([
    'import hashlib, json', 'sources = '+repr(sources), 'hashes = '+repr(hashes),
    'assert all(hashlib.sha256(v.encode()).hexdigest()==hashes[k] for k,v in sources.items())',
    'IO_GUARD = {}', 'exec(compile(sources["io_guard"], "drive_rate_guard.py", "exec"), IO_GUARD)',
    'IO_GUARD["install"]()',
    'REF = {}', 'exec(compile(sources["range_reader"], "range_reader.py", "exec"), REF)',
    'exec(compile(sources["reference"], "screen_runner.py", "exec"), REF)',
    'OPT = {"REF":REF}', 'exec(compile(sources["optimized"], "screen_throughput.py", "exec"), OPT)',
    'REPORT = {}', 'exec(compile(sources["report"], "timing_results_report.py", "exec"), REPORT)',
    'RUNNER = {"REF":REF,"OPT":OPT,"REPORT":REPORT}',
    'SOURCE_COMPAT = {}', 'exec(compile(sources["source_compat"], "source_metadata_compat.py", "exec"), SOURCE_COMPAT)',
    'SOURCE_COMPAT["compile_runner"](sources["runner"], RUNNER, "1h", hashes["source_compat"])',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'RUNNER["run_1h"](config,pilot_identity,hashes,session_hours=4)',
])
ast.parse(code)
intro='''# BTCUSDT time-to-high: independent 1-hour forecast screen
**New 1h-only research; no 15m results are required.** Upload this notebook to Colab and use a **separate CPU runtime** from the running 15m notebook. A second browser tab alone does not establish a separate runtime. Do not disconnect or restart your active 15m runtime. If a separate runtime is unavailable, run this notebook after the first one stops. Do not run two copies of this 1h notebook at once.

Quota-repair update: prefer one active horizon at a time for now. Separate output folders prevent file conflicts but do not isolate Google API quotas. Requests are now paced, and explicit rate-limit errors wait and retry. Model/runner hashes and checkpoint compatibility remain unchanged. If this notebook failed previously, restart its runtime and resume this updated version after a few minutes. Keep its existing checkpoints.

Source metadata compatibility: a hash-pinned execution patch permits only modifiedTime differences when file ID, name, size and MD5 still match. All other source changes stop execution. SOURCE_METADATA_COMPAT_1h.json records observed metadata and original/executed code hashes. The frozen contracts and saved cache/model identities remain unchanged.

Run all and authorize the same Google account. Two fit workers maximum. Four-hour soft session budget includes loading, then finishes active fits/loading month and saves progress. Rerun the same notebook to resume. A session may overrun during reads, active fits, uploads or report generation.

Same 386 features, five chronological validation periods, six inner settings, train-only preprocessing, strict 60-minute label-end purging, baseline evaluation rows and missing-data fallback policy. Reserved rows are excluded. No model/quality shortcuts. No completed 1h pilot fits exist to reuse.

**Parallel write isolation:** everything new goes under **Timing_1h_Run** inside the existing Drive output folder. Completed shared feature caches may be read, but this notebook never writes shared caches. Missing or incomplete shared caches are loaded into private 1h caches using authenticated remote ZIP ranges. This can duplicate some loading/cache storage while both horizons run, but prevents concurrent writes to the same files. No dataset is downloaded to your PC.

The first session also builds the 1h target cache from research partitions only. Expect loading messages before fitting. All original datasets and historical pilot receipts are read-only. Source/runtime/hash checks stop incompatible resumes.

Inside **Timing_1h_Run**:
- **SCREEN_1h_RESULTS.json**: session progress.
- **Timing_1h_Fit_Scores**: compact resumable fit checkpoints; no models or row predictions.
- **Timing_Research_Reports/60m_summary.pdf**: charts and coverage; 60m means 1h.
- **60m_feature_rankings.csv** and **60m_fold_details.csv** in the reports folder: all candidates and completed fold results.

Reports update after each family and at session end. Other-horizon counts show the frozen pilot only, not the live 15m run; the two writers do not merge reports during concurrent execution. These are provisional development results, not significance or trading evidence.

Keep the 15m notebook as it is. Send this notebook's final status and report-folder link when ready. This notebook does not depend on the 15m run finishing.
'''
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
          'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                   {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
dest=root/'Time_To_High_1h_Screen.ipynb'
if dest.exists(): json.loads(dest.read_text())
dest.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
print('Built isolated 1h notebook. No training launched; 15m files untouched.')
