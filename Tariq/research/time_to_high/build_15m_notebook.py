"""Package one-horizon Colab run and render existing results, without training."""
import ast
import hashlib
import json
import runpy
from pathlib import Path

root=Path(__file__).parent
names={'reference':'screen_runner.py','optimized':'screen_throughput.py','report':'timing_results_report.py','runner':'screen_15m_runner.py'}
names['io_guard']='drive_rate_guard.py'
names['source_compat']='source_metadata_compat.py'
sources={k:(root/name).read_text(encoding='utf-8') for k,name in names.items()}
old=json.loads((root/'Time_To_High_Screen_Pilot.ipynb').read_text())
sources['range_reader']=''.join(old['cells'][1]['source']).split('CONFIG =',1)[0]
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
for source in sources.values(): ast.parse(source)
pilot=json.loads((root/'SCREEN_RESULTS.json').read_text())
config=json.loads((root/'screen_config.json').read_text())
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
    'SOURCE_COMPAT["compile_runner"](sources["runner"], RUNNER, "15m", hashes["source_compat"])',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'RUNNER["run_15m"](config,pilot_identity,hashes,session_hours=4)',
])
ast.parse(code)
intro='''# BTCUSDT — 15-minute time-to-high screen
**This notebook performs new 15m-only research.** It tests all 386 frozen individual-feature candidates, resuming work across sessions. It never launches the other nine horizons. Upload to Colab, choose CPU, Run all, and authorize the same Google account. Restart the runtime first if an older interrupted worker test may still be active.

Drive quota repair: all Drive requests are paced, and explicit rate-limit rejections retry with increasing waits. The model/runner source and checkpoint identities are unchanged, so saved fits remain reusable. Resume with this updated notebook after a few minutes' cooldown. For now run only one horizon at a time because separate runtimes can share a project quota. Do not delete checkpoints. Only unsaved fits may repeat. If retries are exhausted, the error is surfaced rather than silently dropping results.

Source metadata compatibility: the original runner source is hash-pinned. A recorded execution patch changes only its startup/final archive checks to permit a modifiedTime-only difference when file ID, name, size and MD5 still match. Any other difference stops execution. SOURCE_METADATA_COMPAT_15m.json records observed metadata and both original/executed code hashes. Frozen source contracts and cache/model identities are not rewritten.

The three completed 15m pilot jobs are reused under their original verified identities. New work uses separate receipts; historical pilot files are not overwritten. All five folds, six inner settings, exact baseline rows, missing-data rules, independent tree traversal and strict label-end purges remain unchanged. Reserved rows are excluded.

Two thread workers maximum. Four-hour soft budget includes loading: finish active fits or the current loading month, flush checkpoints and stop. The session can overrun during a fit, read, upload or report generation. Rerun this same notebook to resume. Completed fits are reused; only unsaved interrupted fits can restart. Do not run concurrent notebook copies against the same output folder.

Data remain remote. Existing family caches are reused, and missing research families are loaded by authenticated ZIP range reads with monthly checkpoints. No ZIP download to your PC. No model binaries or row predictions saved. Changed sources, versions, configuration or implementation identity stop the run for review.

**Results:** the Drive output folder contains `SCREEN_15m_RESULTS.json`, `Timing_15m_Fit_Scores`, and `Timing_Research_Reports`. The reports folder holds:
- `15m_summary.pdf`: progress across ten horizons, ranked comparison chart, consistency heatmap and limitations.
- `15m_feature_rankings.csv`: all 386 features, with pending/partial/complete status, baselines, improvements, coverage and settings.
- `15m_fold_details.csv`: each completed validation period, selected settings and coverage.

Reports update after each family and at session end. Per-fit checkpoints are saved throughout. Only complete five-fold features are ranked. A highest rank can still lose to the baseline. Improvement is provisional development evidence, not significance or profitability.

Send the final status and reports-folder link after a session. Initial flow-only pilot runtimes do not establish how long all 386 features will take at 15m.
'''
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
          'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                   {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
dest=root/'Time_To_High_15m_Screen.ipynb'
if dest.exists(): json.loads(dest.read_text())
dest.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
report=runpy.run_path(str(root/'timing_results_report.py'))
jobs=json.loads((root/'pilot_fit_scores.json').read_text())
baseline=json.loads((root/'BASELINE_RESULTS.json').read_text())
out=root/'Results'/'time_to_high_min'; out.mkdir(parents=True,exist_ok=True)
for name,data in report['artifacts'](config,baseline,jobs).items():
    target=out/name
    if target.exists(): target.read_bytes()
    target.write_bytes(data)
print('Built 15m-only notebook and report from saved pilot scores. No real-data training launched.')
