"""Self-contained all-horizon notebook; no time cutoff and no local real-data fit."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
names={'reference':'screen_runner.py','optimized':'screen_throughput.py','core':'staged_timing_core.py',
       'runner':'staged_timing_runner.py','report':'staged_timing_report.py','source_compat':'source_metadata_compat.py','io_guard':'drive_rate_guard.py'}
sources={k:(root/v).read_text(encoding='utf-8') for k,v in names.items()}
pilot_notebook=json.loads((root/'Time_To_High_Screen_Pilot.ipynb').read_text())
sources['range_reader']=''.join(pilot_notebook['cells'][1]['source']).split('CONFIG =',1)[0]
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
for source in sources.values(): ast.parse(source)
config=json.loads((root/'screen_config.json').read_text()); pilot=json.loads((root/'SCREEN_RESULTS.json').read_text())
code='\n'.join([
    'import hashlib, json', 'sources = '+repr(sources), 'hashes = '+repr(hashes),
    'assert all(hashlib.sha256(v.encode()).hexdigest()==hashes[k] for k,v in sources.items())',
    'IO = {}', 'exec(compile(sources["io_guard"],"drive_rate_guard.py","exec"),IO)', 'IO["install"]()',
    'REF = {}', 'exec(compile(sources["range_reader"],"range_reader.py","exec"),REF)',
    'exec(compile(sources["reference"],"screen_runner.py","exec"),REF)',
    'OPT = {"REF":REF}', 'exec(compile(sources["optimized"],"screen_throughput.py","exec"),OPT)',
    'SOURCE = {}', 'exec(compile(sources["source_compat"],"source_metadata_compat.py","exec"),SOURCE)',
    'CORE = {"REF":REF,"OPT":OPT}', 'exec(compile(sources["core"],"staged_timing_core.py","exec"),CORE)',
    'REPORT = {}', 'exec(compile(sources["report"],"staged_timing_report.py","exec"),REPORT)',
    'RUNNER = {"REF":REF,"OPT":OPT,"CORE":CORE,"SOURCE":SOURCE,"REPORT":REPORT}',
    'exec(compile(sources["runner"],"staged_timing_runner.py","exec"),RUNNER)',
    'config = json.loads('+repr(json.dumps(config))+')',
    'pilot_identity = json.loads('+repr(json.dumps(pilot['identity']))+')',
    'RUNNER["run"](config,pilot_identity,hashes)',
])
ast.parse(code)
intro='''# BTCUSDT time-to-high — staged screen across all ten horizons
## Before starting
Stop the previous 15m/1h notebooks. Run only this notebook: it is now the sole writer of shared research caches. Upload to Colab, choose CPU and Run all. Authenticate before leaving it unattended. No GPU, new packages or local data download required.

**No four-hour or 10 a.m. cutoff.** Runs 15m first, then 1h, 4h, 8h, 16h, 48h, 72h, 120h, 200h and 300h. It continues until the staged work is finished. Colab can still interrupt the runtime; rerun this same notebook to resume. It cannot restart a terminated Colab runtime automatically. Completion near tomorrow 10 a.m. is not guaranteed.

## The changed method
1. Fixed cheap model: group a feature into 16 training-only quantile bins, predict each group's training target median; groups below 64 rows use the training target median. Missing values form a separate group. Minimum 1,000 observed training values and variation are required, otherwise use the baseline fallback on all rows.
2. For each of the five outer periods, select three features using **inner** validation MAE only. Test all six original tree settings on those three. Choose the feature and setting by inner MAE; fit/evaluate one final model on that outer period. Outer scores never choose the shortlist or setting.
3. Report the resulting selection policy separately from cheap individual-feature rankings and the previous exhaustive-tree experiment. A good cheap-model rank does not establish a good tree rank. This is not an exhaustive feature/set/transformation search.

Maximum 950 expensive tree fits across ten horizons before reuse, plus cheap bin fits. That fit count is not a runtime estimate: missing cache loading may still take substantial time. All five chronological folds, exact baseline rows, label-end purges and reserve exclusions remain in place.

## Preservation and recovery
Your original 15m results remain untouched. Completed shared caches/months are reused. Original pilot fits and verified 15m fits are reused when that feature/setting/fold is needed by the shortlist. Cheap screening of those same features is new-model work, not a rerun of the old tree experiment.

Cheap scores checkpoint every ten features and after each family; interrupted unsaved cheap screens may repeat. Tree tuning checkpoints after every completed fit; unsaved fits may repeat. Completed staged horizons are skipped on resume. Only compact scores and receipts are stored—no model binaries or row predictions.

Drive request pacing and bounded rate-limit retries are enabled. Source modifiedTime-only differences are accepted only if ID, name, size and MD5 still match. Other identity/version/checksum mismatches stop the run. Unhandled errors are not converted into successful results or model fallbacks.

## Outputs
Everything new is in **Timing_Staged_All_Horizons** inside the existing Drive output folder:
- `staged_summary.pdf`: all-horizon progress and comparison charts.
- `horizon_matrix.csv`: baseline versus the selected staged policy.
- `cheap_feature_rankings.csv`: all 386 features per horizon, including pending rows.
- `selected_policy_folds.csv`: selected features/settings, errors and reuse evidence.
- `horizon_*m.json`: compact resumable evidence; `CONTRACT.json` identifies the new experiment.

Reports update after each family and horizon. Predictions/errors are provisional development evidence. Original upstream lineage remains unresolved. Shortlisting can miss useful features; no optimality, significance or profitability claim.

Wait for the preflight to pass and observe checkpoint progress before leaving it unattended. Send the final output/report-folder link, or the exact error if it stops. Do not delete checkpoints or start another horizon notebook alongside it.
'''
n={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
   'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
            {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
dest=root/'Time_To_High_All_Horizons_Staged.ipynb'
if dest.exists(): json.loads(dest.read_text())
dest.write_text(json.dumps(n,indent=1)+'\n',encoding='utf-8')
print('Built unified staged notebook; no real-data training launched locally.')
