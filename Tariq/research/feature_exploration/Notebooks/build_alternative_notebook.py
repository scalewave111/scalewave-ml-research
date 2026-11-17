"""Build a standalone, documented Colab notebook using the standard notebook JSON format."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(True)})
def code(title, text):
    md('## '+title)
    source = '# '+title+'\n'+textwrap.dedent(text).strip()+'\n'
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':source.splitlines(True)})

md('''
# Alternative F4Z research: learn a range, rank it later, validate the top ten
### BTCUSDT · forward 48-hour signed return strictly above +500 log bps

**Run this notebook from top to bottom in Google Colab.** It starts with the two original
Parquet files; it does not require outputs from your earlier notebooks. Mount your Drive,
check the paths in Section 2, and choose **Runtime → Run all**. No GPU is needed.
The full run reads every feature and may take substantial time on Google Drive.
Feature checkpoints allow an interrupted screening run to continue.

**What you receive:** a CSV and displayed table of up to ten highest-ranked eligible
individual F4Z columns, their selected value ranges, event rates, quarter-adjusted effects,
block-resampling uncertainty, and separate later-period diagnostics. Charts and audit files
are saved to a dedicated results directory. No results are fabricated to fill ten places.

### How this differs from the previous lift screen
1. **Learn**: divide each feature into up to five value bands using only early data. Pick
   its strongest eligible band in that early period. Middle bands can win too.
2. **Rank**: freeze that band and measure its association in a later, separate period.
   Compare condition-true versus condition-false readings within the same quarter.
3. **Account for clustered observations**: resample whole seven-day blocks within quarters,
   rather than treating adjacent minutes as independent. Rank by the 5th percentile of the
   resampled adjusted risk difference, expressed in percentage points.
4. **Freeze top ten**, then report how those exact columns and bands perform in the final period.
   Do not re-rank using the final period.

This is a **defined descriptive association ranking**, not the universal ten best features,
a causal model, an ML feature-importance ranking, or proof of profitable trading.
Individual columns can be highly related; a separate one-per-parent table is also supplied.
The final period was described in previous project work, so it is a reserved diagnostic,
not a newly collected prospective test. Avoid repeatedly tuning this notebook after viewing it.
''')
code('1. Import libraries and mount Google Drive', '''
import sys, json, hashlib, re, gc, time, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# These libraries are included in standard Colab runtimes; no global installation is needed.
from google.colab import drive
drive.mount('/content/drive')
pd.set_option('display.max_colwidth', 110)
pd.set_option('display.max_columns', 30)
print('Python:', sys.version.split()[0], '| pandas:', pd.__version__, '| Arrow:', pa.__version__)
''')
code('2. Set file paths and freeze the research rules', '''
# Change PROJECT_DIR if your Google Drive folder has a different location.
PROJECT_DIR = Path('/content/drive/MyDrive/ScaleWave')
DATA_DIR = PROJECT_DIR / 'Data'
FILES = {
    'NON_EXTREME': DATA_DIR / 'BTCUSDT_signed_return_between_minus500_and_plus500_20261027_101854_1aefb791.parquet',
    'EXTREME': DATA_DIR / 'BTCUSDT_signed_return_above_500_or_below_minus500_20261027_093028_e1453352.parquet',
}

# All thresholds are chosen before screening. Change them only for a documented new experiment.
CFG = {
    'method_version': 'range_transfer_weekly_bootstrap_v1',
    'symbol': 'BTCUSDT', 'event_threshold': 500.0, 'horizon_minutes': 2880,
    'rank_start': '2023-07-01T00:00:00Z',
    'rank_end': '2024-07-01T00:00:00Z',
    'validation_start': '2024-07-03T00:00:00Z',
    'quantile_bands': 5,
    'min_coverage': 0.10,
    'min_condition_rows': 1000, 'min_control_rows': 1000,
    'min_rows_per_group_per_quarter': 100,
    'min_train_quarters': 2, 'min_rank_quarters': 3,
    'min_condition_blocks': 10,
    'block_days': 7, 'bootstrap_replicates': 500,
    'bootstrap_lower_quantile': 0.05,
    'min_valid_bootstrap_fraction': 0.90,
    'seed': 20261030, 'top_k': 10,
}

# Purge the final 48 hours of learning so its forward labels do not reach the ranking period.
RANK_START = pd.Timestamp(CFG['rank_start'])
LEARN_END = RANK_START - pd.Timedelta(minutes=CFG['horizon_minutes'])
RANK_END = pd.Timestamp(CFG['rank_end'])
VALIDATION_START = pd.Timestamp(CFG['validation_start'])
assert LEARN_END < RANK_START < RANK_END < VALIDATION_START
assert VALIDATION_START - RANK_END >= pd.Timedelta(minutes=CFG['horizon_minutes'])
assert 0 < CFG['min_coverage'] <= 1 and CFG['block_days'] >= 2
assert CFG['bootstrap_replicates'] >= 100
print('Learning ends before:', LEARN_END)
print('Ranking:', RANK_START, 'to before', RANK_END)
print('Reserved validation starts:', VALIDATION_START)
display(pd.Series(CFG, name='Frozen configuration').to_frame())
''')
md('''
### What the rules mean
**Coverage** is usable readings divided by all observations in that period. A band needs
at least 1,000 condition rows and 1,000 other valid rows, across at least ten condition-active
weekly blocks. Each comparable quarter needs at least 100 rows in each group.
Learning needs at least two comparable quarters; ranking needs at least three.
These are predeclared practical support gates, not significance tests. They differ from the
older notebook and can exclude features that become usable only late in history.

The sample is **not balanced or downsampled**: negative extremes remain non-events, together
with ordinary returns. +500 is a strict threshold; exactly +500 is a non-event.
''')
code('3. Verify input schemas and create a resumable results folder', '''
# Fail clearly when a supplied file is missing; never silently choose a different dataset.
for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f'{name} file is missing: {path}. Edit Section 2.')

PFS = {name: pq.ParquetFile(path) for name, path in FILES.items()}
schemas = {name: pf.schema_arrow for name, pf in PFS.items()}
column_sets = [set(s.names) for s in schemas.values()]
if column_sets[0] != column_sets[1]:
    raise ValueError('The two input files have different column sets. Inspect before proceeding.')
common = column_sets[0]
FEATURES = sorted(c for c in common if c.lower().startswith('f4z__') and c.endswith('__value'))
if not FEATURES:
    raise ValueError('No F4Z __value columns found.')
VALID_COL = {f: f.removesuffix('__value') + '__valid' for f in FEATURES}
AVAIL_COL = {f: '__'.join(f.split('__')[:2]) + '__available_after' for f in FEATURES}
required = {'decision_time', 'canonical_symbol', 'signed_return', 'horizon_minutes', 'label_status'}
required |= set(VALID_COL.values()) | set(AVAIL_COL.values())
missing = required - common
if missing:
    raise ValueError(f'Required columns are missing: {sorted(missing)[:20]}')
for schema in schemas.values():
    assert all(pa.types.is_boolean(schema.field(c).type) for c in VALID_COL.values())

# Record schema and file metadata. These identify the local inputs but are not full file hashes.
input_manifest = [{
    'dataset': name, 'path': str(path), 'size_bytes': path.stat().st_size,
    'mtime_ns': path.stat().st_mtime_ns, 'rows': PFS[name].metadata.num_rows,
    'schema_sha256': hashlib.sha256(str(schemas[name]).encode()).hexdigest(),
} for name, path in FILES.items()]
signature = hashlib.sha256(json.dumps({'config': CFG, 'files': input_manifest}, sort_keys=True).encode()).hexdigest()[:16]
RUN_DIR = PROJECT_DIR / 'Results' / '03_Alternative_F4Z_Association' / signature
CHECKPOINT_DIR = RUN_DIR / 'feature_checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / 'config.json').write_text(json.dumps(CFG, indent=2))
(RUN_DIR / 'input_manifest.json').write_text(json.dumps(input_manifest, indent=2))
print(f'Found {len(FEATURES)} individual F4Z columns.')
print('Output/checkpoint directory:', RUN_DIR)
display(pd.DataFrame(input_manifest)[['dataset','rows','size_bytes']])
''')
code('4. Audit labels, partitions, timestamps and feature availability', '''
# Load only labels and seven family availability fields, not the wide feature matrix.
META_COLUMNS = ['decision_time','canonical_symbol','signed_return','horizon_minutes','label_status']
AVAIL_COLUMNS = sorted(set(AVAIL_COL.values()))
parts = []
for name, pf in PFS.items():
    frame = pf.read(columns=META_COLUMNS + AVAIL_COLUMNS).to_pandas()
    frame['decision_time'] = pd.to_datetime(frame['decision_time'], utc=True, errors='raise')
    if frame['decision_time'].isna().any():
        raise ValueError('Missing decision timestamps.')
    y_raw = pd.to_numeric(frame['signed_return'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y_raw).all():
        raise ValueError('Non-finite target values found.')
    assert frame['label_status'].eq('VALID').all(), 'Invalid label rows found.'
    assert frame['canonical_symbol'].eq(CFG['symbol']).all(), 'Unexpected symbol.'
    assert pd.to_numeric(frame['horizon_minutes']).eq(CFG['horizon_minutes']).all(), 'Unexpected horizon.'
    ordinary = (y_raw >= -500) & (y_raw <= 500)
    assert (ordinary if name == 'NON_EXTREME' else ~ordinary).all(), 'File partition violation.'
    frame['signed_return'] = y_raw
    frame['source_dataset'] = name
    for col in AVAIL_COLUMNS:
        frame[col] = pd.to_datetime(frame[col], utc=True, errors='raise')
    parts.append(frame)

# Preserve physical concatenation order so each later feature aligns with the correct target.
physical = pd.concat(parts, ignore_index=True)
ORDER = np.argsort(physical['decision_time'].astype('int64').to_numpy(), kind='stable')
META = physical.iloc[ORDER].reset_index(drop=True)
del physical, parts
assert not META['decision_time'].duplicated().any(), 'Duplicate or overlapping decision timestamps.'
TIMES = pd.DatetimeIndex(META['decision_time'])
TARGET = (META['signed_return'].to_numpy() > CFG['event_threshold']).astype(np.uint8)
AVAILABLE = {}
availability_audit = []
for col in AVAIL_COLUMNS:
    # Missing or late family availability excludes that feature's reading, even when __valid is true.
    AVAILABLE[col] = (META[col].notna() & META[col].le(META['decision_time'])).to_numpy()
    availability_audit.append({'column': col, 'unavailable_rows': int((~AVAILABLE[col]).sum())})
audit = pd.DataFrame(availability_audit)
audit.to_csv(RUN_DIR / 'availability_audit.csv', index=False)
display(audit)
gaps = pd.Series(TIMES).diff()
print('Rows:', len(META), '| positive events:', int(TARGET.sum()))
print('Gaps greater than one minute:', int((gaps > pd.Timedelta(minutes=1)).sum()))
print('Observed history:', TIMES.min(), 'to', TIMES.max())
# Keep target arrays and availability masks; release string-heavy metadata after the next step.
''')
code('5. Construct the three chronological periods and weekly blocks', '''
PERIOD_MASKS = {
    'learn': np.asarray(TIMES < LEARN_END),
    'rank': np.asarray((TIMES >= RANK_START) & (TIMES < RANK_END)),
    'validation': np.asarray(TIMES >= VALIDATION_START),
}
assert not np.any(PERIOD_MASKS['learn'] & PERIOD_MASKS['rank'])
assert not np.any(PERIOD_MASKS['rank'] & PERIOD_MASKS['validation'])

def make_period(mask):
    # Anchor seven-day blocks at each quarter's start; short boundary blocks are retained.
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise ValueError('A configured period is empty. Review dates in Section 2.')
    ts = TIMES[idx]
    q = ts.tz_localize(None).to_period('Q')
    q_labels = sorted(q.astype(str).unique())
    q_lookup = {v: i for i, v in enumerate(q_labels)}
    row_q = np.array([q_lookup[v] for v in q.astype(str)], dtype=np.int16)
    start = q.start_time.tz_localize('UTC')
    week = np.asarray((ts - start).days // CFG['block_days'])
    keys = list(zip(row_q.tolist(), week.tolist()))
    unique_keys = sorted(set(keys))
    key_lookup = {v: i for i, v in enumerate(unique_keys)}
    row_b = np.fromiter((key_lookup[v] for v in keys), dtype=np.int32)
    block_q = np.array([v[0] for v in unique_keys], dtype=np.int16)
    return {'idx': idx, 'y': TARGET[idx], 'row_q': row_q, 'row_b': row_b,
            'block_q': block_q, 'quarters': q_labels, 'n_blocks': len(unique_keys)}

PERIODS = {name: make_period(mask) for name, mask in PERIOD_MASKS.items()}
summary = []
for name, p in PERIODS.items():
    # Validation event rates are intentionally not displayed before the ranking is frozen.
    summary.append({'period': name, 'rows': len(p['idx']),
                    'start': str(TIMES[p['idx'][0]]), 'end': str(TIMES[p['idx'][-1]]),
                    'quarters': len(p['quarters']), 'weekly_blocks': p['n_blocks']})
display(pd.DataFrame(summary))
pd.DataFrame(summary).to_csv(RUN_DIR / 'periods.csv', index=False)
assert len(PERIODS['learn']['quarters']) >= CFG['min_train_quarters']
assert len(PERIODS['rank']['quarters']) >= CFG['min_rank_quarters']
print('Boundary-excluded observations:', int((~np.logical_or.reduce(list(PERIOD_MASKS.values()))).sum()))
del META
gc.collect()
''')
md('''
### The association statistic used below
Within each quarter, count condition-true rows/events and other valid rows/events.
Use only quarters with at least 100 rows in each group. For each retained quarter:

`expected condition events = condition row count × other-valid-row event rate`

Add observed and expected condition events across quarters and divide both by the total
condition-row count. **Adjusted risk difference** is observed rate minus expected rate.
For example, 13% minus 8% is **+5 percentage points**, not a 5% return.
The ratio of those rates is an adjusted relative risk, supplied as a secondary measure.

This controls broad quarter composition, not all other market factors. The compared populations
still differ by feature because validity and availability differ. Counts excluded for insufficient
within-quarter comparison are visible in the saved output.
''')
code('6. Define safe per-feature reading and quarter-adjusted statistics', '''
def feature_metadata(feature):
    # Strip transform and lookback tokens only; retain the entire exact column for the main ranking.
    parts = feature.split('__')
    core = parts[2]
    parent = re.sub(r'_(percentile|robust_zscore|zscore|rz)$', '', core)
    return {'family': parts[1], 'parent_sensor': parent,
            'lookbacks': '__'.join(parts[3:-1])}

def read_feature(feature):
    # Read two columns at a time from each file. Decimal Arrow types are cast safely to float64.
    pieces = []
    for name, pf in PFS.items():
        data = np.empty(pf.metadata.num_rows, dtype=np.float64)
        offset = 0
        for batch in pf.iter_batches(batch_size=65536, columns=[feature, VALID_COL[feature]], use_threads=True):
            values = pc.cast(batch.column(0), pa.float64(), safe=False).to_numpy(zero_copy_only=False).copy()
            valid = pc.fill_null(batch.column(1), False).to_numpy(zero_copy_only=False)
            values[~valid | ~np.isfinite(values)] = np.nan
            data[offset:offset+len(values)] = values
            offset += len(values)
        assert offset == len(data)
        pieces.append(data)
    values = np.concatenate(pieces)[ORDER]
    values[~AVAILABLE[AVAIL_COL[feature]]] = np.nan
    return values

def band_mask(values, low, high):
    # Every interval is left-inclusive, right-exclusive. Infinite endpoints cover future extremes.
    return np.isfinite(values) & (values >= low) & (values < high)

def block_counts(values, period, low, high):
    valid = np.isfinite(values)
    condition = band_mask(values, low, high)
    other = valid & ~condition
    b, y, n = period['row_b'], period['y'], period['n_blocks']
    return np.column_stack([
        np.bincount(b[condition], minlength=n),
        np.bincount(b[condition], weights=y[condition], minlength=n),
        np.bincount(b[other], minlength=n),
        np.bincount(b[other], weights=y[other], minlength=n),
    ]).astype(float)

def adjusted_stats(counts, period):
    # Columns: condition rows, condition events, control rows, control events.
    qstats = np.zeros((len(period['quarters']), 4))
    np.add.at(qstats, period['block_q'], counts)
    cn, ce, on, oe = qstats.T
    eligible = (cn >= CFG['min_rows_per_group_per_quarter']) & (on >= CFG['min_rows_per_group_per_quarter'])
    other_rate = np.divide(oe, on, out=np.full_like(oe, np.nan), where=on>0)
    cond_rate = np.divide(ce, cn, out=np.full_like(ce, np.nan), where=cn>0)
    retained = np.isin(period['block_q'], np.flatnonzero(eligible))
    n = float(cn[eligible].sum())
    events = float(ce[eligible].sum())
    observed = events/n if n else np.nan
    expected = float(np.sum(cn[eligible]*other_rate[eligible]))/n if n else np.nan
    total_condition = float(cn.sum())
    result = {
        'coverage': float(counts[:,[0,2]].sum()/len(period['idx'])),
        'condition_rows_all': int(total_condition),
        'condition_rows': int(n), 'condition_events': int(events),
        'control_rows': int(on[eligible].sum()),
        'comparable_quarters': int(eligible.sum()),
        'quarters_positive': int(np.sum((cond_rate > other_rate) & eligible)),
        'condition_blocks': int(np.sum((counts[:,0]>0) & retained)),
        'positive_blocks': int(np.sum((counts[:,1]>0) & retained)),
        'largest_positive_block_share': float(counts[retained,1].max()/events) if events else np.nan,
        'retained_condition_fraction': n/total_condition if total_condition else 0.0,
        'event_rate': observed, 'matched_rate': expected,
        'adjusted_rd_pp': 100*(observed-expected),
        'adjusted_rr': observed/expected if expected > 0 else np.nan,
    }
    quarter_table = pd.DataFrame({'quarter': period['quarters'], 'condition_rows': cn,
        'condition_events': ce, 'control_rows': on, 'control_events': oe,
        'condition_rate': cond_rate, 'control_rate': other_rate, 'comparable': eligible,
        'risk_difference_pp': 100*(cond_rate-other_rate)})
    return result, quarter_table

def support_reasons(stats, minimum_quarters):
    # Keep the rejection reasons explicit so exclusions can be audited.
    rules = [('coverage', stats['coverage'] >= CFG['min_coverage']),
             ('condition_rows', stats['condition_rows'] >= CFG['min_condition_rows']),
             ('control_rows', stats['control_rows'] >= CFG['min_control_rows']),
             ('comparable_quarters', stats['comparable_quarters'] >= minimum_quarters),
             ('condition_blocks', stats['condition_blocks'] >= CFG['min_condition_blocks'])]
    return [name for name, passed in rules if not passed]
''')
code('7. Define weekly block resampling and the ranking score', '''
BOOTSTRAP_WEIGHTS = {}
def bootstrap_weights(period_name):
    # Draw blocks with replacement separately inside each quarter, preserving quarterly representation.
    if period_name in BOOTSTRAP_WEIGHTS:
        return BOOTSTRAP_WEIGHTS[period_name]
    p = PERIODS[period_name]
    rng = np.random.default_rng(CFG['seed'] + {'learn':0, 'rank':1, 'validation':2}[period_name])
    weights = np.zeros((CFG['bootstrap_replicates'], p['n_blocks']), dtype=float)
    for q in range(len(p['quarters'])):
        ids = np.flatnonzero(p['block_q'] == q)
        weights[:,ids] = rng.multinomial(len(ids), np.repeat(1/len(ids),len(ids)), size=len(weights))
    BOOTSTRAP_WEIGHTS[period_name] = weights
    return weights

def bootstrap_effect(counts, period_name):
    # Freeze the observed comparable quarters. A replicate is usable only if every such quarter
    # still contains both groups; do not silently switch populations between replicates.
    p = PERIODS[period_name]
    _, qt = adjusted_stats(counts, p)
    selected = np.flatnonzero(qt['comparable'].to_numpy())
    if not len(selected):
        return {'rd_p05_pp':np.nan,'rd_p95_pp':np.nan,'bootstrap_valid_fraction':0.0}
    w = bootstrap_weights(period_name)
    cn_total = np.zeros(len(w)); ce_total = np.zeros(len(w)); expected_total = np.zeros(len(w))
    valid = np.ones(len(w), dtype=bool)
    for q in selected:
        ids = np.flatnonzero(p['block_q'] == q)
        z = w[:,ids] @ counts[ids]
        cn, ce, on, oe = z.T
        valid &= (cn>0) & (on>0)
        control_rate = np.divide(oe,on,out=np.zeros_like(oe),where=on>0)
        cn_total += cn; ce_total += ce; expected_total += cn*control_rate
    valid &= cn_total > 0
    effects = 100*(ce_total[valid]-expected_total[valid])/cn_total[valid]
    if not len(effects):
        return {'rd_p05_pp':np.nan,'rd_p95_pp':np.nan,'bootstrap_valid_fraction':0.0}
    lower = CFG['bootstrap_lower_quantile']
    return {'rd_p05_pp':float(np.quantile(effects,lower)),
            'rd_p95_pp':float(np.quantile(effects,1-lower)),
            'bootstrap_valid_fraction':float(valid.mean())}
''')
md('''
### Read the score correctly
If the observed adjusted increase is +4 points but its resampled 5th percentile is −1 point,
the association is less stable than its headline increase suggests. The **ranking score is that
5th-percentile value**; higher is better. The 5th–95th percentile band is a descriptive
resampling interval, not a post-selection confidence guarantee or a multiple-testing correction.

Seven-day blocks reduce the mistake of treating overlapping minute outcomes as independent.
They do not eliminate dependence across block boundaries; short quarter-edge blocks remain.
The seven-day choice is a research setting, not a proven optimal block length. Do not interpret
the resampled fraction or interval as a causal probability, trade win rate or formal p-value.
''')
code('8. Check the core calculations on small known examples', '''
# These checks run before the expensive data scan and catch sign, interval and denominator errors.
toy_n = max(100, CFG['min_rows_per_group_per_quarter'])
toy_p = {'idx':np.arange(4*toy_n), 'quarters':['Q1','Q2'], 'block_q':np.array([0,1])}
toy_counts = toy_n*np.array([[1,.2,1,.1], [1,.4,1,.2]], dtype=float)
toy, toy_q = adjusted_stats(toy_counts, toy_p)
assert np.isclose(toy['event_rate'], 0.30)
assert np.isclose(toy['matched_rate'], 0.15)
assert np.isclose(toy['adjusted_rd_pp'], 15)
assert np.isclose(toy['adjusted_rr'], 2)
assert band_mask(np.array([np.nan,-1,0,1,2]),0,2).tolist() == [False,False,True,True,False]
assert (np.array([-501,500,501]) > CFG['event_threshold']).tolist() == [False,False,True]
print('Known-answer checks passed: target, band endpoints and quarter-adjusted arithmetic.')
''')
code('9. Learn one value band per feature using early data only', '''
def learn_band(values):
    p = PERIODS['learn']
    x = values[p['idx']]
    finite = x[np.isfinite(x)]
    if len(finite) < CFG['min_condition_rows'] + CFG['min_control_rows']:
        return None, 'too_few_valid_learning_rows'
    if len(np.unique(finite)) < 2:
        return None, 'constant_learning_feature'
    # Interior quantiles define boundaries; use infinite outer endpoints to handle later new extremes.
    cuts = np.unique(np.quantile(finite,np.linspace(0,1,CFG['quantile_bands']+1)[1:-1]))
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    candidates = []
    for band_id,(low,high) in enumerate(zip(edges[:-1],edges[1:]),start=1):
        counts = block_counts(x,p,low,high)
        stats,_ = adjusted_stats(counts,p)
        if support_reasons(stats,CFG['min_train_quarters']):
            continue
        if not np.isfinite(stats['adjusted_rd_pp']) or stats['adjusted_rd_pp'] <= 0:
            continue
        candidates.append({'band_id':band_id,'low':float(low),'high':float(high),
                           'number_of_bands':len(edges)-1,'learning_stats':stats})
    if not candidates:
        return None, 'no_positive_supported_learning_band'
    # Deterministic tie-break: more condition rows, then lower band number. No ranking-period tuning.
    best = max(candidates,key=lambda r:(r['learning_stats']['adjusted_rd_pp'],
                                       r['learning_stats']['condition_rows'],-r['band_id']))
    return best, 'ok'

def analyze_feature(feature):
    values = read_feature(feature)
    base = {'feature':feature, **feature_metadata(feature)}
    chosen,reason = learn_band(values)
    if chosen is None:
        return {**base,'status':reason}
    base.update({k:v for k,v in chosen.items() if k != 'learning_stats'})
    base.update({'learn_'+k:v for k,v in chosen['learning_stats'].items()})
    p = PERIODS['rank']
    counts = block_counts(values[p['idx']],p,chosen['low'],chosen['high'])
    stats,_ = adjusted_stats(counts,p)
    base.update({'rank_'+k:v for k,v in stats.items()})
    reasons = support_reasons(stats,CFG['min_rank_quarters'])
    if reasons:
        return {**base,'status':'rank_support_failed:' + ','.join(reasons)}
    if not np.isfinite(stats['adjusted_rd_pp']) or stats['adjusted_rd_pp'] <= 0:
        return {**base,'status':'no_positive_ranking_association'}
    uncertainty = bootstrap_effect(counts,'rank')
    base.update({'rank_'+k:v for k,v in uncertainty.items()})
    if uncertainty['bootstrap_valid_fraction'] < CFG['min_valid_bootstrap_fraction']:
        return {**base,'status':'too_many_unusable_bootstrap_replicates'}
    return {**base,'status':'eligible'}

print('Feature-analysis functions are ready. No ranking or validation data select the learning band.')
''')
code('10. Screen every F4Z feature with checkpoints and progress', '''
# Same inputs and rules reuse the same folder. If you change algorithm code, bump method_version.
records = []
started = time.time()
for i,feature in enumerate(FEATURES,start=1):
    checkpoint = CHECKPOINT_DIR / (hashlib.sha256(feature.encode()).hexdigest()[:20] + '.json')
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        assert record['feature'] == feature
    else:
        try:
            record = analyze_feature(feature)
        except Exception as exc:
            # Do not hide a broken read as a weak feature or silently issue a partial top ten.
            print('FAILED FEATURE:',feature)
            raise RuntimeError(f'Screening stopped; rerun to resume after fixing {feature}') from exc
        # Python JSON permits nonfinite float tokens; CSV exports are the portable result tables.
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2))
        temporary.replace(checkpoint)
    records.append(record)
    if i == 1 or i % 10 == 0 or i == len(FEATURES):
        eligible_n = sum(r['status']=='eligible' for r in records)
        print(f'{i}/{len(FEATURES)} processed | eligible: {eligible_n} | elapsed: {(time.time()-started)/60:.1f} min')
        pd.DataFrame(records).to_csv(RUN_DIR / 'screening_progress.csv',index=False)
        gc.collect()

ALL_RESULTS = pd.DataFrame(records)
assert len(ALL_RESULTS) == len(FEATURES) and ALL_RESULTS['feature'].is_unique
ALL_RESULTS.to_csv(RUN_DIR / 'all_feature_results.csv',index=False)
display(ALL_RESULTS['status'].value_counts().rename_axis('status').to_frame('features'))
''')
code('11. Freeze and display the ten highest-ranked eligible columns', '''
eligible = ALL_RESULTS.loc[ALL_RESULTS['status'].eq('eligible')].copy()
if eligible.empty:
    raise RuntimeError('No feature meets the predeclared rules. See all_feature_results.csv; do not invent a top ten.')

# Rank by the conservative resampled score, then observed effect, coverage and exact name.
RANKED = eligible.sort_values(
    ['rank_rd_p05_pp','rank_adjusted_rd_pp','rank_coverage','feature'],
    ascending=[False,False,False,True]).reset_index(drop=True)
RANKED.insert(0,'association_rank',np.arange(1,len(RANKED)+1))
TOP10 = RANKED.head(CFG['top_k']).copy()
def format_bound(x):
    return '-infinity' if x == -np.inf else ('+infinity' if x == np.inf else f'{x:.8g}')
TOP10['selected_condition'] = TOP10.apply(lambda r:f"{format_bound(r['low'])} <= value < {format_bound(r['high'])}",axis=1)
TOP10['evidence_note'] = np.where(TOP10['rank_rd_p05_pp']>0,
    'positive lower resampling bound; still selection-biased',
    'positive observed effect; resampling interval reaches zero or below')
RANKED.to_csv(RUN_DIR / 'all_eligible_ranked.csv',index=False)
TOP10.to_csv(RUN_DIR / 'top10_frozen_discovery.csv',index=False)
# This auxiliary table is not the main exact-column top ten and is not validated below.
RANKED.drop_duplicates(['family','parent_sensor']).head(10).to_csv(
    RUN_DIR / 'top10_distinct_parent_candidates.csv',index=False)
frozen_rules = []
for _,row in TOP10.iterrows():
    # Encode infinite endpoints explicitly instead of silently converting them to JSON null.
    frozen_rules.append({'feature':row['feature'],'association_rank':int(row['association_rank']),
        'low': '-inf' if row['low'] == -np.inf else float(row['low']),
        'high': 'inf' if row['high'] == np.inf else float(row['high']),
        'interval':'left-inclusive, right-exclusive'})
(RUN_DIR / 'frozen_top10.json').write_text(json.dumps(frozen_rules,indent=2,allow_nan=False))
display(TOP10[['association_rank','feature','selected_condition','rank_coverage',
    'rank_event_rate','rank_matched_rate','rank_adjusted_rd_pp','rank_rd_p05_pp','rank_rd_p95_pp',
    'rank_adjusted_rr','rank_comparable_quarters','rank_quarters_positive','rank_condition_blocks','evidence_note']])
if len(TOP10) < CFG['top_k']:
    warnings.warn(f'Only {len(TOP10)} features met the rules. Reporting fewer than ten is intentional.')
print('Top list is FROZEN before reserved-period feature evaluation.')
print('Rates and coverage are fractions (0.13 = 13%); rd_pp columns are percentage points.')
''')
code('12. Evaluate the frozen top ten in the reserved later period', '''
validation_records = []
quarterly_records = []
for _,r in TOP10.iterrows():
    feature = r['feature']
    values = read_feature(feature)
    # Re-evaluate the identical band; do not fit new quantiles or substitute another feature.
    for period_name in ['rank','validation']:
        p = PERIODS[period_name]
        counts = block_counts(values[p['idx']],p,r['low'],r['high'])
        stats,qt = adjusted_stats(counts,p)
        qt.insert(0,'feature',feature);qt.insert(1,'period',period_name)
        quarterly_records.append(qt)
        if period_name == 'validation':
            unc = bootstrap_effect(counts,period_name)
            # Validation has only about three quarters in the supplied history.
            reasons = support_reasons(stats,min(2,len(p['quarters'])))
            stable_bootstrap = unc['bootstrap_valid_fraction'] >= CFG['min_valid_bootstrap_fraction']
            if not stable_bootstrap:
                reasons.append('bootstrap_support')
            note = 'limited support: '+','.join(reasons) if reasons else (
                'positive observed association' if stats['adjusted_rd_pp']>0 else 'association did not remain positive')
            validation_records.append({'feature':feature,'validation_note':note,
                **{'validation_'+k:v for k,v in {**stats,**unc}.items()}})

FINAL = TOP10.merge(pd.DataFrame(validation_records),on='feature',how='left',validate='one_to_one')
assert FINAL['feature'].tolist() == TOP10['feature'].tolist(), 'Validation must not change the ranking.'
QUARTERLY = pd.concat(quarterly_records,ignore_index=True)
FINAL.to_csv(RUN_DIR / 'top10_with_validation.csv',index=False)
QUARTERLY.to_csv(RUN_DIR / 'top10_quarterly_details.csv',index=False)
display(FINAL[['association_rank','feature','rank_adjusted_rd_pp','validation_adjusted_rd_pp',
               'validation_rd_p05_pp','validation_rd_p95_pp','validation_coverage','validation_note']])
print('Negative/weak later results remain visible. The top ten were not replaced or re-ranked.')
''')
code('13. Plot ranked effects and later-period quarterly behavior', '''
# Labels retain rank and a compact feature name; the CSV contains each complete original column.
labels = [f"{int(r.association_rank)}. {r.feature.removeprefix('f4z__').removesuffix('__value')}" for r in FINAL.itertuples()]
y = np.arange(len(FINAL))
fig,ax = plt.subplots(figsize=(13,max(5,len(FINAL)*0.55)))
ax.hlines(y,FINAL['rank_rd_p05_pp'],FINAL['rank_rd_p95_pp'],color='#087f8c',lw=3,label='Ranking-period 5th–95th block-resampling band')
ax.scatter(FINAL['rank_adjusted_rd_pp'],y,color='#18354a',s=45,label='Ranking-period observed increase',zorder=3)
ax.scatter(FINAL['validation_adjusted_rd_pp'],y,color='#c27832',marker='D',s=40,label='Reserved-period observed increase',zorder=3)
ax.axvline(0,color='black',ls='--',lw=1)
ax.set_yticks(y,labels);ax.invert_yaxis()
ax.set_xlabel('Quarter-adjusted event-frequency increase (percentage points)')
ax.set_title('Frozen top features: historical association, not trading returns')
ax.legend(loc='best',fontsize=9);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(RUN_DIR/'top10_effects.png',dpi=170,bbox_inches='tight');plt.show()

# Cells without sufficient comparison rows are shown as missing, never as successful quarters.
heat = QUARTERLY.copy()
heat.loc[~heat['comparable'],'risk_difference_pp'] = np.nan
heat['period_quarter'] = heat['period'] + ' | ' + heat['quarter']
pivot = heat.pivot(index='feature',columns='period_quarter',values='risk_difference_pp').reindex(FINAL['feature'])
fig,ax = plt.subplots(figsize=(12,max(4,len(FINAL)*0.45)))
limit = max(1,float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else 1)
im = ax.imshow(pivot.to_numpy(),aspect='auto',cmap='RdYlGn',vmin=-limit,vmax=limit)
ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=45,ha='right')
ax.set_yticks(range(len(labels)),labels)
for i in range(len(pivot)):
    for j in range(len(pivot.columns)):
        value = pivot.iloc[i,j]
        ax.text(j,i,'N/A' if pd.isna(value) else f'{value:+.1f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax,label='Within-quarter event-rate difference (percentage points)')
ax.set_title('Does the selected condition recur across quarters?')
fig.tight_layout();fig.savefig(RUN_DIR/'top10_quarterly_heatmap.png',dpi=170,bbox_inches='tight');plt.show()
''')
code('14. Save a readable final summary and confirm completion', '''
lines = ['# Alternative F4Z association ranking',
    'Target: BTCUSDT forward 48-hour signed return strictly above +500 log bps.',
    'Ranking: descending 5th-percentile weekly-block-resampled, quarter-adjusted risk difference.',
    'Bands learned before the ranking period; reserved validation never changes the ordering.',
    f'Features examined: {len(FEATURES)}. Eligible: {len(RANKED)}. Reported: {len(FINAL)}.',
    '', '## Frozen results']
for _,r in FINAL.iterrows():
    lines += [f"### {int(r['association_rank'])}. {r['feature']}",
              f"Condition: {r['selected_condition']}",
              f"Ranking effect: {r['rank_adjusted_rd_pp']:.3f} percentage points; score: {r['rank_rd_p05_pp']:.3f}.",
              f"Reserved-period effect: {r['validation_adjusted_rd_pp']:.3f} points; {r['validation_note']}.",
              f"Evidence: {r['evidence_note']}", '']
lines += ['## Interpretation limits',
    'This is individual-feature association, not causation or a combined trading strategy.',
    'The resampling bands are not selection-adjusted significance statements.',
    'Multiple testing, related columns, changing coverage and dependence across blocks remain.',
    'Data exports are not independently verified against their upstream build.',
    'No transaction costs, drawdowns, position sizing or return expectancy were tested.']
summary_text = '\\n'.join(lines)
(RUN_DIR/'READ_ME_RESULTS.md').write_text(summary_text)
display(Markdown(summary_text))
print('Completed. Open these files in Google Drive:')
for name in ['top10_with_validation.csv','top10_frozen_discovery.csv','all_feature_results.csv',
             'top10_quarterly_details.csv','top10_effects.png','READ_ME_RESULTS.md']:
    print(RUN_DIR/name)
''')
md('''
## 15. Methods references and interpretation notes
### Project evidence
- `Notebooks/01_Data_Inspection.ipynb`: supplied file names, target audit, 48-hour horizon,
  timestamp checks and original discovery/validation boundaries.
- `Notebooks/02_F4Z_Feature_Screening.ipynb`, Steps 12A and 12L: within-quarter comparison
  arithmetic and the documented canonical log-return target. This notebook implements a
  different selection protocol; it does not reproduce the earlier top three by construction.
- `RoadMap/My_RoadMap.pdf`: overlapping labels, incomplete validation and provenance limits.

### Technical references
- [Apache Arrow: ParquetFile and iter_batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)
  documents column-selective, batched Parquet reads used to avoid loading all 522 columns at once.
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  explains chronological evaluation and a train/test gap. Here the code uses explicit timestamp
  masks and two-day boundary exclusions, not the TimeSeriesSplit class.
- [SciPy: bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
  explains resampling distributions and interval terminology. The custom code here resamples
  whole temporal blocks within quarter strata; it does not call SciPy's independent-row bootstrap.

**Research choices specific to this notebook:** five bands, seven-day blocks, support gates,
split dates and ranking by the lower resampling quantile. The links document general tools and
concepts, not an endorsement or validation of these exact choices. The ranking period itself
is used to choose winners, so it is not an unbiased final performance estimate. The reserved
period is reported without replacement or re-ranking, even if the result is disappointing.
''')

notebook={'cells':cells,'metadata':{'colab':{'name':'03_Alternative_F4Z_Top10_Association.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'cell-{i:03d}'
path=ROOT/'Notebooks/03_Alternative_F4Z_Top10_Association.ipynb'
path.write_text(json.dumps(notebook,indent=2,ensure_ascii=False),encoding='utf-8')
print(path)
print('Cells:',len(cells),'Code cells:',sum(c['cell_type']=='code' for c in cells))
