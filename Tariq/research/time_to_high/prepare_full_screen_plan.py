"""Calculate a capacity plan from saved receipts. No network or model fitting."""
import collections
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).parent
config = json.loads((root/'screen_config.json').read_text())
state = json.loads((root/'SCREEN_RESULTS.json').read_text())
jobs = json.loads((root/'pilot_fit_scores.json').read_text())
assert config == state['identity']['config']
def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
assert signature(state['identity']) == state['signature']
for key,job in jobs.items():
    assert signature(job) == state['jobs'][key]['score_sha256']
    assert job['screen_signature'] == state['signature']
    assert len(job['folds']) == len(config['folds'])
fits = [score for job in jobs.values() for score in job['fits'].values()]
completed = sum(s['fit_count'] for s in fits)
total_jobs = len(config['candidate_features'])*len(config['horizons'])
per_job = len(config['folds'])*(len(config['grid'])+1)
remaining = (total_jobs-len(jobs))*per_job
assert completed == 315 and len(jobs) == 9 and remaining == 134785
fit_seconds = sum(s.get('seconds',0) for s in fits)
assert abs(fit_seconds-state['summed_fit_seconds']) < 1e-6
loading = state['target_loading_seconds_this_session']+sum(state['family_loading_seconds_this_session'].values())
warm_wall = state['session_elapsed_seconds']-loading
# Latest worker results were pasted by the user, not independently fetched here.
thread_wall = 105.81872487200053
process_wall = 109.74834588199974
scenarios = [
    {'name':'pilot_fit_time_with_ideal_two_worker_utilization',
     'hours':remaining*(fit_seconds/completed)/2/3600,
     'assumption':'Same average per-fit cost as nine flow pilot jobs, perfect two-worker utilization, no overhead. Not a lower bound.'},
    {'name':'original_pilot_after_subtracting_recorded_loading',
     'hours':remaining*(warm_wall/completed)/3600,
     'assumption':'Same mixed-fold fit/checkpoint pace as original pilot; not a measured optimized full-run pace.'},
    {'name':'latest_fold_thread_benchmark_pace',
     'hours':remaining*(thread_wall/21)/3600,
     'assumption':'Apply latest-fold three-case pace to every fit; includes dispatch, excludes uploads. Not an upper bound.'},
]
for scenario in scenarios:
    scenario['equivalent_four_hour_compute_sessions'] = math.ceil(scenario['hours']/4)
family = collections.Counter(f.split('__')[1] for f in config['candidate_features'])
plan = {
    'status':'PLAN_READY_RUNNER_INTEGRATION_REQUIRED_NO_LAUNCH',
    'target':config['target'], 'pilot_signature':state['signature'],
    'scope':{'features':len(config['candidate_features']),'horizons':config['horizons'],
             'folds':len(config['folds']),'inner_settings':len(config['grid']),
             'total_feature_horizon_jobs':total_jobs,'maximum_total_fits':total_jobs*per_job,
             'reusable_completed_jobs':len(jobs),'reusable_completed_fits':completed,
             'remaining_jobs':total_jobs-len(jobs),'maximum_remaining_fits':remaining},
    'timing_evidence':{'pilot_fit_seconds':fit_seconds,'pilot_recorded_loading_seconds':loading,
                       'pilot_remaining_wall_seconds':warm_wall,
                       'worker_check_user_report':{'threads_seconds':thread_wall,'processes_seconds':process_wall,
                                                  'fits_per_backend':21,'equivalence':'PASS',
                                                  'remote_receipt_verified_locally':False}},
    'runtime_scenarios':scenarios,
    'runtime_limits':['Illustrative extrapolations, not confidence intervals or guaranteed bounds.',
                      'Only three flow features and three horizons measured; other families may cost more or less.',
                      'Ineligible training folds use recorded fallback and can reduce fit count.',
                      'Remaining source-cache loading, restarts, hardware variation and validation add time.',
                      'Four-hour sessions are a proposed soft work budget, not a promise about Colab availability.'],
    'execution_plan':{
        'compute':'Two persistent thread workers; one native thread per worker. Keep process alternative unpromoted.',
        'io':'At most two remote-read workers initially; one dedicated ordered Drive writer with bounded queue.',
        'order':'Flow first to reuse cache, then remaining families in frozen alphabetical order; features/horizons in frozen order. No score-based ordering.',
        'family_candidate_counts':dict(family),
        'first_authorized_batch':'At most four previously uncompleted feature-horizon jobs, all five folds and six inner settings. Retain them as part of full screen, not disposable tests.',
        'session_budget':'Four-hour soft deadline from session start including loading. Stop admitting work near deadline; finish active fits, flush and verify writes, then exit cleanly.',
        'resume':'User reruns Colab notebook. Validate source identities/config/software/receipt hashes, reload completed monthly caches, skip verified saved fits, recover missing aggregates from fold receipts.',
        'checkpoint':'Save compact fit scores and coverage after each fit; serialize immutable snapshots in one writer. Mark jobs complete only after durable verified upload. Unsaved active fits may restart.',
        'storage':'Reuse current compact monthly-checkpointed family/horizon caches. No local datasets, losing model binaries or row-level predictions.',
        'progress':'Stage messages, per-fit completion, regular elapsed/queue/remaining-work status. Treat timeouts as interruption, never as feature ineligibility or score fallback.',
        'matrix':'One row per target/feature/transformation/horizon with model, fold-selected settings, baseline/candidate errors, improvement, coverage, fallbacks and verification limitations.'},
    'quality_requirements':[
        'Keep frozen six-setting grid, five chronological folds and deterministic inner-only selection.',
        'Same baseline evaluation rows; train-only median imputation and missing indicator.',
        'Keep strict label-end purging, research-origin cutoff and final-six-month exclusion.',
        'Keep source validity/availability/baseline-count gates and provisional-data limitations.',
        'Retain independent prediction traversal and exact saved-score equivalence checks.',
        'Do not reduce folds, sample rows, weaken criteria, or drop difficult rows to meet a time budget.'],
    'required_before_launch':[
        'Integrate prepared-array reuse, persistent threading and ordered checkpointing into resumable full runner; current optimized modules are diagnostics only.',
        'Verify saved remote worker/profile receipts against packaged source hashes.',
        'Implement explicit compatibility manifest: preserve historical pilot identity and all nine job hashes; record actual reference/optimized implementation hashes separately. Never relabel old receipts or bypass mismatched identities.',
        'Test interruption after fit upload/before aggregate, failed upload, stale snapshot ordering, identity/version rejection and zero-fit resume of completed work on synthetic data.',
        'Package pilot-safe Colab notebook with an explicit disabled full-launch switch and session limits.',
        'Obtain explicit authorization for the first new batch/full expansion; this plan does not authorize training.'],
    'full_screen_authorized':False,
    'next_target':'time_to_low_min after time_to_high_min feature/set/transformation research',
    'eventual_system':'Allow connected specialist models when chronological evaluation supports them; one shared model is optional.'}
out = root/'full_screen_execution_plan.json'
if out.exists(): json.loads(out.read_text())
out.write_text(json.dumps(plan,indent=2)+'\n')
print('Remaining jobs:',plan['scope']['remaining_jobs'],'Maximum remaining fits:',remaining)
for s in scenarios: print(s['name'],round(s['hours'],1),'hours;',s['equivalent_four_hour_compute_sessions'],'four-hour equivalents')
print('Saved execution plan. No training, downloads or source/result changes.')
