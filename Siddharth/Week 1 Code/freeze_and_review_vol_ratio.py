"""Freeze the fixed-feature recipe and review existing predictions; never trains."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr,pearsonr

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'experiments/baseline_single_feature_v6'
OUT=HERE/'experiments/vol_ratio_frozen_review_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metrics(frame):
 y=frame.truth.to_numpy();b=frame.baseline.to_numpy();c=frame.vol_ratio.to_numpy()
 return dict(rows=len(frame),baseline_mae=float(abs(b-y).mean()),candidate_mae=float(abs(c-y).mean()),
  mae_change=float(abs(c-y).mean()-abs(b-y).mean()),
  baseline_spearman_ic=float(spearmanr(np.round(b,10),y).statistic) if np.ptp(b)>0 else 0.,
  candidate_spearman_ic=float(spearmanr(np.round(c,10),y).statistic) if np.ptp(c)>0 else 0.,
  baseline_pearson_ic=float(pearsonr(b,y).statistic) if np.ptp(b)>0 else 0.,
  candidate_pearson_ic=float(pearsonr(c,y).statistic) if np.ptp(c)>0 else 0.)

def run():
 if (OUT/'COMPLETE.json').exists():print((OUT/'COMPLETE.json').read_text());return
 OUT.mkdir(exist_ok=True);(OUT/'reference_checkpoints').mkdir(exist_ok=True)
 p=pd.read_csv(SOURCE/'predictions.csv',index_col=0,parse_dates=True);p.index=p.index.as_unit('ns')
 assert p.index.is_unique and len(p)==18481
 completed=json.loads((SOURCE/'COMPLETE.json').read_text());assert completed['audit']['status']=='PASS'
 result=metrics(p)
 np.testing.assert_allclose(result['candidate_mae'],completed['test_scores']['vol_ratio']['mae'],atol=1e-10,rtol=0)
 np.testing.assert_allclose(result['baseline_mae'],completed['test_scores']['baseline']['mae'],atol=1e-10,rtol=0)
 contract=json.loads((SOURCE/'CONTRACT.json').read_text())
 _,y,_,_=joblib.load(SOURCE/'prepared.joblib')
 decisions=json.loads((SOURCE/'SELECTIONS.json').read_text());weights=[]
 for d in decisions:
  candidate=next(a for a in d['candidates'] if a['name']=='vol_ratio')
  weights.append(dict(issue=d['issue'],baseline_rf_weight=d['baseline_weight'],candidate_rf_weight=candidate['weight']))
 baseline=HERE.parent/'Experiments/nested_boost_48h_v1/models'
 paths={'baseline_pair':baseline/'2024-10__report_outer_pair__expanding.joblib',
  'baseline_rf':baseline/'2024-10__report_outer_rf__expanding.joblib',
  'candidate_pair':SOURCE/'20241001/vol_ratio_pair_outer.joblib',
  'candidate_rf':SOURCE/'20241001/vol_ratio_rf_outer.joblib'}
 checkpoints={};parameters={}
 for name,path in paths.items():
  dst=OUT/'reference_checkpoints'/f'{name}.joblib';shutil.copy2(path,dst)
  assert sha(path)==sha(dst)
  checkpoints[name]=dict(source=str(path),copy=str(dst),sha256=sha(dst))
  a=joblib.load(dst);parameters[name]=a['model'].get_params()
 frozen=dict(status='FROZEN_AWAITING_NEW_DATA',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
  baseline_features=contract['original_features'],candidate_features=contract['original_features']+[contract['candidates']['vol_ratio']],
  adaptive_feature_selection=False,fixed_candidate='4h/24h normalized realized-volatility ratio in every window',
  feature_definition='(sqrt(sum of squared past 5m log returns over 4h)/sqrt(4)) / max(sqrt(sum over 24h)/sqrt(24),1e-6); endpoints strictly before forecast origin',
  target=contract['target'],preprocessing='Original training-only median imputation, mean/std scaling and missingness indicators',
  parameters=parameters,blend_grid=contract['weights'],blend_rule=contract['weight_selection'],
  freezing_scope='Freeze the complete existing learning procedure, including its original prior-quarter blend rule. No new feature, architecture, parameter-grid, or selection-rule search. Reference checkpoints are the archived October 2024 models, not newly trained models.',
  walk_forward='Retain expanding chronological quarterly fits and strict label_end-before-issuance purges for both pipelines',
  historical_weights=weights,reference_checkpoints=checkpoints,
  existing_evaluation_start=str(p.index.min()),existing_evaluation_end=str(p.index.max()),latest_exposed_label_end=str(y.label_end.max()),
  confirmation_status='NOT_RUN_NO_NEW_DATA',user_confirmed_no_new_dataset=True,
  confirmation_rule='First complete future calendar quarter with matched eligible features and matured 48h outcomes, strictly after the exposed outcome end. Register exact dates based only on availability before scoring outcomes. Apply both frozen procedures to identical rows; no outcome-driven changes.',
  required_data='Same 5m/15m liquidation features with causal availability; canonical-compatible 5m BTC returns or prices with >=24h feature history and >=48h outcome lookahead; chronology and label_end information',
  source_run_contract_sha256=sha(SOURCE/'CONTRACT.json'),source_predictions_sha256=sha(SOURCE/'predictions.csv'),review_source_sha256=sha(__file__))
 dump(OUT/'FROZEN_PROTOCOL.json',frozen)
 quarter=pd.DataFrame([dict(issue=issue,**metrics(f)) for issue,f in p.groupby('issue')]);quarter.to_csv(OUT/'window_checks.csv',index=False)
 months=pd.DataFrame([dict(month=month,**metrics(f)) for month,f in p.groupby(p.index.strftime('%Y-%m'))]);months.to_csv(OUT/'monthly_checks.csv',index=False)
 leaveout=pd.DataFrame([dict(excluded_window=issue,**metrics(p[p.issue!=issue])) for issue in p.issue.unique()]);leaveout.to_csv(OUT/'leave_one_window_out.csv',index=False)
 delta=(abs(p.vol_ratio-p.truth)-abs(p.baseline-p.truth)).to_numpy();uncertainty=[]
 for days in [7,14,28]:
  group=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=days).value
  blocks=pd.DataFrame(dict(delta=delta,group=group)).groupby('group').delta.agg(['sum','count'])
  ids=np.random.default_rng(2026).integers(0,len(blocks),(5000,len(blocks)))
  boot=blocks['sum'].to_numpy()[ids].sum(1)/blocks['count'].to_numpy()[ids].sum(1)
  ci=np.quantile(boot,[.025,.975])
  uncertainty.append(dict(block_days=days,blocks=len(blocks),mae_change=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
 intervals=pd.DataFrame(uncertainty);intervals.to_csv(OUT/'block_bootstrap_checks.csv',index=False)
 summary=dict(status='HISTORICAL_REVIEW_COMPLETE',fresh_confirmation='NOT_RUN_NO_NEW_DATA',no_training_performed=True,
  overall=result,winning_windows=int((quarter.mae_change<0).sum()),total_windows=len(quarter),
  winning_months=int((months.mae_change<0).sum()),total_months=len(months),
  positive_gain_after_every_window_exclusion=bool((leaveout.mae_change<0).all()),block_intervals=uncertainty,
  limitation='All checks reuse previously examined historical predictions. Leaving a window out here does not retrain models or create an untouched test. Intervals are descriptive and not adjusted for earlier feature/model searches.')
 dump(OUT/'COMPLETE.json',summary)
 text='Frozen baseline versus fixed volatility-ratio candidate: historical robustness review. No training and no fresh-data confirmation.\n\n'+json.dumps(summary,indent=2)
 text+='\n\nOriginal outer windows:\n'+quarter.to_string(index=False)+'\n\nMonthly results:\n'+months.to_string(index=False)
 text+='\n\nExclude-one-window diagnostic (existing predictions, no refits):\n'+leaveout.to_string(index=False)
 text+='\n\nBlock-length sensitivity:\n'+intervals.to_string(index=False)
 text+='\n\nNegative MAE change favors the candidate. The recipe, existing blend rule, reference models and hashes are preserved in FROZEN_PROTOCOL.json. Fresh confirmation is pending new matched data.\n'
 (OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
 context=HERE/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8');marker='## Volatility-ratio recipe frozen; fresh confirmation unavailable — 2026-11-04'
 if marker not in old:
  note='\n\n'+marker+'\n\nUser authorized freezing the fixed volatility-ratio candidate and comparing it with the original baseline on untouched data. The user confirmed that no newer dataset is available, then asked what could be done without additional data. The existing feature/evaluation origins end 2025-01-22 00:00 UTC, and their latest target end is 2025-01-24 00:00 UTC. Current canonical returns only cover that old evaluation. No fresh confirmation was run and no models were trained.\n\n'
  note+='The exact existing baseline and fixed-volatility-ratio procedures, including the original predefined blend rule, were frozen. Archived October 2024 model copies and hashes are preserved as reference checkpoints. Historical month/window, exclude-one-window and 7/14/28-day paired-block-bootstrap diagnostics were computed from saved predictions; they do not constitute untouched evidence.\n\n'
  note+=f"Historical MAE improvement persists in {summary['winning_windows']}/{summary['total_windows']} original windows and {summary['winning_months']}/{summary['total_months']} months. Improvement survives every individual window exclusion: {summary['positive_gain_after_every_window_exclusion']}.\n"
  note+='\nSources: `experiments/vol_ratio_frozen_review_v1/FROZEN_PROTOCOL.json`, `COMPLETE.json`, `RESULTS.txt`, `window_checks.csv`, `monthly_checks.csv`, `leave_one_window_out.csv`, and `block_bootstrap_checks.csv`. Script: `freeze_and_review_vol_ratio.py`. Future confirmation requires newer matched liquidation features and price/return data; do not relabel old data as untouched.\n'
  context.write_text(old+note,encoding='utf-8')
 print(json.dumps(summary,indent=2));print(quarter.to_string(index=False));print(leaveout.to_string(index=False))

if __name__=='__main__':run()
