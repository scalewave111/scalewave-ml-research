"""Report the ten-technique campaign without fitting models."""
from pathlib import Path
import json,html,argparse,re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import nested_ic_rf as data
import canonical_return_features as crf
from render_tree_magnitude_report import table,interval
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'Experiments/tree_improvement_campaign_v3'
def ic(a,b):
 return float(spearmanr(np.round(a,10),b).statistic) if np.ptp(a)>0 else 0.
def paired_interval(p):
 y=p.truth.to_numpy();a=p.prediction.to_numpy();b=p.reference.to_numpy()
 ids=((p.index-p.index[0]).total_seconds()//(14*86400)).astype(int)
 blocks=[np.flatnonzero(ids==v) for v in np.unique(ids)];rng=np.random.default_rng(48);samples=[]
 for _ in range(500):
  ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
  samples.append([float(np.mean(abs(a[ix]-y[ix])-abs(b[ix]-y[ix]))),ic(a[ix],y[ix])-ic(b[ix],y[ix])])
 lo,hi=np.quantile(samples,[.025,.975],axis=0)
 return dict(block_days=14,replicates=500,mae_difference=float(np.mean(abs(a-y)-abs(b-y))),mae_interval95=[float(lo[0]),float(hi[0])],ic_difference=ic(a,y)-ic(b,y),ic_interval95=[float(lo[1]),float(hi[1])],interpretation='Retrospective nominal paired block intervals; not adjusted for accumulated model searches')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--horizon',type=int,default=48);args=ap.parse_args();h=args.horizon;out=BASE/f'{h}h'
 if h!=48:raise ValueError('This report template documents the executed48h campaign; other horizons require their own report.')
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True);scores=pd.read_csv(out/'scores.csv')
 selections=json.loads((out/'selections.json').read_text());audit=json.loads((out/'AUDIT.json').read_text())
 complete=json.loads((out/'COMPLETE.json').read_text());monthly=[];components=[];features=[];gates=[];alltests=[]
 for month,g in p.groupby(p.index.strftime('%Y-%m')):
  monthly.append(dict(month=month,rows=len(g),model_mae=float(abs(g.prediction-g.truth).mean()),baseline_mae=float(abs(g.reference-g.truth).mean()),mae_difference=float(abs(g.prediction-g.truth).mean()-abs(g.reference-g.truth).mean()),model_ic=ic(g.prediction,g.truth),baseline_ic=ic(g.reference,g.truth),target_median=float(g.truth.median()),forecast_median=float(g.prediction.median())))
 for dec in selections:
  folder=out/pd.Timestamp(dec['issue']).strftime('%Y%m%d')
  for name,w in dec['weights'].items():
   cfg=dec['definitions'][name]['config'];cols=dec['definitions'][name]['columns']
   implicit='trailing168h volatility' if cfg=='lgb_normal' else 'trailing168h volatility and net return' if cfg=='extra_regime' else ''
   components.append(dict(issue=dec['issue'][:10],component=name,weight=w,predictor_columns='; '.join(cols),additional_transformation_or_routing_inputs=implicit))
  fs=json.loads((folder/'feature_selection.json').read_text())
  for rec in fs['admissions']:features.append(dict(issue=dec['issue'][:10],**rec))
  for rec in json.loads((folder/'feature_tests.json').read_text()):alltests.append(dict(issue=dec['issue'][:10],**rec))
  gates.append(dict(issue=dec['issue'][:10],**dec['validation_gate']))
 monthly=pd.DataFrame(monthly);monthly.to_csv(out/'monthly_scores.csv',index=False)
 pd.DataFrame(components).to_csv(out/'selected_components_and_features.csv',index=False)
 pd.DataFrame(features).to_csv(out/'feature_contributions.csv',index=False)
 pd.DataFrame(alltests).to_csv(out/'all_feature_tests.csv',index=False)
 pd.DataFrame(gates).to_csv(out/'validation_gates.csv',index=False)
 # Extreme targets are described, never excluded from reported MAE.
 decile=pd.qcut(p.truth,10,labels=False,duplicates='drop');errors=[]
 total_error=float(abs(p.prediction-p.truth).sum())
 for d,g in p.groupby(decile):
  errors.append(dict(target_decile=int(d)+1,rows=len(g),min_target=float(g.truth.min()),max_target=float(g.truth.max()),model_mae=float(abs(g.prediction-g.truth).mean()),baseline_mae=float(abs(g.reference-g.truth).mean()),absolute_error_share=float(abs(g.prediction-g.truth).sum()/total_error)))
 pd.DataFrame(errors).to_csv(out/'target_decile_errors.csv',index=False)
 weeks=((p.index-p.index[0]).total_seconds()//(7*86400)).astype(int);weekly=[]
 for _,g in p.groupby(weeks):
  weekly.append(dict(start=str(g.index.min()),end=str(g.index.max()),rows=len(g),model_mae=float(abs(g.prediction-g.truth).mean()),baseline_mae=float(abs(g.reference-g.truth).mean()),total_excess_absolute_error=float((abs(g.prediction-g.truth)-abs(g.reference-g.truth)).sum())))
 pd.DataFrame(weekly).sort_values('total_excess_absolute_error',ascending=False).to_csv(out/'weekly_error_episodes.csv',index=False)
 # Diagnostic regimes use strictly prior observed returns and train-only cutoffs.
 raw,y,dev,report=data.load();r=crf.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet')
 state=crf.features(r,raw.index);vol=state['PRICE__realized_vol_168h'];trend=r.rolling('168h',closed='left',min_periods=2016).sum().reindex(raw.index)
 if h!=48:y=crf.make_labels(r,raw.index,h)
 regime=[]
 for issue,g in p.groupby('issue'):
  start=pd.Timestamp(issue);tr=y.index[y.label_end<start];threshold=float(vol.loc[tr].median())
  names=pd.Series(np.where(vol.loc[g.index]>threshold,'high_vol','low_vol'),index=g.index)+'_'+pd.Series(np.where(trend.loc[g.index]>0,'positive_trend','nonpositive_trend'),index=g.index)
  names.loc[vol.loc[g.index].isna()|trend.loc[g.index].isna()]='missing'
  for name,ix in names.groupby(names).groups.items():
   z=g.loc[ix];regime.append(dict(issue=issue,regime=name,threshold=threshold,rows=len(z),model_mae=float(abs(z.prediction-z.truth).mean()),baseline_mae=float(abs(z.reference-z.truth).mean()),model_ic=ic(z.prediction,z.truth),baseline_ic=ic(z.reference,z.truth)))
 pd.DataFrame(regime).to_csv(out/'regime_errors.csv',index=False)
 uncertainty=paired_interval(p)
 (out/'PAIRED_UNCERTAINTY.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
 model=scores[scores.split=='test'].iloc[0];ref=scores[scores.split=='test_reference'].iloc[0]
 beats=bool(model.mae<ref.mae and model.rank_ic>ref.rank_ic and (h!=48 or model.mae<181.5238678328609))
 status='Improved both point estimates; fresh confirmation required' if beats else 'Did not beat the baseline on both MAE and IC'
 gate_count=sum(z['passed'] for z in gates);wins=int((monthly.mae_difference<0).sum())
 methods=pd.DataFrame([
 ['Feature selection','Eight tests; train-only screening, core pair/singleton comparison, global correlation pruning, add-one ablation and block permutation. At most two extra predictors admitted.'],
 ['Feature engineering','Past 24h means, 1h differences, standardized innovations, signed logs, cross-family products and trailing availability; causal canonical volatility/trend used as regime inputs.'],
 ['Hyperparameter tuning','Up to 20 fixed algorithm/pool recipes each month; tree families, depths2Ã¢â‚¬â€œ4, leaves384Ã¢â‚¬â€œ768,96Ã¢â‚¬â€œ160trees, 12-month versus expanding history. Duplicate pools skipped.'],
 ['Regularization','Shallow trees, large leaves, learning rate 0.025, L2=100 for boosting, feature caps and conservative tree ensembles.'],
 ['Target transformation','Log/sqrt targets guide forest split structure; leaves are then estimated as original-target medians. This avoids treating the inverse transformed mean as a median. Another candidate predicts a volatility-normalized target and reverses that normalization.'],
 ['Outlier analysis','Checked invalid/nonfinite labels, duplicates, target tails and error concentration. One candidate clips training split targets at their training 99.5th percentile. Every valid test target remains unmodified.'],
 ['More training data','Compared 12-month windows with all available data since April 2022, including 180-day recency weights. Monthly refits ingest newly matured labels. No additional historical feature dataset was obtained.'],
 ['Regime-aware models','High/low past 168h volatility and positive/nonpositive past 7-day return. Tree experts blend 50% with a global tree model; insufficiently populated states use global predictions.'],
 ['Ensemble learning','Single models, equal-weight top 3 and pair weights 25/50/75%, chosen only on inner forecasts. No non-tree model contributes to new predictions.'],
 ['Error analysis','Monthly, target-decile, weekly episode and observable-regime errors; full test MAE/IC, pooled train/validation scores and 14-day paired bootstrap intervals.']
 ],columns=['Technique','What was actually tested'])
 body=f"""<!doctype html><html><head><meta charset="utf-8"><title>Tree improvement campaign results</title><style>
body{{font:16px system-ui;max-width:1300px;margin:36px auto;padding:24px;color:#163247}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{border-bottom:1px solid #ccd6df;padding:9px;text-align:left;overflow-wrap:anywhere}}th{{background:#edf2f6}}pre{{white-space:pre-wrap;background:#f1f5f7;padding:18px}}.result{{border-left:5px solid #ab4545;background:#fff2ee;padding:20px}}details{{margin:22px 0}}
</style></head><body><h1>Ten-technique tree regression campaign: {h}-hour magnitude</h1>
<p>Executed on real data,3November2026. Audit: <b>{audit['status']}</b>. All new predictors are tree regressors or ensembles of tree regressors.</p>
<div class="result"><b>{status}.</b><p>Test MAE <b>{model.mae:.6f}bps</b>; Spearman IC <b>{model.rank_ic:.6f}</b>. Matched reference: MAE{ref.mae:.6f}, IC{ref.rank_ic:.6f}.</p></div>
<h2>Overall training, validation and testing</h2>{table(scores)}
<p>Training is in-sample performance on the intersection of each selected ensemble's training rows. Pooled training and validation predictions repeat timestamps across monthly models; unique counts are shown. Test rows are unique. Pearson IC and RMSE are additional metrics, not substitutes for Spearman IC and MAE.</p>
<p>For48hours, the original report gives MAE181.523868; its reconstruction gives181.525993 and supplies the matched IC. The older25%RF hybrid scored181.799753 and standalone historicalRF scored183.749584. This campaign does not beat either hybrid on MAE. The previous tree-only RF control scored182.521803, also below this campaign.</p>
<h2>Ten requested techniques</h2>{table(methods)}
<h2>Evaluation discipline</h2><p>Monthly outer issuances run fromJanuary2024 throughJanuary2025. Each uses two preceding three-month inner folds, with initial feature screens fitted before the first inner fold. All training labels end strictly before issuance; inner validation labels must mature before the outer issuance. Prediction timestamps and targets are unchanged in the final comparison. Genuine extreme events were retained.</p>
<p>The reference hybrid is used only for evaluation and inner selection. It is not a constituent of the new tree ensemble. An inner candidate must improve both MAE and IC, win at least four of six monthly MAE comparisons and have a positive mean monthly IC gain to pass the gate. If no tree candidate qualifies, the run still records the best-MAE tree challenger and evaluates it on all timestamps; that does not constitute promotion.</p>
<h2>What the errors show</h2><p>{gate_count} of 13 monthly selections passed the inner reference gate, but the final model won MAE in only{wins} of 13 outer months. The largest target decile accounts for{100*errors[-1]['absolute_error_share']:.2f}% of absolute error. These observations describe this historical run, not proof of a causal failure mechanism.</p>
{table(monthly)}
<p>June2024 is the largest monthly MAE deterioration: forecast median about195.68bps versus observed median142.52bps. March and July underprediction also persists. Monthly refitting and regime experts did not reliably track these shifts.</p>
<h2>Scope and limitations</h2><p>This new campaign executed48hours only. The runner accepts1/4/24/48/72/120hour labels from the verified canonical return source, but those other horizons have not been newly trained or tested here. Their older experiment results remain separate. Saved models are historical artifacts, not current live-trading models.</p>
<p>This period has been exposed by earlier experiments, so even an improvement would be retrospective evidence rather than fresh holdout confirmation. Several techniques changed together; the result does not establish a separate causal gain or loss for every technique. Nominal block-bootstrap intervals are not corrected for the accumulated model search.</p>
<details><summary>Selected components, weights and exact inputs</summary>{table(pd.DataFrame(components))}</details>
<details><summary>Feature contribution tests</summary>{table(pd.DataFrame(features).drop(columns=['permutations'],errors='ignore'))}</details>
<details><summary>Target-decile errors</summary>{table(pd.DataFrame(errors))}</details>
<details><summary>Regime errors</summary>{table(pd.DataFrame(regime))}</details>
<details><summary>Paired uncertainty</summary><pre>{html.escape(json.dumps(uncertainty,indent=2))}</pre></details>
<h2>Reproduction and inference</h2><pre>{html.escape(str(out))}
python Code/tree_improvement_campaign.py --horizons48
python Code/audit_tree_improvement_campaign.py --horizon48
python Code/render_tree_campaign_report.py --horizon48
python Code/predict_tree_campaign.py --model-folder Experiments/tree_improvement_campaign_v3/48h/20250101 --features YOUR_FEATURES.parquet --return-ends-csv YOUR_CANONICAL_RETURN_ENDS.csv --start YOUR_UTC_START --output predictions.csv</pre>
<p>Inference requires the same source feature columns,24hours of preceding feature history, and168hours of regular canonical five-minute returns indexed at their end times. It requires no future target labels. Optional return CSV columns: return_end, log_return_bps.</p>
<p>Real-data predictive fits:{complete['fit_count']}. One additional synthetic preflight fit tested median-leaf estimation. A pre-run interaction-schema error was fixed before any real-data fit. The audit checks label purges, features, median leaf estimates, selected ensembles, saved prediction replay and scores without refitting.</p></body></html>"""
 replacements={'data,3November2026':'data, 3 November 2026','}bps':'} bps','reference: MAE':'reference: MAE ',' IC0':' IC 0','For48hours':'For 48 hours','MAE181.523868':'MAE 181.523868','gives181.525993':'gives 181.525993','older25%RF':'older 25% RF','scored181.799753':'scored 181.799753','historicalRF':'historical RF','scored183.749584':'scored 183.749584','scored182.521803':'scored 182.521803','fromJanuary2024 throughJanuary2025':'from January 2024 through January 2025','of 13':'of 13','only4':'only 4','June2024':'June 2024','about195.68bps':'about 195.68 bps','median142.52bps':'median 142.52 bps','executed48hours':'executed 48 hours','accepts1/4/24/48/72/120hour':'accepts 1/4/24/48/72/120-hour','--horizons48':'--horizons 48','--horizon48':'--horizon 48','columns,24hours':'columns, 24 hours','and168hours':'and 168 hours','fits:':'fits: '}
 for a,b in replacements.items():body=body.replace(a,b)
 body=re.sub(r'(\d)bps',r'\1 bps',body)
 body=re.sub(r'only(?=\d)','only ',body)
 (out/'RESULTS.html').write_text(body,encoding='utf-8')
 summary=dict(status=status,horizon_hours=h,beats_both=beats,test_mae=float(model.mae),test_ic=float(model.rank_ic),reference_mae=float(ref.mae),reference_ic=float(ref.rank_ic),mae_difference=float(model.mae-ref.mae),ic_difference=float(model.rank_ic-ref.rank_ic),monthly_mae_wins=wins,inner_gate_passes=gate_count,fit_count=complete['fit_count'],audit=audit['status'])
 (out/'RESULT_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2));print('Report:',out/'RESULTS.html')
if __name__=='__main__':main()
