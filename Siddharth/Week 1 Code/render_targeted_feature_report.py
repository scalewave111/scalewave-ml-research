"""Read saved results and create a reviewable report; no training."""
import json,html
import numpy as np,pandas as pd
import targeted_feature_extension as m
def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.6f}',escape=True,border=0)
def main():
 out=m.OUT;c=json.loads((out/'COMPLETE.json').read_text());audit=json.loads((out/'AUDIT.json').read_text())
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True);scores=pd.read_csv(out/'scores.csv')
 choices=json.loads((out/'SELECTIONS.json').read_text());trials=json.loads((out/'VALIDATION_TRIALS.json').read_text())
 fs=[];monthly=[];windows=[];family=[];gates=[]
 for d in choices:
  for col in d['features']:fs.append(dict(issue=d['issue'],selected=d['selected'],feature=col,role='original control' if col in m.PAIR else 'additional predictor'))
  windows.append(dict(issue=d['issue'][:10],pack=d['selected'],rf_weight=d['weight'],features=len(d['features']),train_rows=d['train_rows'],validation_rows=d['validation_rows'],test_rows=d['test_rows'],mae=d['test']['mae'],ic=d['test']['rank_ic'],baseline_mae=d['baseline_test']['mae'],baseline_ic=d['baseline_test']['rank_ic']))
  rows=[a for a in trials if a['issue']==d['issue']]
  gates.append(dict(issue=d['issue'][:10],candidates=len(rows),mae_pass=sum(a['mae']<a['baseline']['mae']-.01 for a in rows),ic_pass=sum(a['rank_ic']>a['baseline']['rank_ic']+.0001 for a in rows),two_months_pass=sum(a['monthly_mae_wins']>=2 for a in rows),same_weight_ablation_pass=sum(a['mae']<a['group_ablation']['mae']-.01 for a in rows),all_requirements_pass=sum(a['passed'] for a in rows)))
  for name in sorted(set(a['pack'] for a in rows)):
   best=min([a for a in rows if a['pack']==name],key=lambda a:a['mae'])
   family.append(dict(issue=d['issue'][:10],pack=name,best_inner_mae=best['mae'],ic_at_best_mae=best['rank_ic'],baseline_mae=best['baseline']['mae'],baseline_ic=best['baseline']['rank_ic'],any_joint_gate_pass=any(a['passed'] for a in rows if a['pack']==name)))
 for month,z in p.groupby(p.index.strftime('%Y-%m')):
  a=m.metric(z.prediction,z.truth);b=m.metric(z.baseline,z.truth)
  monthly.append(dict(month=month,rows=len(z),mae=a['mae'],baseline_mae=b['mae'],mae_gain=b['mae']-a['mae'],rank_ic=a['rank_ic'],baseline_ic=b['rank_ic']))
 pd.DataFrame(gates).to_csv(out/'validation_gate_summary.csv',index=False)
 pd.DataFrame(fs).to_csv(out/'selected_features.csv',index=False)
 pd.DataFrame(windows).to_csv(out/'window_scores.csv',index=False)
 pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False)
 pd.DataFrame(family).to_csv(out/'feature_pack_validation.csv',index=False)
 # Paired moving-block bootstrap of the identical final test rows; descriptive, not selection-adjusted.
 rng=np.random.default_rng(48);n=len(p);block=14*48;deltas=[]
 for _ in range(500):
  starts=rng.integers(0,n,size=int(np.ceil(n/block)))
  ix=np.concatenate([(np.arange(block)+s)%n for s in starts])[:n];z=p.iloc[ix]
  a=m.metric(z.prediction,z.truth);b=m.metric(z.baseline,z.truth)
  deltas.append([a['mae']-b['mae'],a['rank_ic']-b['rank_ic']])
 ci=dict(block_days=14,resamples=500,mae_difference_95=np.quantile(np.array(deltas)[:,0],[.025,.975]).tolist(),ic_difference_95=np.quantile(np.array(deltas)[:,1],[.025,.975]).tolist(),limitation='Nominal paired intervals; historical research selection not corrected')
 m.dump(out/'UNCERTAINTY.json',ci)
 data=json.loads((out/'DATA.json').read_text());a=m.metric(p.prediction,p.truth);b=m.metric(p.baseline,p.truth)
 status='Target reached on retrospective point estimates' if c['meets_under180_and_ic'] else 'Did not reach MAE below 180 with improved IC'
 text=f"""<!doctype html><html><head><meta charset="utf-8"><title>Controlled feature extension</title><style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:24px;color:#173348}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}}th{{background:#edf3f6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.result{{padding:20px;background:#edf3f6}}details{{margin-top:24px}}</style></head><body>
<h1>Additional market features: controlled pair + Random Forest extension</h1>
<p>Executed 3 November 2026. Horizon: 48 hours. Target: absolute endpoint log-return magnitude in bps. Audit: {audit['status']}.</p>
<div class="result"><b>{status}.</b><p>Test MAE {a['mae']:.6f}; Spearman IC {a['rank_ic']:.6f}. Matched baseline: {b['mae']:.6f} / {b['rank_ic']:.6f}. Original PDF MAE: 181.523868.</p></div>
<h2>What changed</h2><p>Only the Random Forest feature set is extended. The median-regression component keeps the original 5m and 15m liquidation activity features. The RF retains 128 trees, absolute-error splitting, depth 3, leaf minimum 128, bootstrap sample 4096, random seed 48. Expanding training history and four quarterly issue dates match the reconstructed stronger hybrid.</p>
<p>The candidate pool is the38unique original columns supplied by the user in two tables, spanning liquidation, flow, volatility, basis, funding and price. All38 names map exactly to dataset columns. The original5m/15m liquidation controls remain protected. Reported tableIC/p-values are reference evidence, not weights or admission scores. Engineered variants are derived from the eligible supplied columns. No other forecast horizon is trained.</p>
{table(pd.DataFrame([dict(family=k,candidates=v) for k,v in data['candidate_counts'].items()]))}
<h2>Training, validation and test scores</h2>{table(scores)}
<p>Training pools the selected outer models' in-sample predictions: overlapping timestamps repeat across expanding fits. Validation pools selected inner predictions and therefore is selection-exposed. Test rows are unique, identical to the baseline. Last-three-month scores are separately reported; the overall evaluation is the original full period.</p>
<h2>Walk-forward selections</h2>{table(pd.DataFrame(windows))}
<h2>Exact selected predictor columns</h2>{table(pd.DataFrame(fs))}
<h2>Filtering and selection</h2><p>Each inner training population supplies missingness, variance/uniqueness, Spearman association, mutual information, KS and top-decile event lift. Composite training ranks feed individual-family and pooled packs. Rank correlation above 0.98 and VIF above 20 remove redundant additions; the original pair is protected. Engineering includes past 1h/24h changes, trailing 24h z-scores, absolute values and pair interactions.</p>
<p>The preceding three calendar months form inner validation. Both training and validation labels must mature before their respective next boundary. A candidate must beat the original inner-selected hybrid on MAE and IC, improve MAE in at least two validation months, and beat the two-feature RF at the same blend weight. Weights are 10%, 25% or 50% RF; otherwise the original hybrid is retained. Selected extra-feature permutation diagnostics are saved separately. No outer outcomes select features or weights.</p>
<h2>Validation acceptance and rejection counts</h2>{table(pd.DataFrame(gates))}<p>Counts refer to feature-pack/weight candidates. A rejected candidate was not necessarily worse on every metric; all gates must pass together. Rejected candidates are not refitted on outer data.</p>
<h2>Monthly test results</h2>{table(pd.DataFrame(monthly))}
<details><summary>All feature-pack inner validation comparisons</summary>{table(pd.DataFrame(family))}</details>
<h2>Uncertainty and limitations</h2><pre>{html.escape(json.dumps(ci,indent=2))}</pre>
<p>This history has been examined in earlier research. Results are retrospective, not fresh holdout confirmation. No feature group is declared useless solely because it failed this specific RF test. These results apply only to 48h. New horizons require their own targets, folds, fitted models and baseline.</p>
<h2>Verification</h2><pre>{html.escape(json.dumps(audit,indent=2))}</pre>
<p>Source: Code/targeted_feature_extension.py. Inference: Code/predict_targeted_feature_extension.py. Exact artifacts, candidate features, validation scores, pruning reasons and model files are retained in this experiment directory.</p></body></html>"""
 (out/'RESULTS.html').write_text(text,encoding='utf-8')
 print(status,flush=True);print(scores.to_string(index=False),flush=True)
if __name__=='__main__':main()
