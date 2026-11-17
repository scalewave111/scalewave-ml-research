"""Render hybrid experiment results and predeclared all-model blend diagnostics."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
import all_model_hybrid as m
from render_tree_campaign_report import paired_interval
from render_tree_magnitude_report import table
def main():
 out=m.OUT;X=pd.read_csv(out/'component_predictions.csv',index_col=0,parse_dates=True)
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True);scores=pd.read_csv(out/'scores.csv')
 choices=json.loads((out/'SELECTIONS.json').read_text());registry=json.loads((out/'REGISTRY.json').read_text());rmap={r['name']:r for r in registry}
 audit=json.loads((out/'AUDIT.json').read_text());prep=json.loads((out/'PREPARATION.json').read_text())
 # These fixed candidate policies were specified before results, not selected on test MAE.
 fixed=p[['truth','baseline']].copy();fixed['all_model_equal']=fixed.baseline;fixed['family_balanced']=fixed.baseline
 weights=[];monthly=[];component_scores=[]
 for d in choices:
  date=pd.Timestamp(d['issue']);ix=p.index[(p.index>=date)&(p.index<date+pd.DateOffset(months=1))]
  names=d['eligible']
  if names:
   fixed.loc[ix,'all_model_equal']=.5*X.loc[ix,'baseline']+.5*X.loc[ix,names].mean(axis=1)
   groups=m.family_mix(names,rmap)
   family=pd.DataFrame({g:X.loc[ix,cols].mean(axis=1) for g,cols in groups.items()})
   fixed.loc[ix,'family_balanced']=.5*X.loc[ix,'baseline']+.5*family.mean(axis=1)
  for name,w in d['weights'].items():weights.append(dict(issue=d['issue'][:10],method=d['method'],component=name,weight=w))
  g=p.loc[ix];monthly.append(dict(month=d['issue'][:7],rows=len(g),hybrid_mae=m.metric(g.prediction,g.truth)['mae'],baseline_mae=m.metric(g.baseline,g.truth)['mae'],hybrid_ic=m.metric(g.prediction,g.truth)['rank_ic'],baseline_ic=m.metric(g.baseline,g.truth)['rank_ic'],method=d['method']))
 for name in X:
  valid=X[name].notna();y=p.loc[valid,'truth'];a=m.metric(X.loc[valid,name],y);b=m.metric(X.loc[valid,'baseline'],y)
  component_scores.append(dict(model=name,family='baseline' if name=='baseline' else rmap[name]['family'],**a,baseline_mae_same_rows=b['mae'],baseline_ic_same_rows=b['rank_ic']))
 fixed.to_csv(out/'fixed_all_model_blends.csv',index_label='decision_time')
 comparison=[]
 for name,values in [('selected_hybrid',p.prediction),('baseline',p.baseline),('all_model_equal',fixed.all_model_equal),('family_balanced',fixed.family_balanced)]:
  comparison.append(dict(model=name,**m.metric(values,p.truth)))
 pd.DataFrame(comparison).to_csv(out/'HYBRID_COMPARISON.csv',index=False)
 pd.DataFrame(weights).to_csv(out/'weights_by_month.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_scores.csv',index=False);pd.DataFrame(component_scores).to_csv(out/'component_scores.csv',index=False)
 q=p.rename(columns={'baseline':'reference'});ci=paired_interval(q)
 m.dump(out/'PAIRED_UNCERTAINTY.json',ci)
 a=m.metric(p.prediction,p.truth);b=m.metric(p.baseline,p.truth)
 win=a['mae']<181.5238678328609 and a['rank_ic']>b['rank_ic']
 status='Improves both retrospective point estimates; fresh confirmation required' if win else 'Does not beat the baseline on both MAE and IC'
 full=models_used=sorted({n for d in choices for n,w in d['weights'].items() if n!='baseline' and w>0})
 body=f"""<!doctype html><html><head><meta charset="utf-8"><title>All-model hybrid results</title><style>body{{font:16px system-ui;max-width:1350px;margin:35px auto;padding:24px;color:#173348}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:9px;border-bottom:1px solid #cbd5df;text-align:left;overflow-wrap:anywhere}}th{{background:#eaf0f5}}pre{{white-space:pre-wrap;background:#f1f5f8;padding:18px}}.result{{padding:20px;border-left:5px solid #3e6a88;background:#eef4f8}}details{{margin:24px 0}}</style></head><body>
<h1>Historical and current models combined: 48-hour hybrid</h1>
<p>Executed on 3 November 2026. All forecasts target the same BTC absolute endpoint log-return magnitude, in basis points. Audit: <b>{audit['status']}</b>.</p>
<div class="result"><b>{status}.</b><p>Selected hybrid test MAE <b>{a['mae']:.6f}</b>; Spearman IC <b>{a['rank_ic']:.6f}</b>. Matched baseline MAE <b>{b['mae']:.6f}</b>; IC <b>{b['rank_ic']:.6f}</b>.</p></div>
<h2>Direct comparison on all18,481 timestamps</h2>{table(pd.DataFrame(comparison))}
<p>The original baseline PDF reports181.523868bps; its reconstruction produces181.525993 and supplies the comparable IC. The older hybrid reported181.799753, with a25%RF /75%frozen-baseline blend selected on development data.</p>
<p>The equal all-model and family-balanced candidates allocate50% to the baseline and50% to all eligible forecasts or equally weighted model families. These are reported as fixed candidate diagnostics, not post-test replacements for the selected policy. All models were considered; the selected hybrid is allowed to give weak or duplicate candidates zero weight.</p>
<h2>What was combined</h2><p>The pool contains{prep['candidate_forecasts']} compatible candidate forecast streams plus the baseline: yesterday's Random Forest variants and historical hybrids; median-regression and rolling-baseline variants; CatBoost and LightGBM candidates; nested RF/boosting strategies; residual corrections; recent tree-only policies; and all registered current-campaign component recipes, including regime-aware ExtraTrees and volatility-normalized LightGBM. Only48-hour forecasts enter the pool. Other horizons are never averaged into it.</p>
<p>{prep['new_component_fits']} additional base-model fits completed missing historical component forecasts;{prep['reused_original_artifacts']} original current-campaign artifacts were reused. New fits used their original point-in-time training recipes. Existing files and historical results were preserved.</p>
<h2>How the hybrid was selected</h2><p>For each monthly issuance, the combiner considers only earlier outer forecasts with fully matured48-hour labels. Weight fitting uses the earlier three-month window; candidate validation uses the following two months. Individual models, whole families, equal all-model mixtures, and regularized nonnegative family-weight fits are tested. Challenger exposure is fixed to one of2.5%,5%,10%,20%,35%or50%; the baseline retains at least50%.</p>
<p>A candidate must reduce validation MAE by more than0.01bps, increase Spearman IC by more than0.0001, and improve MAE in both validation calendar months. Among qualifying candidates, the lowest-MAE candidate is selected. Otherwise the baseline is retained. January and February2024 are baseline-only warm-up months; there is no invented prehistory for today's models.</p>
<p>Models whose forecasts start in November2024 cannot inform earlier weights. Identical forecasts are merged using prior validation values only. Full-period duplicate diagnostics do not influence selection. Sparse historical forecast gaps, up to5%, use the baseline for validation; current selected forecasts must be complete.</p>
<h2>Overall scores</h2>{table(scores)}
<p>“Weight calibration” measures the blending layer on earlier out-of-sample base forecasts. It is not the constituent models' in-sample training score. Training and validation timestamps repeat across issuances; unique counts are shown. Unsupported early calibration rows are omitted rather than fabricated. The last-three-month comparison uses its own matched baseline; its MAE must not be compared directly with the full-period181.523868 target.</p>
<h2>Selected weights by month</h2>{table(pd.DataFrame(weights))}
<h2>Monthly test scores</h2>{table(pd.DataFrame(monthly))}
<h2>Limits on the conclusion</h2><p>This is retrospective research on an already inspected historical period. Some legacy recipe choices, including the181.799753 hybrid weight, were selected using development results. The new combiner does not fit weights on future test labels, but it cannot erase earlier selection exposure. An apparent improvement would require a fresh evaluation period before treating it as established.</p>
<p>Adding every model does not automatically improve performance. Zero weights and baseline-only months are deliberate validation outcomes. The displayed overall scores belong to the monthly hybrid policy, not one fixed global weight vector. Negative IC or worse MAE is not relabeled as success.</p>
<details><summary>Every component's retrospective score and matching-row reference</summary>{table(pd.DataFrame(component_scores))}</details>
<details><summary>Source inventory</summary>{table(pd.DataFrame(registry).drop(columns=['source_sha256'],errors='ignore'))}</details>
<details><summary>Paired14-day bootstrap uncertainty</summary><pre>{html.escape(json.dumps(ci,indent=2))}</pre></details>
<h2>Files and inference</h2><pre>{html.escape(str(out))}
Code/prepare_all_model_hybrid.py
Code/all_model_hybrid.py
Code/audit_all_model_hybrid.py
Code/predict_all_model_hybrid.py
Code/render_all_model_hybrid_report.py

python Code/predict_all_model_hybrid.py --selections Experiments/all_model_hybrid_v1/SELECTIONS.json --issue2025-01-01 --forecasts YOUR_ALIGNED_48H_FORECASTS.csv --output hybrid_predictions.csv</pre>
<p>Inference requires forecasts from the named baseline and nonzero-weight components. Weights are saved for historical months; live use needs newly trained base forecasts and newly calibrated weights from matured outcomes. No future target labels are needed to apply an already selected weight vector.</p></body></html>"""
 fixes={'all18,481':'all 18,481','reports181.523868bps':'reports 181.523868 bps','produces181.525993':'produces 181.525993','reported181.799753':'reported 181.799753','a25%RF /75%frozen-baseline':'a 25% RF / 75% frozen-baseline','allocate50%':'allocate 50%','and50%':'and 50%','contains73':'contains 73','Only48-hour':'Only 48-hour','forecasts;24':'forecasts; 24','matured48-hour':'matured 48-hour','one of2.5%,5%,10%,20%,35%or50%':'one of 2.5%, 5%, 10%, 20%, 35% or 50%','least50%':'least 50%','than0.01bps':'than 0.01 bps','than0.0001':'than 0.0001','February2024':'February 2024','November2024':'November 2024','to5%':'to 5%','full-period181.523868':'full-period 181.523868','the181.799753':'the 181.799753','Paired14-day':'Paired 14-day','--issue2025':'--issue 2025'}
 for x,z in fixes.items():body=body.replace(x,z)
 (out/'RESULTS.html').write_text(body,encoding='utf-8')
 m.dump(out/'RESULT_SUMMARY.json',dict(status=status,beats_both=win,test_mae=a['mae'],test_ic=a['rank_ic'],baseline_mae=b['mae'],baseline_ic=b['rank_ic'],mae_difference=a['mae']-b['mae'],ic_difference=a['rank_ic']-b['rank_ic'],active_nonbaseline_models=models_used,audit=audit['status']))
 print(pd.DataFrame(comparison).to_string(index=False));print(status);print('Report:',out/'RESULTS.html')
if __name__=='__main__':main()
