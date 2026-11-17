"""Render document-driven48h results, paired intervals and observable-state diagnostics."""
import json,html,re
from pathlib import Path
import numpy as np
import pandas as pd
import nested_ic_rf as old
from render_tree_magnitude_report import interval,table,ic
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/tree_document_methods_v2'
def main():
 p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
 scores=pd.read_csv(OUT/'scores.csv');sel=json.loads((OUT/'selections.json').read_text())
 audit=json.loads((OUT/'AUDIT.json').read_text())
 ci=interval(p,'report_hybrid')
 (OUT/'PAIRED_UNCERTAINTY.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
 monthly=[]
 for month,g in p.groupby(p.index.strftime('%Y-%m')):
  for col in ['prediction','report_hybrid']:
   monthly.append(dict(month=month,model=col,mae=float(abs(g[col]-g.truth).mean()),rank_ic=ic(g[col],g.truth),rows=len(g)))
 pd.DataFrame(monthly).to_csv(OUT/'monthly_scores.csv',index=False)
 raw,y,dev,report=old.load();states=[]
 statecols=['F4Z__volatility__realized_variance_level_rz__1h__value','F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value']
 for dec in sel:
  issue=pd.Timestamp(dec['issue']);window=dec['model'].split('_')[1]
  mask=y.label_end<issue
  if window!='exp':mask&=y.index>=issue-pd.DateOffset(months=int(window))
  tr=y.index[mask];g=p[p.issue==str(issue)]
  for feature in statecols:
   threshold=float(raw.loc[tr,feature].median());a=raw.loc[g.index,feature]
   labels=pd.Series(np.where(a.isna(),'missing',np.where(a>threshold,'high','low')),index=g.index)
   for state in ['low','high','missing']:
    f=g.loc[labels==state]
    if not len(f):continue
    states.append(dict(issue=str(issue),feature=feature,training_median_threshold=threshold,state=state,rows=len(f),model_mae=float(abs(f.prediction-f.truth).mean()),baseline_mae=float(abs(f.report_hybrid-f.truth).mean()),model_ic=ic(f.prediction,f.truth),baseline_ic=ic(f.report_hybrid,f.truth)))
 pd.DataFrame(states).to_csv(OUT/'observable_state_diagnostics.csv',index=False)
 selected=pd.DataFrame([dict(issue=z['issue'][:10],model=z['model'],trees=z['selected_iterations'],features='; '.join(z['columns'])) for z in sel])
 selected.to_csv(OUT/'selected_features.csv',index=False)
 result=scores[scores.split=='test'].iloc[0];base=scores[scores.split=='report_hybrid'].iloc[0]
 win=bool(result.mae<base.mae and result.rank_ic>base.rank_ic and result.mae<181.5238678328609)
 status='Joint point-estimate improvement; fresh confirmation still required' if win else 'Did not beat the 48-hour baseline on both MAE and IC'
 fitcount=json.loads((OUT/'COMPLETE.json').read_text())['predictive_fit_count']
 text=f"""<!doctype html><html><head><meta charset="utf-8"><title>Document-driven tree model results</title><style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;color:#183247}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ccd4db;text-align:left;overflow-wrap:anywhere}}th{{background:#eaf0f5}}pre{{white-space:pre-wrap;background:#f3f6f8;padding:16px}}.result{{padding:20px;background:#edf2f6;border-left:5px solid #355d7c}}details{{margin:20px 0}}</style></head><body>
<h1>Document-driven tree regression:48-hour follow-up</h1>
<p>Executed3November2026. Separate experiment after the user requested closer application ofD6/D8. The first experiment remains unchanged.</p>
<div class="result"><b>{status}.</b><p>Test MAE{result.mae:.6f}bps; SpearmanIC{result.rank_ic:.6f}. Matched reconstructedreference MAE{base.mae:.6f}, IC{base.rank_ic:.6f}. OriginalPDF MAE181.523868; its exact forecast artifact/IC were unavailable.</p></div>
<h2>Overall scores</h2>{table(scores)}
<p>All18,481 full-period testtimestamps are retained. Training is resubstitution performance; pooled training/validation windows repeat timestamps, so rowcounts and unique timestampcounts are shown. This is a quarterly model-selection procedure, not one fixed model or a blend of three models.</p>
<h2>Specific report methods implemented</h2>
<table><tr><th>Source (physical PDF page)</th><th>Implementation</th></tr>
<tr><td>D6p13:chronological early stopping</td><td>Each training period reserves its last30days for early stopping. The earlier fit is purged by the48h label end. Select up to300trees with patience30, then refit all mature training rows at that tree count.</td></tr>
<tr><td>D6p13:depth, shrinkage and leaf constraints</td><td>CatBoostMAE andLightGBML1, depth3, learningrate0.025, L2=100; LightGBM maximum8leaves and minimum512samples perleaf.</td></tr>
<tr><td>D8p23:select training lookback using validation</td><td>Compare6months,12months and expanding history inside two chronological3month innerfolds.</td></tr>
<tr><td>D8p19:compact versus broader feature sets</td><td>Compare a compact liquidation control with a screened broader liquidation/flow/volatility pool. Broad admission requires lowerMAE, higherIC and monthly stability versus the best compact configuration.</td></tr>
<tr><td>D8pp18ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“19:redundancy checks</td><td>Pairwise Spearman threshold0.95, followed by iterativeVIF<=3.3. The document mentions both5.0 and3.3; this experiment fixes3.3 before its test evaluation. This is a tested recipe, not a universal requirement for trees.</td></tr>
<tr><td>D8p22:observable regime diagnostics</td><td>Report errors in high/low volatility and liquidation states using training-only median cutoffs, including a missing-data state. These diagnostics do not remove testrows or select the model.</td></tr></table>
<h2>Filtering and engineering</h2><p>The same eight-test pipeline is retained:missing values, variance, correlation, VIF, Spearman, mutual information, KS/lift, permutation and add/drop evaluation. Candidate sources remain the magnitude-only liquidation/flow/volatility tables plus the original liquidation controls. Prior24hmeans,1hchanges, standardized innovations, signedlogs and bounded cross-family products are tested. Screens are fitted before the first innerfold. The broader pool is capped at8features after iterative redundancy removal. Conditional feature usefulness is assessed through block permutations and drop-column refits; a feature is removed when its removal passes the joint validation improvement gate.</p>
<h2>Selected models and exact features</h2>{table(selected)}
<p>There were12pre-registered recipes per issuance and{fitcount}predictive fits, including early-stopping probes and refits. No non-tree model or hybrid reference is used to generate new predictions. The reference is a comparator only.</p>
<h2>What these results establish</h2><p>The six-horizon experiment separately supports1/4/24/48/72/120hours. This additional document-method experiment was executed only at48hours. Neither shorter-horizon MAE nor a favorable market-state subset can be substituted for the full48-hour objective. The historical evaluation period was already exposed by earlier experiments; even a point-estimate improvement would require fresh data for confirmation. These PDFs are AI-generated research references, not guarantees of performance.</p>
<h2>Audit</h2><p>{html.escape(audit['status'])}:training-window and early-stop label purges, medians, tree-only artifacts, feature pools, validation scores, ablation/permutation decisions and predictions replayed. The stored stopping counts and data boundaries are checked; stopping trajectories were not retrained.</p>
<details><summary>Monthly scores</summary>{table(pd.DataFrame(monthly))}</details>
<details><summary>Observable-state diagnostics (allrows accounted for)</summary>{table(pd.DataFrame(states))}</details>
<details><summary>Nominal14-day paired bootstrap intervals</summary><pre>{html.escape(json.dumps(ci,indent=2))}</pre></details>
<h2>Files</h2><pre>{html.escape(str(OUT))}
Code/tree_document_methods.py
Code/audit_tree_document_methods.py
Code/predict_tree_magnitude.py
Code/render_tree_document_report.py</pre>
<p><a href="../tree_magnitude_filter_v1/RESULTS.html">Six-horizon experiment and all eight feature tests</a></p>
</body></html>"""
 # Readability for narrative numeric tokens; table values remain exact.
 for before_text,after_text in {
  'Executed3November2026':'Executed 3 November 2026', 'ofD6/D8':'of D6/D8', 'Test MAE':'Test MAE ', 'bps; SpearmanIC':' bps; Spearman IC ', 'Matched reconstructedreference MAE':'Matched reconstructed reference MAE ', 'OriginalPDF MAE181.523868':'Original PDF MAE 181.523868', 'All18,481':'All 18,481', 'testtimestamps':'test timestamps', 'timestampcounts':'timestamp counts', 'rowcounts':'row counts', 'D6p13':'D6 p13', 'D8p23':'D8 p23', 'D8p19':'D8 p19', 'D8pp18Ã¢â‚¬â€œ19':'D8 pp18Ã¢â‚¬â€œ19', 'D8p22':'D8 p22', 'last30days':'last 30 days', 'the48h':'the 48h', 'to300trees':'to 300 trees', 'patience30':'patience 30', 'CatBoostMAE':'CatBoost MAE', 'LightGBML1':'LightGBM L1', 'depth3':'depth 3', 'learningrate0.025':'learning rate 0.025', 'maximum8leaves':'maximum 8 leaves', 'minimum512samples perleaf':'minimum 512 samples per leaf', 'Compare6months,12months':'Compare 6 months, 12 months', 'three3month':'three 3-month', 'chronological3month innerfolds':'chronological 3-month inner folds', 'lowerMAE':'lower MAE', 'higherIC':'higher IC', 'threshold0.95':'threshold 0.95', 'iterativeVIF<=3.3':'iterative VIF &le; 3.3', 'both5.0 and3.3':'both 5.0 and 3.3', 'fixes3.3':'fixes 3.3', 'testrows':'test rows', 'tests:missing':'tests: missing', 'Prior24hmeans,1hchanges':'Prior 24h means, 1h changes', 'at8features':'at 8 features', 'There were12pre-registered':'There were 12 pre-registered', 'and{fitcount}predictive':'and {fitcount} predictive', 'supports1/4/24/48/72/120hours':'supports 1/4/24/48/72/120 hours', 'at48hours':'at 48 hours', 'full48-hour':'full 48-hour', 'allrows':'all rows', 'Nominal14-day':'Nominal 14-day', 'innerfold':'inner fold', 'signedlogs':'signed logs'
 }.items():text=text.replace(before_text,after_text)
 text=re.sub(r'and(\d+)predictive',r'and \1 predictive',text)
 (OUT/'RESULTS.html').write_text(text,encoding='utf-8')
 (OUT/'RESULT_SUMMARY.json').write_text(json.dumps(dict(status=status,beats_both_point_estimates=win,mae=float(result.mae),rank_ic=float(result.rank_ic),reference_mae=float(base.mae),reference_ic=float(base.rank_ic),audit=audit['status']),indent=2),encoding='utf-8')
 print(scores.to_string(index=False));print(status);print('Report:',OUT/'RESULTS.html')
if __name__=='__main__':main()
