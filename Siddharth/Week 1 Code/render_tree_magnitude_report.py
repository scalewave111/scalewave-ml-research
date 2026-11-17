"""Render audited results and paired dependent-sample uncertainty without fitting."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/tree_magnitude_filter_v1'
def ic(a,b):
 return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else 0.
def interval(p,reference,repeats=500):
 y=p.truth.to_numpy();a=p.prediction.to_numpy();b=p[reference].to_numpy()
 ids=((p.index-p.index[0]).total_seconds()//(14*86400)).astype(int)
 blocks=[np.flatnonzero(ids==z) for z in np.unique(ids)]
 rng=np.random.default_rng(48);values=[]
 for _ in range(repeats):
  ix=np.concatenate([blocks[z] for z in rng.integers(0,len(blocks),len(blocks))])
  values.append([float(np.mean(abs(a[ix]-y[ix])-abs(b[ix]-y[ix]))),ic(a[ix],y[ix])-ic(b[ix],y[ix])])
 lo,hi=np.quantile(values,[.025,.975],axis=0)
 return dict(reference=reference,block_days=14,replicates=repeats,mae_difference=float(np.mean(abs(a-y)-abs(b-y))),mae_interval95=[float(lo[0]),float(hi[0])],ic_difference=ic(a,y)-ic(b,y),ic_interval95=[float(lo[1]),float(hi[1])],interpretation='Nominal retrospective paired block interval; not adjusted for prior model searches or source-table selection')
def table(frame):
 return frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.6f}',border=0)
def main():
 complete=sorted(OUT.glob('*h/COMPLETE.json'),key=lambda p:int(p.parent.name[:-1]))
 assert len(complete)==6
 scores=pd.concat([pd.read_csv(p.parent/'scores.csv') for p in complete],ignore_index=True)
 monthly=pd.concat([pd.read_csv(p.parent/'monthly_scores.csv') for p in complete],ignore_index=True)
 scores.to_csv(OUT/'scores_all_horizons.csv',index=False)
 monthly.to_csv(OUT/'monthly_scores_all_horizons.csv',index=False)
 comparisons=[];selections=[];intervals=[];diagnostics=[];testlast=[]
 for file in complete:
  h=int(file.parent.name[:-1]);folder=file.parent
  p=pd.read_csv(folder/'predictions.csv',index_col=0,parse_dates=True)
  model=scores[(scores.horizon_hours==h)&(scores.split=='test')].iloc[0]
  rf=scores[(scores.horizon_hours==h)&(scores.split=='test_rf_baseline')].iloc[0]
  comparisons.append(dict(horizon_hours=h,test_rows=len(p),model_MAE=model.mae,RF_control_MAE=rf.mae,model_IC=model.rank_ic,RF_control_IC=rf.rank_ic,beats_RF_both=bool(model.mae<rf.mae and model.rank_ic>rf.rank_ic)))
  for ref in ['rf_baseline']+(['report_hybrid'] if h==48 else []):
   intervals.append(dict(horizon_hours=h,**interval(p,ref)))
  for sel in json.loads((folder/'selections.json').read_text()):
   selections.append(dict(horizon_hours=h,issue=sel['issue'][:10],model=sel['model'],features=len(sel['columns']),columns='; '.join(sel['columns'])))
  f=pd.read_csv(folder/'feature_tests.csv')
  for (stage,reason),g in f.groupby(['stage','reason']):
   diagnostics.append(dict(horizon_hours=h,stage=stage,reason=reason,count=len(g)))
  for issue,g in p.groupby('issue'):
   diagnostics.append(dict(horizon_hours=h,stage='outer_calibration',reason=issue,count=len(g),target_median=float(g.truth.median()),prediction_median=float(g.prediction.median()),mae=float(abs(g.prediction-g.truth).mean()),ic=ic(g.prediction,g.truth)))
  last=p.loc[p.index>=pd.Timestamp('2024-11-01',tz='UTC')]
  for name in ['prediction','rf_baseline']+(['report_hybrid'] if h==48 else []):
   testlast.append(dict(horizon_hours=h,model=name,rows=len(last),mae=float(abs(last[name]-last.truth).mean()),ic=ic(last[name],last.truth)))
 pd.DataFrame(comparisons).to_csv(OUT/'horizon_comparison.csv',index=False)
 pd.DataFrame(selections).to_csv(OUT/'selected_features_all_horizons.csv',index=False)
 pd.DataFrame(diagnostics).to_csv(OUT/'filter_and_calibration_summary.csv',index=False)
 pd.DataFrame(testlast).to_csv(OUT/'last_three_calendar_months.csv',index=False)
 (OUT/'PAIRED_UNCERTAINTY.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
 audit=json.loads((OUT/'AUDIT.json').read_text())
 sources=json.loads((OUT/'SOURCES.json').read_text())
 full48=scores[(scores.horizon_hours==48)&scores.split.isin(['train','validation','test','test_rf_baseline','test_report_hybrid'])]
 bits=['<!doctype html><html><head><meta charset="utf-8"><title>Tree-only magnitude filtering results</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#182b39}h1,h2{color:#173a54}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}th,td{padding:8px;border-bottom:1px solid #ccd6df;text-align:left;overflow-wrap:anywhere}th{background:#eaf1f5}pre{white-space:pre-wrap;background:#f3f6f8;padding:18px}.fail{border-left:5px solid #b63d36;padding:18px;background:#fff1ef}code{overflow-wrap:anywhere}details{margin:16px 0}</style></head><body>']
 bits+=['<h1>Tree-only magnitude models: eight-test filtering</h1><p>Executed 3 November 2026. Six separately trained horizons: 1, 4, 24, 48, 72 and 120 hours. All predictive estimators are tree regressors.</p>',
 '<div class="fail"><b>The 48-hour objective was not achieved.</b> Test MAE 182.738986 bps and Spearman IC 0.014126 are worse than the matched reconstructed hybrid baseline: MAE 181.525993 and IC 0.077390. The original PDF reports MAE 181.523868; its exact prediction artifact and IC were unavailable. No model has been promoted.</div>',
 '<h2>Overall 48-hour scores</h2>',table(full48),
 '<p>Training is resubstitution performance. Training/validation totals pool predictions from overlapping issuance windows; their unique timestamp counts are shown separately. Test timestamps are unique. IC means Spearman rank correlation; Pearson is reported separately. Scores belong to the pre-defined model-selection procedure, which may select a different tree family each quarter.</p>',
 '<h2>All horizons, matched tree control</h2>',table(pd.DataFrame(comparisons)),
 '<p>The RF control is a new 48-tree, depth-3, absolute-error forest using the two original liquidation features, refitted under the same walk-forward schedule at each horizon. It is not the historical 183.749584 RF result. The 181.523868 target applies only to the 48-hour target and full evaluation period, not to other horizons.</p>',
 '<h2>How features were filtered</h2><ol>',
 '<li><b>Missing values:</b> at least 70% coverage and 1,000 finite training observations. Median imputation is fitted separately on each training fold.</li>',
 '<li><b>Variance:</b> at least three unique values, largest point mass below 99.5%, nonzero 1st–99th percentile spread; variance is recorded.</li>',
 '<li><b>Correlation:</b> absolute Spearman correlation of 0.95 or above identifies redundant candidates. The representative uses pre-inner evidence.</li>',
 '<li><b>VIF:</b> estimated after training-only median imputation and standardization, with an intercept. Values above 10 are flagged; VIF alone is not a tree-feature rejection rule.</li>',
 '<li><b>Spearman:</b> absolute feature–target correlation of at least 0.02 is one route through the predictive screen, not mandatory for nonlinear signals.</li>',
 '<li><b>Mutual information:</b> estimated on observations spaced at least one horizon apart; compared against three circular-shift nulls. Diagnostic, not a calibrated significance test.</li>',
 '<li><b>KS and lift:</b> KS compares feature distributions in the top and bottom target quartiles. Lift is the maximum/minimum median target across feature quintiles, not classification precision lift. Thresholds are 0.10 KS and 1.10 lift. Evidence must persist in at least half of the latest four training quarters.</li>',
 '<li><b>Permutation and ablation:</b> add one candidate to the current tree model and refit on each inner fold. Require lower MAE, higher IC, four of six monthly MAE wins, positive mean monthly IC gain, and positive median importance across three seven-day block permutations. Retained features also undergo drop-column refits; their final permutation diagnostics are saved.</li></ol>',
 '<p>The magnitude-only PDF sections contain 38 unique mapped features, of which 23 are liquidation, flow or volatility. No direction-table values or funding/basis/price features enter the new pool. After screening, the bounded search considers at most two raw candidates per family and eight engineered candidates per outer fold. At most two raw and two engineered additions can be admitted. The original two liquidation controls start the model and are subject to final drop ablation. Thus this is a bounded candidate search, not proof of the globally best feature subset.</p>',
 '<h2>Feature engineering and models</h2><p>Candidate engineering: strictly prior 24-hour mean, one-hour difference, prior-window standardized innovation, signed log, and bounded cross-family products. Engineered candidates face the same screen and validation admission. Signed-log transforms need not help trees; results, not the transformation name, determine admission.</p>',
 '<p>Models: LightGBM L1 with depth 1 or 2; CatBoost MAE with depth 2; Random Forest with absolute-error splits and depth 3. The base configuration is selected on inner-validation MAE with IC as tie-breaker. Feature additions require joint MAE/IC improvement. There is no non-tree predictor, residual median-correction model or hybrid fallback in this experiment.</p>',
 '<h2>Walk-forward design and limitations</h2><p>Quarterly outer issuances: January, April, July and October 2024. The final model forecasts through 22 January 2025, subject to horizon label availability. Training expands from April 2022. Each issuance has two preceding three-month inner walk-forward folds. Screening data ends before the first inner fold; every training label must end strictly before its model issuance. Validation labels must mature before the outer issuance. No outer-test scores drive feature or model selection.</p>',
 '<p>The target is absolute endpoint log return in basis points, not maximum excursion. Canonical five-minute returns reproduce all original 48-hour labels to numerical precision. Each other horizon has its own labels, maturity purge, feature decisions and models. A common pipeline does not guarantee one feature set works for every horizon.</p>',
 '<p>This historical period has already been inspected in earlier experiments, and the published candidate tables may reflect broader historical analysis. These are retrospective comparisons, not a fresh untouched holdout. Nominal 14-day paired block-bootstrap intervals are saved; they are not corrected for previous model searches. Failing to beat the baseline is not repaired by relabeling another period or excluding difficult timestamps.</p>',
 '<h2>What the reference PDFs contributed</h2><p>D6, physical PDF pages 11–15: threshold interactions, tree-depth/leaf constraints, shrinkage, subsampling, and caution about importance under correlated predictors. D8, pages 14–17: dependent-sample inference, target-specific modeling, regime drift and incremental experimentation. Its excursion discussion concerns a different target. These AI-generated reports provide experiment ideas, not verified evidence that any technique will beat this baseline.</p>',
 '<h2>Verification</h2><p>Audit status: <b>'+html.escape(audit['status'])+'</b>. Label maturity, training-only imputation, feature causality, filter recomputation, inner admission decisions, saved-model prediction replay and scores are checked without new predictive fits.</p>',
 '<details><summary>Selected models and exact features, every horizon and issuance</summary>',table(pd.DataFrame(selections)),'</details>',
 '<details><summary>Overall train, validation and test scores, all horizons</summary>',table(scores),'</details>',
 '<details><summary>Monthly test scores</summary>',table(monthly),'</details>',
 '<details><summary>Last three calendar months: November 2024–January 2025 (partial January)</summary>',table(pd.DataFrame(testlast)),'</details>',
 '<details><summary>Paired uncertainty</summary><pre>'+html.escape(json.dumps(intervals,indent=2))+'</pre></details>',
 '<h2>Files and reproduction</h2><pre>'+html.escape(str(OUT))+'\nCode/tree_magnitude_filter.py\nCode/audit_tree_magnitude_filter.py\nCode/predict_tree_magnitude.py\n\npython Code/tree_magnitude_filter.py\npython Code/audit_tree_magnitude_filter.py\npython Code/render_tree_magnitude_report.py\n\npython Code/predict_tree_magnitude.py --model Experiments/tree_magnitude_filter_v1/48h/20241001/selected.joblib --features YOUR_FEATURES.parquet --start YOUR_UTC_START --output predictions.csv</pre>',
 '<p>Inference requires the same canonical source columns and at least 24 hours of preceding feature history. Saved models are historical research artifacts; they have not been freshly trained for current live trading.</p>',
 '<details><summary>Source paths and inspection scope</summary><pre>'+html.escape(json.dumps(sources,indent=2))+'</pre></details></body></html>']
 (OUT/'RESULTS.html').write_text('\n'.join(bits),encoding='utf-8')
 print(scores.to_string(index=False))
 print('Report saved:',OUT/'RESULTS.html')
if __name__=='__main__':main()
