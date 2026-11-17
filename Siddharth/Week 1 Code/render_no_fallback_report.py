from pathlib import Path
import json,html
import pandas as pd
import targeted_challenger_no_fallback as m
def table(f):return f.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')
def main():
 out=m.OUT;c=json.loads((out/'COMPLETE.json').read_text());d=json.loads((out/'SELECTIONS.json').read_text())
 scores=pd.read_csv(out/'scores.csv');fs=pd.read_csv(out/'features.csv')
 windows=[]
 for a in d:
  windows.append(dict(issue=a['issue'][:10],feature_count=len(a['features']),rf_weight=a['weight'],rf_mae=a['test_rf']['mae'],rf_ic=a['test_rf']['rank_ic'],hybrid_mae=a['test_hybrid']['mae'],hybrid_ic=a['test_hybrid']['rank_ic'],baseline_mae=a['baseline']['mae'],baseline_ic=a['baseline']['rank_ic']))
 pd.DataFrame(windows).to_csv(out/'window_scores.csv',index=False)
 union=sorted(set(fs.feature))
 lines=['Exact features by outer prediction window; missing-value indicators are additional design inputs.','Model:128-tree RandomForestRegressor, absolute_error, max_depth 3,min_samples_leaf 128,max_samples4096,seed 48.','Hybrid: refitted two-feature median regression + expanded RF. No baseline fallback.','']
 for a in d:
  lines += [a['issue']+' RF weight='+str(a['weight'])]+[str(i)+'. '+f for i,f in enumerate(a['features'],1)]+['']
 lines+=['Unique union ('+str(len(union))+'):']+union+['','DELTA24H: current value minus value24h earlier.','Z24H: (current-past24h mean)/past24h standard deviation, clipped[-10,10].','PAIR_PRODUCT: multiply source feature by original 15m liquidation historical-active-rate control.','ABS: absolute value. Missing-value flags and training-only median imputation/standardization applied.']
 (out/'EXACT_FEATURES.txt').write_text('\n'.join(lines),encoding='utf-8')
 body=f"""<!doctype html><html><head><meta charset="utf-8"><title>New-feature challenger results</title><style>body{{font:16px system-ui;max-width:1300px;margin:40px auto;padding:20px;color:#173348}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}th{{background:#edf2f5}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body>
<h1>48-hour new-feature challengers: their own test results</h1>
<p>These forecasts always use the selected expanded-feature RF. They are not replaced by baseline forecasts when validation is worse. Four new outer RF fits were made; all 18,481 original test timestamps are retained.</p>
<h2>Models</h2><p>Expanded RF: 128 trees, absolute-error splitting, depth 3, minimum leaf 128, bootstrap sample 4096, seed 48. Expanded hybrid: this RF blended with the refitted original 5m/15m liquidation median-regression model. Its RF weights are 50%, 10%, 10%, 10% across the four outer windows. The standalone RF uses the feature set selected for its hybrid; it was not separately optimized.</p>
<h2>Overall test metrics</h2>{table(scores[scores.split=='test'])}
<h2>Training, validation and last-three-month metrics</h2>{table(scores[scores.split!='test'])}
<p>Training observations repeat across expanding fits. Validation is used to select the feature pack and weight. The test population has unique timestamps. These historical periods have been examined before; this is retrospective evaluation, not fresh confirmation.</p>
<h2>Window-by-window comparison</h2>{table(pd.DataFrame(windows))}
<h2>Exact features used in each window</h2>{table(fs)}
<p>There are {len(union)} distinct predictor columns across windows, including two original controls and engineered variants. Each predictor also gets a missing-value flag. Median regression always uses only the original two controls.</p>
<h2>Transformations</h2><p>DELTA24H subtracts the value 24 hours earlier. Z24H standardizes against the preceding 24 hours and clips to [-10,10]. PAIR_PRODUCT multiplies the source value by the original 15m liquidation control. ABS takes absolute value. Missing-value imputation and final scaling use training data only.</p>
<h2>Selection and verification</h2><p>Each window selects the lowest inner-validation MAE expanded-feature hybrid, breaking ties by higher IC, fewer columns, pack and weight. The preceding 3 calendar months form inner validation; 48h labels must mature strictly before the next boundary. The chosen feature pack is refitted using all eligible earlier history. No outer outcome chooses the configuration. Models were reopened and predictions replayed; preprocessing, purges, original validation scores and test alignment passed verification.</p>
<p>This standalone comparison was requested after the earlier gated procedure rejected all candidates. That earlier 181.525993 result was the unchanged baseline fallback; the scores above are the expanded-feature models' own results.</p>
<p>Source: Code/targeted_challenger_no_fallback.py. Exact feature inventory: EXACT_FEATURES.txt. Scores: scores.csv. Fitted models and per-window selections are saved alongside this report.</p></body></html>"""
 (out/'RESULTS.html').write_text(body,encoding='utf-8')
 print(scores[scores.split=='test'].to_string(index=False));print('Unique predictors',len(union))
if __name__=='__main__':main()
