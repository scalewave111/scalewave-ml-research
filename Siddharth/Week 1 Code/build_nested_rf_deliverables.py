"""Create a review notebook and a self-contained HTML score report from saved outputs."""
from pathlib import Path
import json, html, hashlib
import pandas as pd
import numpy as np
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Experiments/nested_ic_rf_v3'
code_path=ROOT/'Code/nested_ic_rf.py'

def build_notebook():
 n=nbf.v4.new_notebook()
 n.cells=[
 nbf.v4.new_markdown_cell("""# BTC 48-hour magnitude: IC-guided nested Random Forest
This notebook reviews and reproduces the saved experiment. It does not start training by default.

**Outer test:** November 2024, December 2024, and January 1–22, 2025 on the supplied prepared dataset. **Inner selection:** three preceding monthly walk-forward folds before each outer month. Each model trains on its preceding three calendar months; training labels and inner selection labels must mature before the relevant issuance.

All 522 masked F4Z inputs remain candidates. The magnitude tables of the Top 20 PDF supply 38 candidate names, 32 of which satisfy the outer-window admission rules. Fixed causal lags, rolling summaries, cross-window slopes/balances, family summaries, matured magnitude histories, training-only IC screening and feature interactions expand the search. Direction-table rankings are not used.

Training MAE/IC are in-sample diagnostics. Overall training includes overlapping fit windows and therefore repeats some timestamps. Overall test includes each outer timestamp exactly once. Historically exposed data and feature priors remain exposed; this is not an untouched confirmation set."""),
 nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
# Set this to the copied workspace root when running elsewhere.
WORKSPACE = Path.cwd()
if not (WORKSPACE / 'Code/nested_ic_rf.py').exists() and (WORKSPACE.parent / 'Code/nested_ic_rf.py').exists():
    WORKSPACE = WORKSPACE.parent
OUTPUT = WORKSPACE / 'Experiments/nested_ic_rf_v3'
assert (WORKSPACE / 'Code/nested_ic_rf.py').exists(), 'Set WORKSPACE to the folder containing Code, Report, and Dataset.'
"""),
 nbf.v4.new_markdown_cell("""## Overall training and testing scores
MAE and RMSE use original log basis points; lower is better. IC is time-series correlation against magnitude. Rank IC is Spearman correlation, Pearson IC is also shown. High IC does not automatically imply lower MAE.
"""),
 nbf.v4.new_code_cell("""overall = pd.read_csv(OUTPUT / 'overall_training_testing.csv')
display(overall)
display(pd.read_csv(OUTPUT / 'scoreboard.csv'))
"""),
 nbf.v4.new_markdown_cell("## Monthly training, inner-validation, and outer-test results"),
 nbf.v4.new_code_cell("""display(pd.read_csv(OUTPUT / 'train_validation_test_by_month.csv'))
display(pd.read_csv(OUTPUT / 'monthly_mae.csv'))
display(pd.read_csv(OUTPUT / 'monthly_rank_ic.csv'))
"""),
 nbf.v4.new_markdown_cell("## Selected recipe and IC evidence for each outer month"),
 nbf.v4.new_code_cell("""for path in sorted((OUTPUT / 'outer').glob('*_selection.json')):
    selection = json.loads(path.read_text())
    print(selection['outer'], 'standalone:', selection['selected_standalone'],
          'selected:', selection['selected_forecast'], 'hybrid:', selection['selected_is_hybrid'])
    display(pd.DataFrame(selection['scores']).T.sort_values('mae'))
    name = pd.Timestamp(selection['outer']).strftime('%Y-%m') + '__magnitude.csv'
    if (OUTPUT / 'ic' / name).exists():
        display(pd.read_csv(OUTPUT / 'ic' / name).head(20))
"""),
 nbf.v4.new_markdown_cell("## Verification, uncertainty, and fit ledger"),
 nbf.v4.new_code_cell("""print(json.dumps(json.loads((OUTPUT / 'VERIFICATION.json').read_text()), indent=2))
display(pd.DataFrame(json.loads((OUTPUT / 'paired_intervals.json').read_text())))
ledger = pd.DataFrame(json.loads((OUTPUT / 'fit_ledger.json').read_text()))
display(ledger[['id','train_rows','status','seconds']])
"""),
 nbf.v4.new_markdown_cell("""## Reproduce without overwriting evidence
Copy the workspace's Code folder, the magnitude mapping JSON in Report/ic_rf_sources, and the complete nested dataset package. Dependencies are recorded in environment.json. For Colab, mount Drive and set WORKSPACE to the copied folder before executing the configuration cell.

The experiment uses the prepared 48-hour dataset; it does not silently switch to another Parquet or target. The original and optimized Colab notebooks and old results remain unchanged.

The following optional cells replay existing saved models or launch a new bounded run. A new run must use a fresh output folder and will spend CPU time. A different output directory does not make previously inspected test history untouched."""),
 nbf.v4.new_code_cell("""import importlib.util
spec = importlib.util.spec_from_file_location('nested_ic_rf_replay', WORKSPACE / 'Code/nested_ic_rf.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
engine.OUT = OUTPUT
VERIFY_SAVED_MODELS = False
if VERIFY_SAVED_MODELS:
    engine.verify()
"""),
 nbf.v4.new_code_cell("""RUN_NEW_EXPERIMENT = False
NEW_OUTPUT = WORKSPACE / 'Experiments/nested_ic_rf_repeat'
if RUN_NEW_EXPERIMENT:
    assert not NEW_OUTPUT.exists(), 'Choose a new directory; preserve previous evidence.'
    engine.OUT = NEW_OUTPUT
    engine.run()
"""),
 nbf.v4.new_markdown_cell("## Exact training source snapshot\nThe cell below is a string for inspection, not a training command."),
 nbf.v4.new_code_cell("TRAINING_SOURCE = "+('r'+chr(39)*3+code_path.read_text(encoding='utf-8')+chr(39)*3)+"\nprint('Loaded exact source text; no training executed.')")
 ]
 n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
 nbf.validate(n)
 path=ROOT/'Code/BTC_48h_Nested_IC_RandomForest.ipynb'
 nbf.write(n,path)
 for cell in n.cells:
  if cell.cell_type=='code': compile(cell.source,str(path),'exec')
 print('Notebook written and syntax-validated:',path)

def build_report():
 result=json.loads((OUT/'RESULT.json').read_text())
 verify=json.loads((OUT/'VERIFICATION.json').read_text())
 overall=pd.read_csv(OUT/'overall_training_testing.csv')
 monthly=pd.read_csv(OUT/'train_validation_test_by_month.csv')
 gaps=overall.pivot(index='model',columns='split',values='mae').reset_index()
 gaps['test_minus_train_mae']=gaps.test_outer_unique-gaps.train_in_sample_pooled
 gaps.to_csv(OUT/'train_test_gap.csv',index=False)
 scores=pd.read_csv(OUT/'scoreboard.csv')
 uncertainty=pd.DataFrame(json.loads((OUT/'paired_intervals.json').read_text()))
 selected=[]
 for path in sorted((OUT/'outer').glob('*_selection.json')):
  s=json.loads(path.read_text())
  selected.append(dict(month=s['outer'][:7],standalone=s['selected_standalone'],selected=s['selected_forecast'],
                       hybrid=s['selected_is_hybrid'],inner_rows=s['inner_rows']))
 selected=pd.DataFrame(selected)
 train=pd.read_csv(OUT/'training_predictions.csv')
 unique_train=train.decision_time.nunique()
 bymonth=pd.read_csv(OUT/'monthly_mae.csv',index_col=0)
 fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
 show=overall[overall.model=='nested_standalone_rf']
 axes[0].bar(['Training (in-sample)','Testing (walk-forward)'],show.mae,color=['#7897ac','#16756b'])
 axes[0].set_ylabel('MAE (log bps)'); axes[0].set_title('Overall standalone Random Forest')
 for i,v in enumerate(show.mae): axes[0].text(i,v+1,f'{v:.3f}',ha='center')
 bymonth[['nested_standalone_rf','previous_rf','previous_hybrid','frozen_pair']].plot.bar(ax=axes[1],rot=0)
 axes[1].set_ylabel('MAE (log bps)'); axes[1].set_title('Same-row monthly test comparison')
 axes[1].legend(fontsize=8)
 fig.savefig(OUT/'train_test_comparison.png',dpi=160); plt.close(fig)
 def table(df):
  return df.to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0,classes='data',na_rep='—')
 comparisons=result['comparisons']['nested_standalone_rf']
 outcome='The nested standalone RF beats all three historical comparators on these test rows.' if all(v['beats'] for v in comparisons.values()) else 'The nested standalone RF does not beat every historical comparator on these test rows.'
 changes=pd.DataFrame([dict(baseline=k,**v) for k,v in comparisons.items()])
 sources=[
 ('D6 research report','../../Report/D6__DR-Report-by-Alphasense-10-26-2026-14-54%20%283%29.pdf','IC, incremental predictive value and model-family comparisons.'),
 ('D8 research report','../../Report/D8__DR-Report-by-Alphasense-10-28-2026-03-04.pdf','Point-in-time features, redundancy control, purged validation and inner hyperparameter selection.'),
 ('Top 20 magnitude PDF',next((Path.home()/'Downloads').glob('Top 20*MAGNITUDE*.pdf')).as_uri(),'Magnitude tables only; direction-only rankings excluded.'),
 ('Magnitude feature mapping','../../Report/ic_rf_sources/magnitude_feature_mapping.csv','Only magnitude-table candidates from the Top 20 PDF; 38 exact existing column matches.')]
 source_html=''.join(f'<li><a href="{link}">{name}</a>: {desc}</li>' for name,link,desc in sources)
 text=f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Nested IC Random Forest results</title>
 <style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:42px auto;padding:0 24px;color:#18232d}}h1,h2{{line-height:1.2}}h1{{font-size:32px}}h2{{margin-top:36px}}.lead{{font-size:19px}}.note{{background:#edf4f7;padding:16px;border-left:4px solid #16756b}}table{{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px;width:100%}}th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid #dce3e8}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f7}}img{{max-width:100%}}a{{color:#12685f}}code{{background:#eef1f3;padding:2px 4px}}</style>
 <h1>BTC 48-hour magnitude: nested IC-guided Random Forest</h1>
 <p class="lead">{outcome}</p>
 <p>Executed on {result['rows']:,} unique test rows from {html.escape(result['start'])} through {html.escape(result['end'])}. {result['fit_count']} RF fit attempts (including the documented interrupted resource-only restart). Verification: <strong>{verify['status']}</strong>.</p>
 <h2>Overall training and testing scores</h2>
 <p>Primary score: original-scale mean absolute error (MAE), in log basis points. Lower is better. RMSE, Spearman Rank IC and Pearson IC are also shown. Training predictions are in-sample; testing predictions are monthly outer walk-forward forecasts.</p>
 {table(overall)}
 <h3>Train-test MAE gap</h3>{table(gaps)}
 <p>Differences also reflect changing market periods; the gap alone is not a proof of overfitting.</p>
 <p class="note">Overall training pools {len(train):,} model-row predictions covering {unique_train:,} unique timestamps. Three overlapping monthly training windows repeat timestamps; this is a fit diagnostic, not out-of-sample evidence. Overall testing pools each of the {result['rows']:,} test timestamps once. The latest month is partial.</p>
 <img src="train_test_comparison.png" alt="Overall train-test MAE and monthly baseline comparison">
 <h2>Same-row test baselines and improvement</h2>{table(scores)}{table(changes)}
 <p>Yesterday's full-development headline scores cover 18,481 rows and are not the comparison scores above. All historical predictions were rescored on the exact 3,937 outer test rows. The previous hybrid is the fixed 25% old RF plus 75% frozen-pair recipe selected yesterday.</p>
 <h2>Monthly training, inner validation, and outer testing</h2>{table(monthly)}
 <h2>Selected recipes</h2>{table(selected)}
 <p><code>nested_raw_rf</code> is the best uncalibrated standalone RF selected by inner MAE. <code>nested_standalone_rf</code> is the best inner-selected calibrated standalone RF or average of the top three RFs. <code>nested_selected</code> also permits a clearly labeled frozen-pair-plus-RF-residual hybrid. Calibration uses inner predictions only; no outer score selects a recipe.</p>
 <h2>Nested protocol and feature search</h2>
 <p>For each outer month, three preceding monthly validation folds choose the configuration. Each inner and outer model fits the preceding three calendar months, excluding any label whose 48-hour endpoint has not strictly passed at issuance. Inner validation outcomes are additionally excluded until their endpoints precede the outer issuance. Earlier outer-month outcomes may enter later months only after maturation.</p>
 <p>The search starts with all 522 original masked F4Z features. The Top 20 magnitude tables map to 38 distinct columns, of which 32 pass the original coverage and variation admission in each outer training window. Engineering includes absolute values; exact-time 1-hour and 24-hour lags/differences; strictly past 6-hour, 1-day and 7-day means, standard deviations, innovations and standardized innovations; cross-horizon differences/balances; family percentile summaries; calendar cycles; and matured-magnitude histories. No direction-only table rankings or future targets become predictors.</p>
 <p>Per-fit pairwise-complete Spearman IC, monthly sign consistency and IC dispersion rank candidates. Near-duplicate inputs are removed inside training. Variants retain 24, 64, 128 or all admitted inputs. Training-only rank transforms, robust clipping/scaling/log compression, pairwise interactions, raw/log/square-root target variants and a separately labeled residual hybrid are tested. IC guides screening; original-scale inner MAE decides advancement. The search is broad and bounded, not an exhaustive enumeration of every possible transformation.</p>
 <h2>Paired uncertainty</h2>{table(uncertainty)}
 <p>Intervals are nominal 95% paired calendar-block bootstrap intervals for model MAE minus baseline MAE; negative favors the model. Seven- and fourteen-day blocks address dependence descriptively, with few blocks in this short test period. They do not correct historical feature/report selection or establish future trading profit.</p>
 <h2>Verification and reproducibility</h2>
 <pre>{html.escape(json.dumps(verify,indent=2))}</pre>
 <p>Original dataset manifest hashes were checked. Fitted models, preprocessing, IC tables, all attempted fits, inner predictions, frozen outer recipes, training/test predictions and environment versions are saved alongside this report. The notebook is <a href="../../Code/BTC_48h_Nested_IC_RandomForest.ipynb">available here</a>. Run <code>python Code/nested_ic_rf.py verify</code> from the workspace to replay without fitting.</p>
 <h2>Source use and limits</h2><ul>{source_html}</ul>
 <p>The PDFs are research leads, not independently verified institutional claims. Their historical IC rankings are candidate priors, not scores for this 48-hour experiment. The data and historical feature rankings were already exposed. Nested tuning prevents current outer-label leakage but cannot retroactively turn this period into a fresh holdout.</p>
 </html>"""
 (OUT/'RESULTS.html').write_text(text,encoding='utf-8')
 print('HTML report written:',OUT/'RESULTS.html')

if __name__=='__main__':
 build_notebook()
 if (OUT/'VERIFICATION.json').exists(): build_report()
