"""Render saved experiment results without training."""
from pathlib import Path
import json,sys,html
import pandas as pd
out=Path(sys.argv[1]);c=json.loads((out/'CONTRACT.json').read_text())
scores=pd.read_csv(out/'scores.csv');monthly=pd.read_csv(out/'monthly_scores.csv')
result=json.loads((out/'RESULT.json').read_text())
selections=[json.loads(p.read_text()) for p in sorted((out/'outer').glob('*_selection.json'))]
def table(f):return f.to_html(index=False,border=0,float_format=lambda v:f'{v:.6f}',na_rep='undefined')
test=scores[scores.split=='test'].sort_values('mae')
base=test[test.model=='report_hybrid'];chosen=test[test.model=='selected']
claim='Compare the same-row test scores below.'
if len(base):
 a=chosen.iloc[0];b=base.iloc[0]
 claim=f"Selected procedure versus reconstructed report hybrid: MAE change {a.mae-b.mae:+.6f} log bps; Spearman IC change {a.rank_ic-b.rank_ic:+.6f}. Lower MAE and higher IC are better."
baseline=json.loads((out/'report_baseline_reproduction.json').read_text()) if (out/'report_baseline_reproduction.json').exists() else {}
page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{c['horizon_hours']}h nested walk-forward results</title>
<style>body{{font:16px/1.5 Segoe UI,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#192b39;background:#fafcfd}}h1,h2{{color:#123f59}}table{{border-collapse:collapse;width:100%;background:white;margin:18px 0 32px;font-size:14px}}td,th{{padding:9px 12px;text-align:right;border-bottom:1px solid #dae3e8}}td:first-child,th:first-child{{text-align:left}}th{{background:#e8f0f5}}.note{{padding:16px;background:#e8f0f5;border-left:4px solid #26718c}}code{{font-size:13px}}</style></head><body>
<h1>{c['horizon_hours']}-hour magnitude: nested walk-forward results</h1>
<p>Outer evaluation starts {c['outer_months'][0]}; {result['test_rows']} unique observations. Each monthly model uses trailing three-month training, with strict horizon label purging. Earlier three-month inner folds select the recipe and blend. No outer-score retuning.</p>
<p class="note">{claim}</p>
<h2>Overall test comparison</h2>{table(test.drop(columns=['split']))}
<h2>Training and inner validation</h2><p>Training is in-sample. Inner validation is used for selection, so it is not an independent final test. Both pools contain repeated timestamps across different outer decisions. Training: {result['training_model_rows']} model-row observations / {result['training_unique_timestamps']} unique timestamps. Validation: {result['validation_model_rows']} model-row observations / {result['validation_unique_timestamps']} unique timestamps.</p>
{table(scores[scores.split!='test'])}
<h2>Selected recipe by month</h2>{table(pd.DataFrame([dict(month=s['outer'],selected=s['selected'],catboost=s['best_cat'],lightgbm=s['best_lgb']) for s in selections]))}
<h2>Monthly results</h2>{table(monthly)}
<h2>Evidence and limitations</h2><p>Historically exposed development data, not a fresh holdout. Correlations are time-series correlations of predicted and observed magnitude. The report comparator is reconstructed from the PDF recipe; original prediction hashes are unavailable. Full-period reconstruction matches report MAE to 0.001 bps: {baseline.get('aggregate_matches_report','not applicable')}. See uncertainty.json for nominal paired 14-day-block intervals; these do not correct the adaptive research history.</p>
<p>Only {c['horizon_hours']}h has real-data scores in this run. The shared pipeline supports separate 1h, 4h, 24h, 72h and 120h runs when canonical prices are supplied. Synthetic target checks do not establish real-data accuracy. See USAGE.json.</p>
<p>Files: <a href="scores.csv">overall scores</a> · <a href="monthly_scores.csv">monthly scores</a> · <a href="VERIFICATION.json">replay verification</a> · <a href="INDEPENDENT_CHECK.json">independent audit</a> · <a href="USAGE.json">usage</a></p>
</body></html>"""
(out/'RESULTS.html').write_text(page,encoding='utf-8')
print('Report rendered from saved results.')
